import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import load_data

# Hyperparameters
LEARNING_RATE = 0.0005  
BATCH_SIZE = 32         
EPOCHS = 25
NUM_CLASSES = 3
SEQUENCE_LENGTH = 310
NUM_FEATURES = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_dim=128, num_layers=2, num_classes=3, dropout_prob=0.5, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=bidirectional
        )
        multiplier = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * multiplier, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return F.log_softmax(out, dim=1)
    
def load_lstm_data():
    datasets = load_data.read_data_sets(one_hot=False)
    scaler = StandardScaler()
    train_numerical = scaler.fit_transform(datasets.train.data.reshape(-1, SEQUENCE_LENGTH * NUM_FEATURES))
    test_numerical = scaler.transform(datasets.test.data.reshape(-1, SEQUENCE_LENGTH * NUM_FEATURES))
    train_numerical = train_numerical.reshape(-1, SEQUENCE_LENGTH, NUM_FEATURES)
    test_numerical = test_numerical.reshape(-1, SEQUENCE_LENGTH, NUM_FEATURES)
    train_labels = datasets.train.labels
    test_labels = datasets.test.labels
    return torch.tensor(train_numerical, dtype=torch.float).to(device), torch.tensor(train_labels, dtype=torch.long).to(device), \
           torch.tensor(test_numerical, dtype=torch.float).to(device), torch.tensor(test_labels, dtype=torch.long).to(device)

def train_model():
    datasets = load_data.read_data_sets(one_hot=False)
    scaler = StandardScaler()
    train_numerical = scaler.fit_transform(datasets.train.data.reshape(-1, SEQUENCE_LENGTH * NUM_FEATURES))
    test_numerical = scaler.transform(datasets.test.data.reshape(-1, SEQUENCE_LENGTH * NUM_FEATURES))
    train_numerical = train_numerical.reshape(-1, SEQUENCE_LENGTH, NUM_FEATURES)
    test_numerical = test_numerical.reshape(-1, SEQUENCE_LENGTH, NUM_FEATURES)

    train_labels = torch.tensor(datasets.train.labels, dtype=torch.long).to(device)
    test_labels = torch.tensor(datasets.test.labels, dtype=torch.long).to(device)
    train_numerical = torch.tensor(train_numerical, dtype=torch.float).to(device)
    test_numerical = torch.tensor(test_numerical, dtype=torch.float).to(device)

    train_loader = DataLoader(TensorDataset(train_numerical, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_numerical, test_labels), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMClassifier(input_size=NUM_FEATURES, hidden_dim=128, num_layers=2, num_classes=NUM_CLASSES, bidirectional=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_acc = evaluate_model(model, train_loader)
        test_acc = evaluate_model(model, test_loader)
        scheduler.step(avg_loss)

        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    return model

def evaluate_model(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    trained_model = train_model()
