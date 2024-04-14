import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from imager import convert_to_images  
import load_data  

# Hyperparameters
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_CLASSES = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        out += identity
        return F.relu(out)

class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.conv1_1 = nn.Conv2d(5, 32, kernel_size=5, padding='same')
        self.batch_norm32 = nn.BatchNorm2d(32)
        self.enc1_2 = ResBlock(32, 32, 5)
        self.enc1_3 = ResBlock(32, 32, 5)
        self.enc1_4 = ResBlock(32, 32, 5)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding='same')
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.enc2_2 = ResBlock(64, 64, 5)
        self.enc2_3 = ResBlock(64, 64, 5)
        self.enc2_4 = ResBlock(64, 64, 5)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding='same')
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.enc3_2 = ResBlock(128, 128, 5)
        self.enc3_3 = ResBlock(128, 128, 5)
        self.enc3_4 = ResBlock(128, 128, 5)
        
        # AdaptiveAvgPool2d will pool each feature map to size (1, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # The fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.batch_norm32(self.conv1_1(x)))
        x = self.enc1_2(x)
        x = self.enc1_3(x)
        x = self.enc1_4(x)

        x = F.relu(self.batch_norm64(self.conv2_1(x)))
        x = self.enc2_2(x)
        x = self.enc2_3(x)
        x = self.enc2_4(x)

        x = F.relu(self.batch_norm128(self.conv3_1(x)))
        x = self.enc3_2(x)
        x = self.enc3_3(x)
        x = self.enc3_4(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # Flatten the dimensions for the fully connected layer
        x = self.dropout(x)
        x = self.fc(x)  # Final classification layer
        return F.log_softmax(x, dim=1)


def load_cnn_data():
    datasets = load_data.read_data_sets(one_hot=False)
    train_images, train_labels = convert_to_images(datasets.train.data, datasets.train.labels)
    test_images, test_labels = convert_to_images(datasets.test.data, datasets.test.labels)
    train_images = np.transpose(train_images, (0, 3, 1, 2))
    test_images = np.transpose(test_images, (0, 3, 1, 2))
    return torch.tensor(train_images, dtype=torch.float).to(device), torch.tensor(train_labels, dtype=torch.long).to(device), torch.tensor(test_images, dtype=torch.float).to(device), torch.tensor(test_labels, dtype=torch.long).to(device)

def evaluate_accuracy(data_loader, model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_model():
    train_data, train_labels, test_data, test_labels = load_cnn_data()
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=BATCH_SIZE)
    
    model = CNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_accuracy = evaluate_accuracy(train_loader, model)
        test_accuracy = evaluate_accuracy(test_loader, model)
        scheduler.step()

        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

    return model

if __name__ == "__main__":
    trained_model = train_model()
