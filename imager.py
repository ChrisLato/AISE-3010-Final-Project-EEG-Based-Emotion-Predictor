import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import load_data  
import os

def transform(dataset, weight=1.2):
    """
    Transform a dataset entry into a structured 9x9 image based on differential entropy features.
    """
    im = np.zeros([9, 9])

    im[0, 3:6] = dataset[0:3]
    im[1, [3, 5]] = dataset[3:5]
    im[2] = dataset[5:14]  # Fills the whole second row
    im[3] = dataset[14:23]  # Fills the whole third row
    im[4] = dataset[23:32]  # Fills the whole fourth row
    im[5] = dataset[32:41]  # Fills the whole fifth row
    im[6] = dataset[41:50]  # Fills the whole sixth row
    im[7, 1:8] = dataset[50:57]
    im[8, 2:7] = dataset[57:62]

    # Apply weighting to specified indices
    for i in [2, 3, 4, 5, 6, 7]:
        im[i, [0, 8]] *= weight
    
    return im.reshape(9, 9, 1)

def convert_to_images(data, labels):
    """
    Convert the numerical EEG data into images.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data.T).T
    
    images = []
    for i in range(normalized_data.shape[0]):
        # Concatenate transformed segments to create a multi-channel image representation
        images.append(np.concatenate([transform(normalized_data[i][j:j+62], 1.2) for j in range(0, 310, 62)], axis=2))
    
    return np.array(images), np.array(labels)

def main():
    """
    Main function to load data, transform it to images, shuffle, and save the processed data.
    """
    # Load data using the structure defined in your load_data module
    data = load_data.read_data_sets(one_hot=True)
    
    # Transform training data to images
    train_imgs, train_labels = convert_to_images(data.train.data, data.train.labels)
    # Shuffle the training data
    train_indices = np.random.permutation(len(train_imgs))
    train_imgs, train_labels = train_imgs[train_indices], train_labels[train_indices]
    
    # Transform test data to images
    test_imgs, test_labels = convert_to_images(data.test.data, data.test.labels)
    # Shuffle the test data
    test_indices = np.random.permutation(len(test_imgs))
    test_imgs, test_labels = test_imgs[test_indices], test_labels[test_indices]
    
    # Print the shape of the first few images and labels in the training set
    print("First few images and labels from the training set:")
    for i in range(min(5, len(train_imgs))):  
        print(f"Image {i+1} shape: {train_imgs[i].shape}, Label: {train_labels[i]}")
    print(train_imgs.shape)
    
    # Print the shape of the first few images and labels in the test set
    print("\nFirst few images and labels from the test set:")
    for i in range(min(5, len(test_imgs))):  
        print(f"Image {i+1} shape: {test_imgs[i].shape}, Label: {test_labels[i]}")
    print(test_imgs.shape)
    
    # Check if the 'data' directory exists, create it if it doesn't
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    # Save the transformed and shuffled training data
    with open('./data/train_images.pkl', 'wb') as f:
        pickle.dump({'data': train_imgs, 'label': train_labels}, f)

    # Save the transformed and shuffled test data
    with open('./data/test_images.pkl', 'wb') as f:
        pickle.dump({'data': test_imgs, 'label': test_labels}, f)
    
if __name__ == "__main__":
    main()