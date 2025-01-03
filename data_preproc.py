import load_data
import pandas as pd

# return DataSet class
data = load_data.read_data_sets(one_hot=True)

# get train data and labels by batch size
BATCH_SIZE = 256
train_x, train_label = data.train.next_batch(BATCH_SIZE)

# get test data
test_x = data.test.data

# get test labels
test_labels = data.test.labels

# get sample number
n_samples = data.train.num_examples