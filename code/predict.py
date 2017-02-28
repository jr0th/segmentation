import helper.data_provider
import helper.model_builder
import helper.visualize

import skimage.io

import matplotlib.pyplot as plt

out_label = 'pred'
out_dir = '/home/jr0th/github/segmentation/out/'
data_dir = '/home/jr0th/github/segmentation/data/set02/'
# use latest checkpoint
weights_path = '/home/jr0th/github/segmentation/checkpoints/checkpoint.hdf5'

# get generator for test data
[_, _, _, _, test_x, test_y] = helper.data_provider.data_from_array(data_dir)

dim1 = test_x.shape[1]
dim2 = test_x.shape[2]

# build model and laod weights
model = helper.model_builder.get_model(dim1, dim2)
model.load_weights(weights_path)

# get model predictions
y_pred = model.predict(test_x).reshape((-1, 128, 128, 3))

# visualize them
helper.visualize.visualize(y_pred, test_x, test_y, out_dir=out_dir, label=out_label)