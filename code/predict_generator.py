import helper.data_provider
import helper.model_builder
import helper.visualize

import skimage.io

import matplotlib.pyplot as plt

out_label = 'pred_generator'
out_dir = '/home/jr0th/github/segmentation/out/'
data_dir = '/home/jr0th/github/segmentation/data/BBBC022/'
weights_path = '/home/jr0th/github/segmentation/checkpoints/checkpoint.hdf5'
batch_size = 10
bit_depth = 8

# get generator for test data
[_, _, test_generator, dim1, dim2] = helper.data_provider.data_from_images(data_dir, batch_size=batch_size, bit_depth=bit_depth)

# build model and laod weights
model = helper.model_builder.get_model_3d_output(dim1, dim2)
model.load_weights(weights_path)

# get one batch of data from the generator
(test_x, test_y) = next(test_generator)

# get model predictions
y_pred = model.predict_on_batch(test_x)

# visualize them
helper.visualize.visualize(y_pred, test_x, test_y, out_dir=out_dir, label=out_label)