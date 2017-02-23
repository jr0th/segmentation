import helper.data_provider
import helper.model_builder
import helper.visualize

import skimage.io

import matplotlib.pyplot as plt

out_label = 'pred_generator'
out_dir = '/home/jr0th/github/segmentation/out/'
data_dir = '/home/jr0th/github/segmentation/data/set03/'
weights_path = '/home/jr0th/github/segmentation/results/set03/0223_test/checkpoints/checkpoint.hdf5'
batch_size = 10

[_, _, test_generator, dim1, dim2] = helper.data_provider.data_from_images(data_dir, batch_size=batch_size)
model = helper.model_builder.get_model_3d_output(dim1, dim2)
model.load_weights(weights_path)

(x, y) = next(test_generator)

y_pred = model.predict_on_batch(x)
helper.visualize.visualize(y_pred, x, y, out_dir=out_dir, label=out_label)