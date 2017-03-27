
import helper.model_builder
import os.path
import skimage.io

import numpy as np

import tensorflow as tf

normalize = False

# build session running on GPU 1
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "2"
session = tf.Session(config = configuration)

out_dir = '/home/jr0th/github/segmentation/experiments/BBBC022_hand/DL_probmap/'
data_dir = '/home/jr0th/github/segmentation/data/BBBC022_hand/all_images/'

# use latest checkpoint
weights_path = '/home/jr0th/github/segmentation/results/BBBC022/0324_sample_size_500/checkpoints/checkpoint.hdf5'

# get images
images_coll = skimage.io.imread_collection(data_dir + '*.png')

# assume that images are all the same shape
dim1 = images_coll[0].shape[0]
dim2 = images_coll[0].shape[1]

# build model and load weights
model = helper.model_builder.get_model_3d_output(dim1, dim2)
model.load_weights(weights_path)

for i in range(len(images_coll)):
    img = images_coll[i]
    filename = os.path.basename(images_coll.files[i])
    filename_wo_ext = os.path.splitext(filename)[0]
    
    print("Processing:", filename)

    scale = skimage.dtype_limits(img)
    
    if(normalize):
        # normalize
        percentile = 99.9
        high = np.percentile(img, percentile)
        low = np.percentile(img, 100-percentile)
        
        img = (img - low) / (high - low) * scale[1]
        img = np.minimum(scale[1], img)

    img = img * 1./scale[1]

    # reshape to have last dimension    
    img = img.reshape(-1, dim1, dim2, 1)
    
    # get model predictions
    y_pred = model.predict(img)

    # remove batch dimension
    y_pred = y_pred.squeeze()
    
    # save
    np.save(out_dir + filename_wo_ext, y_pred)