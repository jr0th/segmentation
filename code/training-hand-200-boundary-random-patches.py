import keras.backend
import keras.callbacks
import keras.layers
import keras.models
import keras.optimizers

import matplotlib
matplotlib.use('SVG')

import helper.callbacks
import helper.model_builder
import helper.visualize
import helper.objectives
import helper.data_provider
import helper.metrics

import skimage.io
import sklearn.metrics

import scipy.stats
import pandas as pd

import tensorflow as tf
import numpy as np

# constants
const_lr = 1e-4

chkpt_file = "../checkpoints/boundary_4_random_augment/checkpoint_{epoch:04d}.hdf5"
csv_log_file = "../logs/log_boundary_4_random_augment.csv"

out_dir = "../out_boundary_4_random_augment/"
tb_log_dir = "../logs/logs_tensorboard_boundary_4_random_augment/"

train_dir_x = '/home/jr0th/github/segmentation/data/BBBC022_hand_200/training/x_big/'
train_dir_y = '/home/jr0th/github/segmentation/data/BBBC022_hand_200/training/y_big_boundary_4/'

val_dir_x = '/home/jr0th/github/segmentation/data/BBBC022_hand_200/validation/x/'
val_dir_y = '/home/jr0th/github/segmentation/data/BBBC022_hand_200/validation/y_boundary_4/'

data_type = "images" # "images" or "array"

hard = True

nb_epoch = 100 # 100
batch_size = 10
nb_batches = int(400 / batch_size) # 100 images, 400 patches

# images and masks are in 8 bit
bit_depth = 8

# make sure these matches number for to the validation set
val_steps = int(200 / batch_size) # 50 images, 200 patches

dim1 = 256
dim2 = 256

# build session running on a specific GPU
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "2"
session = tf.Session(config = configuration)

# apply session
keras.backend.set_session(session)
    
# get training generator
train_gen = helper.data_provider.random_sample_generator(train_dir_x, train_dir_y, batch_size, bit_depth, dim1, dim2)
val_gen = helper.data_provider.single_data_from_images_1d_y(val_dir_x, val_dir_y, batch_size, bit_depth, dim1, dim2)

# build model
model = helper.model_builder.get_model_1_class(dim1, dim2)
model.summary()
if(hard):
    loss = "binary_crossentropy"
    metrics = [keras.metrics.binary_accuracy, helper.metrics.recall, helper.metrics.precision]
else:
    loss = "mean_squared_error"
    metrics = []
optimizer = keras.optimizers.RMSprop(lr = const_lr)

model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

# CALLBACKS
# save model after each epoch
callback_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=chkpt_file, save_weights_only=True, save_best_only=False)
callback_csv = keras.callbacks.CSVLogger(filename=csv_log_file)
callback_splits_and_merges = helper.callbacks.SplitsAndMergesLoggerBoundary(data_type, val_gen, gen_calls = val_steps, log_dir=tb_log_dir)
callback_tensorboard = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

# Use callback_tensorboard instead of callback_splits_and_merges to speed up the training process
callbacks=[callback_model_checkpoint, callback_csv, callback_splits_and_merges]

statistics = model.fit_generator(
    epochs=nb_epoch,
    steps_per_epoch=nb_batches,
    generator=train_gen,
    validation_data=val_gen,
    validation_steps=val_steps,
    max_q_size=1,
    callbacks=callbacks,
    verbose=1
)
    
# visualize learning stats
helper.visualize.visualize_learning_stats_boundary(statistics, out_dir, metrics)
print('Done! :)')
