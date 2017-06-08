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

import matplotlib.pyplot as plt

import sys

# constants
const_lr = 1e-4

tag = 'DL_on_Hand_boundary_soft'

out_dir = '/home/jr0th/github/segmentation/out/' + tag + '/'
tb_log_dir = "/home/jr0th/github/segmentation/tensorboard/" + tag + "/"
chkpt_file = "/home/jr0th/github/segmentation/checkpoints/" + tag + "/checkpoint_{epoch:04d}.hdf5"
csv_log_file = "/home/jr0th/github/segmentation/logs/" + tag + ".csv"

train_dir = "/home/jr0th/github/segmentation/data/BBBC022/training/"
val_dir = "/home/jr0th/github/segmentation/data/BBBC022/validation/"

y = "y_boundary_soft/"

rescale_labels = True
hard = False

epochs = 200

batch_size = 10
steps_per_epoch = 100 * 4 / batch_size

loss = "categorical_crossentropy"

# make sure these number for to the validation set
val_batch_size = 10
val_steps = int(50 * 4 / val_batch_size)

# generator only params
dim1 = 256
dim2 = 256

bit_depth = 8

data_type = 'images'

# build session running on GPU 1
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "3"
session = tf.Session(config = configuration)

# apply session
keras.backend.set_session(session)
    
train_gen = helper.data_provider.single_data_from_images_1d_y(
    train_dir + "x/",
    train_dir + y,
    batch_size,
    bit_depth,
    dim1,
    dim2,
    rescale_labels
)

val_gen = helper.data_provider.single_data_from_images_1d_y(
    val_dir + 'x/',
    val_dir + y,
    val_batch_size,
    bit_depth,
    dim1,
    dim2,
    rescale_labels
)


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
callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=chkpt_file,
    save_weights_only=True,
    save_best_only=False
)
callback_csv = keras.callbacks.CSVLogger(filename=csv_log_file)
callback_splits_and_merges = helper.callbacks.SplitsAndMergesLoggerBoundary(
    data_type,
    val_gen, 
    gen_calls = val_steps,
    log_dir=tb_log_dir
)

callbacks=[callback_model_checkpoint, callback_csv, callback_splits_and_merges]

# TRAIN
statistics = model.fit_generator(
    generator=train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose = 1
)
    
# visualize learning stats
if(hard):
    helper.visualize.visualize_learning_stats_boundary_hard(statistics, out_dir, metrics)
else:
    helper.visualize.visualize_learning_stats_boundary_soft(statistics, out_dir, metrics)
print('Done! :)')