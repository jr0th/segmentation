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

out_dir = "../out/"
tb_log_dir = "../logs/logs_tensorboard/"

train_dir_x = '/home/jr0th/github/segmentation/data/BBBC022_hand_200/training/x'
train_dir_y = '/home/jr0th/github/segmentation/data/BBBC022_hand_200/training/y_label_binary'

val_dir_x = '/home/jr0th/github/segmentation/data/BBBC022_hand_200/validation/x'
val_dir_y = '/home/jr0th/github/segmentation/data/BBBC022_hand_200/validation/y_label_binary'

data_type = "images" # "images" or "array"

nb_epoch = 50
batch_size = 10
nb_batches = int(400 / batch_size) # 100 images, 400 patches

# images and masks are in 8 bit
bit_depth = 8

# SEGMENTATION DATA GENERATOR
file_path = '/home/jr0th/github/segmentation/data/BBBC022_hand_200/all_files_wo_ext.txt'
classes = 3

# make sure these matches number for to the validation set
val_steps = int(200 / batch_size) # 50 images, 200 patches

dim1 = 256
dim2 = 256



# build session running on GPU 1
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "1"
session = tf.Session(config = configuration)

# apply session
keras.backend.set_session(session)
    
# get training generator

#train_gen = helper.data_provider.data_from_images_segmentation(file_path, images_dir, labels_dir, classes, batch_size, dim1, dim2)
train_gen = helper.data_provider.single_data_from_images(train_dir_x, train_dir_y, batch_size, bit_depth, dim1, dim2)
val_gen = helper.data_provider.single_data_from_images(val_dir_x, val_dir_y, batch_size, bit_depth, dim1, dim2)

callback_splits_and_merges = helper.callbacks.SplitsAndMergesLogger(data_type, val_gen, gen_calls = val_steps, log_dir='../logs/logs_tensorboard')
    
# build model
model = helper.model_builder.get_model_3_class(dim1, dim2)
model.summary()
loss = "categorical_crossentropy"
optimizer = keras.optimizers.RMSprop(lr = const_lr)
metrics = [keras.metrics.categorical_accuracy, helper.metrics.recall, helper.metrics.precision]
model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

# CALLBACKS
# save model after each epoch
callback_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath="../checkpoints/checkpoint.hdf5", save_weights_only=True, save_best_only=True)
callback_csv = keras.callbacks.CSVLogger(filename="../logs/log.csv")
callback_tensorboard = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

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
helper.visualize.visualize_learning_stats(statistics, out_dir, metrics)
print('Done! :)')
