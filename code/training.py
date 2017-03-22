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

# constants
const_lr = 1e-4

data_dir = "/home/jr0th/github/segmentation/data/BBBC022_10/"
data_type = "images" # "images" or "array"

out_dir = "../out/"
tb_log_dir = "../logs/logs_tensorboard/"

nb_epoch = 10
batch_size = 2

# generator only params
if(data_type == "images"):
    dim1 = 256
    dim2 = 256

    nb_batches = 1
    bit_depth = 8

# build session running on GPU 1
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "2"
session = tf.Session(config = configuration)

# apply session
keras.backend.set_session(session)

if data_type == "array":
    
    [training_x, training_y, validation_x, validation_y, _, _] = helper.data_provider.data_from_array(data_dir)
    
    # reshape y to fit network output
    training_y_vec = training_y.reshape((-1, 128 * 128, 3))
    test_y_vec = test_y.reshape((-1,128 * 128, 3))
    validation_y_vec = validation_y.reshape((-1,128 * 128, 3))

    print(training_y_vec.shape)
    print(test_y_vec.shape)
    print(validation_y_vec.shape)

    print(np.unique(training_y_vec))
    print(np.unique(test_y_vec))
    print(np.unique(validation_y_vec))

    dim1 = training_x.shape[1]
    dim2 = training_x.shape[2]
    
    # build model
    model = helper.model_builder.get_model_3d_output(dim1, dim2)
    loss = "categorical_crossentropy"

    callback_splits_and_merges = helper.callbacks.SplitsAndMergesLogger(data_type, [validation_x, validation_y], tb_log_dir)
    
elif data_type == "images":
    
    training_gen = helper.data_provider.single_data_from_images(data_dir + "training/", batch_size, bit_depth, dim1, dim2)
    validation_gen = helper.data_provider.single_data_from_images(data_dir + "validation/", batch_size, bit_depth, dim1, dim2)

    model = helper.model_builder.get_model_3d_output(dim1, dim2)
    loss = "categorical_crossentropy"
    
    callback_splits_and_merges = helper.callbacks.SplitsAndMergesLogger(data_type = data_type, data = validation_gen, log_dir='../logs/logs_tensorboard')
    

# TODO include precision and recall
optimizer = keras.optimizers.RMSprop(lr = const_lr)
metrics = [keras.metrics.categorical_accuracy, helper.metrics.recall, helper.metrics.precision]
#, "precision", "recall"
model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

# CALLBACKS
# save model after each epoch
callback_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath="../checkpoints/checkpoint.hdf5", save_weights_only=True, save_best_only=True)
callback_csv = keras.callbacks.CSVLogger(filename="../logs/log.csv")
callback_tensorboard = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

callbacks=[callback_model_checkpoint, callback_csv, callback_splits_and_merges]

# TRAIN
if data_type == "array":
    statistics = model.fit(nb_epoch=nb_epoch,
                           batch_size=batch_size,
                           x=training_x,
                           y=training_y,
                           validation_data=(validation_x, validation_y),
                           callbacks = callbacks,
                           verbose = 1)
    
elif data_type == "images":
    statistics = model.fit_generator(nb_epoch=nb_epoch,
                                     samples_per_epoch = nb_batches * batch_size,
                                     generator = training_gen,
                                     validation_data = validation_gen,
                                     nb_val_samples=batch_size,
                                     callbacks=callbacks,
                                     verbose=1)
    
# visualize learning stats
helper.visualize.visualize_learning_stats(statistics, out_dir, metrics)
print('Done! :)')