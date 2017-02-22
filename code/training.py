import keras.backend
import keras.callbacks
import keras.layers
import keras.models
import keras.optimizers

import matplotlib
matplotlib.use('SVG')

import helper.batch_logger
import helper.model_builder
import helper.visualize
import helper.objectives
import helper.data_provider

import skimage.io
import sklearn.metrics

import scipy.stats
import pandas as pd

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# constants
const_lr = 1e-4
data_dir = "/home/jr0th/github/segmentation/data/set03/"
data_type = "images" # "images" or "array"
out_dir = "../out/"

nb_epoch = 10
batch_size = 2

# generator only params
nb_batches = 2

# build session running on GPU 1
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "0"
session = tf.Session(config = configuration)

# apply session
keras.backend.set_session(session)

if data_type == "array":
    [training_x, training_y, validation_x, validation_y, test_x, test_y] = helper.data_provider.data_from_array(data_dir)
    
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
    model = helper.model_builder.get_model(dim1, dim2)
    loss = "categorical_crossentropy"
    # loss = helper.objectives.w_categorical_crossentropy
    
elif data_type == "images":
    [training_generator, validation_generator, test_generator, dim1, dim2] = helper.data_provider.data_from_images(data_dir, batch_size=batch_size)
    model = helper.model_builder.get_model_3d_output(dim1, dim2)
    loss = "categorical_crossentropy"
    # loss = helper.objectives.w_categorical_crossentropy_3d
    

# TODO include precision and recall
metrics = ["categorical_accuracy"]
optimizer = keras.optimizers.RMSprop(lr = const_lr)

# model.compile(loss=loss,  metrics=metrics, optimizer=optimizer)
model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

# CALLBACKS
# save model after each epoch
callback_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath="../checkpoints/checkpoint.hdf5", save_weights_only=True, save_best_only=True)
# collect logs about each batch
callback_batch_stats = helper.batch_logger.BatchLogger(metrics, verbose=False)
callback_csv = keras.callbacks.CSVLogger(filename="../logs/log.csv")
callback_tensorboard = keras.callbacks.TensorBoard(log_dir='../logs/logs_tensorboard', histogram_freq=1)
callbacks=[callback_batch_stats, callback_model_checkpoint, callback_csv, callback_tensorboard]

# TRAIN
if data_type == "array":
    statistics = model.fit(nb_epoch=nb_epoch,
                           batch_size=batch_size,
                           x=training_x,
                           y=training_y_vec,
                           validation_data=(validation_x, validation_y_vec),
                           callbacks = callbacks,
                           verbose = 1)
    
    # print test results
    res = model.evaluate(test_x, test_y_vec)
    print(res)

    # visualize learning stats
    helper.visualize.visualize_learning_stats(callback_batch_stats, statistics, out_dir, metrics)

    # visualize some test
    prediction_test = model.predict(test_x).reshape((-1, 128, 128, 3))
    helper.visualize.visualize(prediction_test, test_x, test_y, out_dir, 'test')

    # visualize some training
    prediction_training = model.predict(training_x).reshape((-1, 128, 128, 3))
    helper.visualize.visualize(prediction_training, training_x, training_y, out_dir, 'training')
    
elif data_type == "images":
    statistics = model.fit_generator(nb_epoch=nb_epoch,
                                     samples_per_epoch = nb_batches * batch_size,
                                     generator = training_generator,
                                     validation_data = validation_generator,
                                     nb_val_samples=batch_size,
                                     callbacks=callbacks,
                                     verbose=1)
    
    # print test results
    res = model.evaluate_generator(generator = test_generator, val_samples = batch_size)
    print(res)
    
    # visualize learning stats
    helper.visualize.visualize_learning_stats(callback_batch_stats, statistics, out_dir, metrics)
    
    # TODO visualize some test
