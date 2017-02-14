import keras.backend
import keras.callbacks
import keras.layers
import keras.models
import keras.optimizers

import helper.batch_logger
import helper.model_builder
import helper.visualize
import helper.objectives

import skimage.io
import sklearn.metrics

import scipy.stats
import pandas as pd

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt

# constants
const_lr = 1e-4
out_dir = '../out/'

# build session running on GPU 1
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "1"

session = tf.Session(config = configuration)

# apply session
keras.backend.set_session(session)

# load  x
training_x = np.load("../data/training/x.npy")
test_x = np.load("../data/test/x.npy")
validation_x = np.load("../data/validation/x.npy")

print(training_x.shape)
print(test_x.shape)
print(validation_x.shape)

# normalize
training_x = training_x / 255
test_x = test_x / 255
validation_x = validation_x / 255


# load y
training_y = np.load("../data/training/y.npy")
test_y = np.load("../data/test/y.npy")
validation_y = np.load("../data/validation/y.npy")

print(training_y.shape)
print(test_y.shape)
print(validation_y.shape)

# reshape # TODO in network
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

# get class weights
freq = scipy.stats.itemfreq(np.argmax(training_y[0,:,:,:], axis = 2))
print(freq)
freq_df = pd.DataFrame(freq[:,1]/np.sum(freq[:,1]), ['background', 'interior', 'boundary'], columns=['freq'])

print(freq_df)

class_weights = list(1/freq_df["freq"])
print(class_weights)

# build model
model = helper.model_builder.get_model(dim1, dim2)

loss = "categorical_crossentropy"
# TODO include precision and recall
metrics = ["categorical_accuracy"]
optimizer = keras.optimizers.RMSprop(lr = const_lr)

# model.compile(loss=loss,  metrics=metrics, optimizer=optimizer)
model.compile(loss=helper.objectives.w_categorical_crossentropy, metrics=metrics, optimizer=optimizer)

# add callbacks

# TODO implement early stopping
# callback_early_stopping = keras.callbacks.EarlyStopping(patience=4)

# save model after each epoch
callback_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath="../checkpoints/checkpoint.hdf5", save_weights_only=True, save_best_only=True)

# collect logs about each batch
callback_batch_stats = helper.batch_logger.BatchLogger(metrics, verbose=False)

callback_csv = keras.callbacks.CSVLogger(filename="../logs/log.csv")

callbacks=[callback_batch_stats, callback_model_checkpoint, callback_csv]

# TRAIN
statistics = model.fit(nb_epoch=5, batch_size=60, class_weight=class_weights, validation_data=(validation_x, validation_y_vec), x=training_x, y=training_y_vec, callbacks = callbacks , verbose = 1)


plt.figure()

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(statistics.history["loss"])
plt.plot(statistics.history["val_loss"])
plt.legend(["Training", "Validation"])

plt.savefig(out_dir + "plot_loss")


plt.figure()

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(statistics.history["categorical_accuracy"])
plt.plot(statistics.history["val_categorical_accuracy"])
plt.legend(["Training", "Validation"])

plt.savefig(out_dir + "plot_accuracy")


plt.figure()

plt.xlabel("Batch")
plt.ylabel("Metric")
plt.plot(callback_batch_stats.losses['loss'])
for metric in metrics:
    plt.plot(callback_batch_stats.losses[metric])
plt.legend(['loss'] + metrics)

plt.savefig(out_dir + "plot_batch_metrics")


plt.figure()

plt.xlabel("Batch")
plt.ylabel("Size")
plt.plot(callback_batch_stats.losses['size'])   
plt.legend(['size'])

plt.savefig(out_dir + "plot_batch_size")


# print test results
res = model.evaluate(test_x, test_y_vec)
print(res)


# visualize some test
prediction_test = model.predict(test_x).reshape((-1, 128, 128, 3))
helper.visualize.visualize(prediction_test, test_x, test_y, out_dir, 'test')

# visualize some training
prediction_training = model.predict(training_x).reshape((-1, 128, 128, 3))
helper.visualize.visualize(prediction_training, training_x, training_y, out_dir, 'training')

