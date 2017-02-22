import keras.metrics
import tensorflow as tf

def w_categorical_crossentropy(y_true, y_pred):
    flag_debug = False

    # get number of classes
    nb_classes = y_pred.get_shape()[2].value

    # calculate frequency over batch
    freq = tf.reduce_mean(y_true, axis = [0, 1])

    if flag_debug:    
        freq = tf.Print(freq, [freq], message="Freq: ")
    
    # calculate weights
    weight = tf.divide(1, freq)
    if flag_debug:    
        weight = tf.Print(weight, [weight], message="Weight (unnormalized): ")
    
    # normalize weights
    weight = tf.divide(weight, tf.reduce_sum(weight))
    weight = tf.multiply(weight, nb_classes)    
    if flag_debug:    
        weight = tf.Print(weight, [weight], message="Weight (normalized): ")

    # enlarge vector to prepare for multiplication    
    weight = tf.reshape(weight, [1, 1, nb_classes])

    # multiply true labels with weights to get weight for each pixel 
    mask = tf.multiply(y_true, weight)
    mask = tf.reduce_max(mask, axis = 2)

    loss =  tf.multiply(keras.metrics.categorical_crossentropy(y_pred, y_true), mask)
    
    return loss

def categorical_crossentropy_3d(y_true, y_pred):

    # get number of classes
    nb_classes = y_pred.get_shape()[3].value
    dim1 = y_pred.get_shape()[1].value
    dim2 = y_pred.get_shape()[2].value

    y_pred_reshape = tf.reshape(y_pred, (-1, dim1 * dim2, nb_classes))
    y_true_reshape = tf.reshape(y_true, (-1, dim1 * dim2, nb_classes))

    loss = keras.metrics.categorical_crossentropy(y_pred_reshape, y_true_reshape)
    
    return loss
