import keras.backend as K
import tensorflow as tf

def w_categorical_crossentropy(y_true, y_pred):

    print(y_true.get_shape())
    print(y_pred.get_shape())
    mask_int = tf.argmax(y_true, axis=2)
    mask = tf.to_float(mask_int)

    weight_bg =  1/0.7
    weight_cell = 1/0.2
    weight_b = 1/0.1

    # scale up to 1  
    c = 1/(weight_bg + weight_cell + weight_bg)
    weight_bg *= c
    weight_cell *= c
    weight_b *= c
    
    weight_tensor = tf.constant([[[weight_bg, weight_cell, weight_b]]], dtype=tf.float32)
 
    mask = tf.multiply(y_true, weight_tensor)
    mask = tf.reduce_max(mask, axis = 2)
#    mask = tf.reshape(mask, [-1, 128*128])

    return tf.multiply(K.categorical_crossentropy(y_pred, y_true), mask)
