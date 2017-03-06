import tensorflow as tf
import keras.backend as K
def probmap_to_pred(probmap, boundary_boost_factor):

    # we need to boost the boundary class to make it more visible
    # this shrinks the cells a little bit but avoids undersegmentation
    pred = tf.argmax(probmap * [1, 1, boundary_boost_factor], 2)
    
    return pred


def splits_and_merges(y_true, y_pred):

    boundary_boost_factor = 100
    y_pred = probmap_to_pred(y_pred, boundary_boost_factor)
    
    y_true = tf.argmax(y_true, 2)
    #tf_session = K.get_session()
    #foo = tf.reduce_sum(y_true)
    #print(y_true.eval(tf_session))
    #print(foo.eval(tf_session))
    print('called')
    x = 5
    result = tf.constant(0+x)
    return 


configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "2"
session = tf.Session(config = configuration)

# test
Y = tf.constant([[[[0.5, 0.3, 0.2], [0.1, 0.3, 0.6]], [[0.45, 0.05, 0.5], [0.1, 0.8, 0.1]]]])
Y_gt = tf.constant([[[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]]]])

print('Y \n', Y.eval(session = session))
print('Y_gt \n', Y_gt.eval(session = session))

Y_pred = probmap_to_pred(Y, 1)

print('Y_pred \n', Y_pred.eval(session = session))

S_M = splits_and_merges(Y, Y_gt)

print('S_M \n', S_M.eval(session = session))