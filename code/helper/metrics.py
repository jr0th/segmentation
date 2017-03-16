import numpy as np
import skimage.segmentation
import keras.backend as K

def precision(y_true, y_pred):
    # taken from Keras 1
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # taken from Keras 1
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def probmap_to_pred(probmap, boundary_boost_factor):
    # we need to boost the boundary class to make it more visible
    # this shrinks the cells a little bit but avoids undersegmentation
    pred = np.argmax(probmap * [1, 1, boundary_boost_factor], 3)
    
    return pred


def splits_and_merges(y_pred, y_pred_gt):

    boundary_boost_factor = 100
    y_pred = probmap_to_pred(y_pred, boundary_boost_factor)
    
    result = 0
    return result

# test
y = np.array([[[[0.5, 0.3, 0.2], [0.1, 0.3, 0.6]], [[0.45, 0.05, 0.5], [0.1, 0.8, 0.1]]]])
y_pred_gt = np.array([[[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]]]])

y_pred = probmap_to_pred(y, 1)
print('y_pred: ', y_pred)

S_M = splits_and_merges(y, y_pred)
print('S_M: ', S_M)