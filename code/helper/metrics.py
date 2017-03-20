import numpy as np
import skimage.segmentation
import skimage.io
import keras.backend as K

debug = False

def precision(y_true, y_pred):
    # taken from Keras 1
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    
    return precision


def recall(y_true, y_pred):
    # taken from Keras 1
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    
    return recall


def probmap_to_pred(probmap, boundary_boost_factor):
    # we need to boost the boundary class to make it more visible
    # this shrinks the cells a little bit but avoids undersegmentation
    pred = np.argmax(probmap * [1, 1, boundary_boost_factor], -1)
    
    return pred


def pred_to_label(pred, cell_min_size=300, cell_label=1):
    
    cell = (pred == cell_label)
    # fix cells
    cell = skimage.morphology.remove_small_holes(cell, min_size=cell_min_size)
    cell = skimage.morphology.remove_small_objects(cell, min_size=cell_min_size)
    
    # label cells only
    [label, num] = skimage.morphology.label(cell, return_num=True)
    return label


def splits_and_merges(y_model_pred, y_gt_pred):
    
    # get segmentations
    label_gt = pred_to_label(y_gt_pred, cell_min_size=2)
    label_model = pred_to_label(y_model_pred, cell_min_size=2)
    
    # get number of detected nuclei
    nb_nuclei_gt = np.max(label_gt)
    nb_nuclei_model = np.max(label_model)
    
    # catch the case of an empty picture in model and gt
    if nb_nuclei_gt == 0 and nb_nuclei_model == 0:
        return [0, 0, 1]
    
    # catch the case of empty picture in model
    if nb_nuclei_model == 0:
        return [0, nb_nuclei_gt, 0]
    
    # catch the case of empty picture in gt
    if nb_nuclei_gt == 0:
        return [nb_nuclei_gt, 0, 0]
    
    # build IoU matrix
    IoUs = np.full((nb_nuclei_gt, nb_nuclei_model), -1, dtype = np.float32)

    # calculate IoU for each nucleus index_gt in GT and nucleus index_pred in prediction    
    # TODO improve runtime of this algorithm
    for index_gt in range(1,nb_nuclei_gt+1):
        nucleus_gt = label_gt == index_gt
        number_gt = np.sum(nucleus_gt)
        for index_model in range(1,nb_nuclei_model+1):
            nucleus_model = label_model == index_model 
            number_model = np.sum(nucleus_model)
            
            same_and_1 = np.sum((nucleus_gt == nucleus_model) * nucleus_gt)
            
            IoUs[index_gt-1,index_model-1] = same_and_1 / (number_gt + number_model - same_and_1)
    
    # get matches and errors
    detection_map = (IoUs > 0.5)
    nb_matches = np.sum(detection_map)

    detection_rate = IoUs * detection_map
    
    nb_overdetection = nb_nuclei_model - nb_matches
    nb_underdetection = nb_nuclei_gt - nb_matches
    
    mean_IoU = np.mean(np.sum(detection_rate, axis = 1))
    
    result = [nb_overdetection, nb_underdetection, mean_IoU]
    return result

# test
if(debug):
    y_model = np.load('/home/jr0th/github/segmentation/code/analysis/data/y_probmap/0f1f106b-057a-482c-93bb-c9a0b044c054.npy')
    y_gt_pred = skimage.io.imread('/home/jr0th/github/segmentation/code/analysis/data/y_gt/0f1f106b-057a-482c-93bb-c9a0b044c054.png')

    # y_model = np.array([[[[0.5, 0.3, 0.2], [0.1, 0.3, 0.6]], [[0.45, 0.5, 0.05], [0.1, 0.8, 0.1]]]])
    # y_gt = np.array([[[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]]]])

    # get prediction from model output
    print(y_model.shape)
    y_model_pred = probmap_to_pred(y_model, 1)
    print(y_model_pred.shape)
    print('y_gt_pred: ', y_gt_pred)
    print('y_model_pred: ', y_model_pred)

    # get splits and merges from prediction
    S_M = splits_and_merges(y_model_pred, y_gt_pred)
    print('S_M: ', S_M)