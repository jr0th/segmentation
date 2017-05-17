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


## PROBMAP TO CONTOURS TO LABEL

def probmap_to_contour(probmap):
    # assume 2D input
    outline = probmap >= 0.7
    
    return outline

def contour_to_label(outline, image):
    # see notebook contours_to_labels for why we do what we do here
    
    # get connected components
    labels = skimage.morphology.label(outline, background=1)
    skimage.morphology.remove_small_objects(labels, min_size = 100, in_place = True)
    
    n_ccs = np.max(labels)

    # buffer label image
    filtered_labels = np.zeros_like(labels, dtype=np.uint16)

    # relabel as we don't know what connected component the background has been given before
    label_index = 1
    
    # start at 1 (0 is contours), end at number of connected components
    for i in range(1, n_ccs + 1):

        # get mask of connected compoenents
        mask = labels == i

        # get mean
        mean = np.mean(np.take(image.flatten(),np.nonzero(mask.flatten())))

        if(mean > 50/255):
            filtered_labels[mask] = label_index
            label_index = label_index + 1
            
    return filtered_labels


## PROBMAP TO PRED TO LABEL

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


def compare_two_labels(label_model, label_gt, return_IoU_matrix):
    
    # get number of detected nuclei
    nb_nuclei_gt = np.max(label_gt)
    nb_nuclei_model = np.max(label_model)
    
    # catch the case of an empty picture in model and gt
    if nb_nuclei_gt == 0 and nb_nuclei_model == 0:
        if(return_IoU_matrix):
            return [0, 0, 1, np.empty(0)]     
        else:
            return [0, 0, 1]
    
    # catch the case of empty picture in model
    if nb_nuclei_model == 0:
        if(return_IoU_matrix):
            return [0, nb_nuclei_gt, 0, np.empty(0)]     
        else:
            return [0, nb_nuclei_gt, 0]
    
    # catch the case of empty picture in gt
    if nb_nuclei_gt == 0:
        if(return_IoU_matrix):
            return [nb_nuclei_model, 0, 0, np.empty(0)]     
        else:
            return [nb_nuclei_model, 0, 0]
    
    # build IoU matrix
    IoUs = np.full((nb_nuclei_gt, nb_nuclei_model), -1, dtype = np.float32)

    # calculate IoU for each nucleus index_gt in GT and nucleus index_pred in prediction    
    # TODO improve runtime of this algorithm
    for index_gt in range(1,nb_nuclei_gt+1):

        nucleus_gt = label_gt == index_gt
        number_gt = np.sum(nucleus_gt)

        for index_model in range(1,nb_nuclei_model+1):
            
            if debug:
                print(index_gt, "/", index_model)
            
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
    
    if(return_IoU_matrix):
        result = [nb_overdetection, nb_underdetection, mean_IoU, IoUs]
    else:
        result = [nb_overdetection, nb_underdetection, mean_IoU]
    return result

def splits_and_merges_3_class(y_model_pred, y_gt_pred):
    
    # get segmentations
    label_gt = pred_to_label(y_gt_pred, cell_min_size=2)
    label_model = pred_to_label(y_model_pred, cell_min_size=2)
    
    # compare labels
    result = compare_two_labels(label_model, label_gt, False)
        
    return result

def splits_and_merges_boundary(y_model_outline, y_gt_outline, image):
    
    # get segmentations
    label_gt = contour_to_label(y_gt_outline, image)
    label_model = contour_to_label(y_model_outline, image)
    
    # compare labels
    result = compare_two_labels(label_model, label_gt, False)
        
    return result