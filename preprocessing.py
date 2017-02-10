import keras.utils.np_utils
import numpy

import skimage.io
import skimage.util

def window_gray(images):
    windows = []

    for image in images:
        
        # get blocks for each image
        blocks = skimage.util.view_as_windows(image, (128, 128), 128)
        
        # list blocks
        blocks = blocks.reshape((-1, 128, 128))
    
        # store
        windows.append(blocks)

    # turn list into an array
    windows = numpy.asarray(windows)
    
    return windows.reshape((-1, 128, 128, 1))

def gray_to_class(img_gray):
    img_class = img_gray
    img_class[img_class == 100] = 1
    img_class[img_class == 200] = 2

    # TODO this must be fixed.
    img_class[img_class == 31611] = 1
    img_class[img_class == 54484] = 2
    
    return img_class
    
def class_to_one_hot(img_class):
    # get shape and edit
    shape = img_class.shape
    shape_new_list = list(shape)
    shape_new_list[-1] = -1
    shape_new = tuple(shape_new_list)
    
    # remove artifacts and make on hot matrix
    class_images = gray_to_class(img_class)
    one_hot = keras.utils.np_utils.to_categorical(class_images)
    
    # restore shape
    one_hot = one_hot.reshape(shape_new)
    
    return one_hot

def images_to_numpy(img_dir, ):
    load_pattern_x = img_dir + "/x/*.png"
    load_pattern_y = img_dir + "/y/*.png"

    images = skimage.io.imread_collection(load_pattern_x)
    labels = skimage.io.imread_collection(load_pattern_y)

    x = window_gray(images)
    y = window_gray(labels)

    # y needs some more processing
    y = gray_to_class(y)
    y = class_to_one_hot(y)

    # debug shape
    print('shape x', x.shape)
    print('shape y', y.shape)

    # permute
    random_indeces = numpy.random.permutation(len(x))
    x = x[random_indeces]
    y = y[random_indeces]

    numpy.save(img_dir + "/x.npy", x)
    numpy.save(img_dir + "/y.npy", y)

# create training data
images_to_numpy("./data/training")
images_to_numpy("./data/test")
images_to_numpy("./data/validation")
