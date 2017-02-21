import numpy as np

def data_from_array(data_dir):
    # load  x
    training_x = np.load(data_dir+"training/x.npy")
    test_x = np.load(data_dir+"test/x.npy")
    validation_x = np.load(data_dir+"validation/x.npy")

    print(training_x.shape)
    print(test_x.shape)
    print(validation_x.shape)

    # normalize
    training_x = training_x / 255
    test_x = test_x / 255
    validation_x = validation_x / 255

    # load y
    training_y = np.load(data_dir+"training/y.npy")
    test_y = np.load(data_dir+"test/y.npy")
    validation_y = np.load(data_dir+"validation/y.npy")

    print(training_y.shape)
    print("hier")
    print(test_y.shape)
    print(validation_y.shape)
    
    return [training_x, training_y, validation_x, validation_y, test_x, test_y]

def data_from_images(data_dir):
    return 0