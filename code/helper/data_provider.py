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
    print(test_y.shape)
    print(validation_y.shape)

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
    
    return [training_x, training_y_vec, validation_x, validation_y_vec, test_x, test_y_vec]

def data_from_images(data_dir):
    return 0