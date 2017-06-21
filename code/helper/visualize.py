import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skimage.io
import sklearn.metrics

import numpy as np

out_format = 'eps'

def visualize(pred_y, true_x, true_y, out_dir='./', label=''):

    # TODO
    skimage.io.imsave(out_dir + label + '_' + 'img_probmap_boundary_test.png', pred_y[1,:,:,2])

    plt.figure()
    plt.hist(pred_y[1,:,:,2].flatten())
    plt.savefig(out_dir + label + '_' + 'hist_probmap_boundary' + '.' + out_format, format=out_format)

    # print all samples for visual inspection
    nSamples = pred_y.shape[0]

    for sampleIndex in range(nSamples):
        nCols = 2
        figure, axes = plt.subplots(ncols=nCols, nrows=2, figsize=(nCols*5, nCols*5))

        predFig = axes[0,0]
        trueFig = axes[0,1]
        compFig = axes[1,0]
        cmatFig = axes[1,1]
        
        pred_prob_map = pred_y[sampleIndex,:,:,:]
        pred_prob_map_vec = pred_prob_map.reshape((-1, 3))
        pred = np.argmax(pred_prob_map, axis=2)
        
        true_prob_map = true_y[sampleIndex,:,:,:]
        true_prob_map_vec = true_prob_map.reshape((-1, 3))
        true = np.argmax(true_prob_map, axis=2)

        comp = pred != true
        
        cmat = sklearn.metrics.confusion_matrix(y_true = true.flatten(), y_pred = pred.flatten(), labels=[0,1,2])

        predFig.imshow(skimage.color.label2rgb(pred, image=true_x[sampleIndex,:,:,0]))
        trueFig.imshow(skimage.color.label2rgb(true, image=true_x[sampleIndex,:,:,0]))
        compFig.imshow(skimage.color.label2rgb(comp, image=true_x[sampleIndex,:,:,0]))
        cmatFig.matshow(cmat, cmap = "cool")
        
        predFig.set_title('Prediction', fontsize=16)
        trueFig.set_title('Truth', fontsize=16)
        compFig.set_title('Errors', fontsize=16)


        predFig.axis('off')
        trueFig.axis('off')
        compFig.axis('off')
        
        cmatFig.set_ylabel('truth', fontsize=16)
        cmatFig.set_xlabel('prediction', fontsize=16)
        
        cmatFig.tick_params(axis='both', which='major', labelsize=12)
        cmatFig.set_title('Confusion Matrix', fontsize=16, y=1.08)
        
        # annotate share of pred
        for x in range(3):
            for y in range(3):
                cmatFig.annotate(str(np.round(cmat[x,y],2)), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center', fontsize = 15)
        plt.tight_layout(pad = 1)
        plt.savefig(out_dir + label + '_' + str(sampleIndex) + '_vis' + '.' + out_format, format=out_format)
        classNames = ['background', 'interior', 'boundary']
        f = open(out_dir + '/' + label + '_' + str(sampleIndex) + '.txt', 'w')
        f.write(sklearn.metrics.classification_report(pred.flatten(), true.flatten(), target_names=classNames) + '\n')
        f.write('Jaccard: ' +
                str(sklearn.metrics.jaccard_similarity_score(y_pred = pred.flatten(), y_true = true.flatten())) +
                '\n')
        f.write('Cross Entropy: ' +
                str(sklearn.metrics.log_loss(y_pred = pred_prob_map_vec, y_true = true_prob_map_vec)) +
               '\n')
        f.close()
        
def visualize_boundary_hard(pred_y, true_x, true_y, out_dir='./', label=''):

    print('VISUALIZE', pred_y.shape, true_y.shape)
    plt.figure()
    plt.hist(pred_y.flatten())
    plt.savefig(out_dir + label + '_' + 'hist_probmap_boundary' + '.' + out_format, format=out_format)

    # print all samples for visual inspection
    nSamples = pred_y.shape[0]

    for sampleIndex in range(nSamples):

        nCols = 3
        nRows = 2
        figure, axes = plt.subplots(ncols=nCols, nrows=2, figsize=(nCols*5+2, nRows*5+2))

        origFig = axes[0,0]
        trueFig = axes[0,1]

        predProbMapFig = axes[1,0]
        predFig = axes[1,1]

        compFig = axes[0,2]
        cmatFig = axes[1,2]
        
        pred_prob_map = pred_y[sampleIndex,:,:,0]
        pred = pred_prob_map >= 0.5
        
        true = true_y[sampleIndex,:,:,0]
        true = true.astype(np.bool)
        
        print('TRUE MEAN', np.mean(true))
        print('PRED_PROB_MAP MEAN', np.mean(pred_prob_map))
        print('PRED MEAN', np.mean(pred))
        
        comp = pred != true
        
        cmat = sklearn.metrics.confusion_matrix(y_true = true.flatten(), y_pred = pred.flatten()) #, labels=[0,1])

        mappable = origFig.imshow(true_x[sampleIndex,:,:,0])
        # figure.colorbar(mappable, ax=origFig)
        
        trueFig.imshow(skimage.color.label2rgb(true, image=true_x[sampleIndex,:,:,0]))

        mappable = predProbMapFig.imshow(pred_prob_map)
        cbar = figure.colorbar(mappable, ax=predProbMapFig)
        cbar.ax.tick_params(labelsize=18) 
        
        predFig.imshow(skimage.color.label2rgb(pred, image=true_x[sampleIndex,:,:,0]))

        compFig.imshow(skimage.color.label2rgb(comp, image=true_x[sampleIndex,:,:,0]))
        cmatFig.matshow(cmat, cmap = "cool")

        predProbMapFig.set_title('Prediction (not thresholded)', fontsize=18)
        origFig.set_title('Image', fontsize=18)
        predFig.set_title('Prediction (thresholded)', fontsize=18)
        trueFig.set_title('Ground Truth', fontsize=18)
        compFig.set_title('Pixelwise Errors', fontsize=18)
        cmatFig.set_title('Confusion Matrix', fontsize=18)

        predProbMapFig.axis('off')
        origFig.axis('off')
        predFig.axis('off')
        trueFig.axis('off')
        compFig.axis('off')
        
        cmatFig.set_ylabel('truth', fontsize=18)
        cmatFig.set_xlabel('prediction', fontsize=18)
        
        cmatFig.tick_params(axis='both', which='major', labelsize=18)
        
        # annotate share of pred
        for x in range(2):
            for y in range(2):
                cmatFig.annotate(str(np.round(cmat[x,y],2)), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center', fontsize = 15)
        figure.tight_layout(pad = 1)
        plt.savefig(out_dir + label + '_' + str(sampleIndex) + '_vis' + '.' + out_format, format=out_format)
        classNames = ['background', 'boundary']
        
        # write cross entropy
        ce = sklearn.metrics.log_loss(y_pred = pred.flatten(), y_true = true.flatten())
        
        f = open(out_dir + '/' + label + '_' + str(sampleIndex) + '.txt', 'w')
        f.write('Cross Entropy: ' + str(ce) + '\n')
        f.close()
        
def visualize_boundary_soft(pred_y, true_x, true_y, out_dir='./', label=''):

    plt.figure()
    plt.hist(pred_y.flatten())
    plt.savefig(out_dir + label + '_' + 'hist_probmap_boundary' + '.' + out_format, format=out_format)

    # print all samples for visual inspection
    nSamples = pred_y.shape[0]

    for sampleIndex in range(nSamples):

        nCols = 4
        figure, axes = plt.subplots(ncols=nCols, nrows=1, figsize=(nCols*5+2, 5+2))

        origFig = axes[0]
        predFig = axes[1]
        trueFig = axes[2]
        compFig = axes[3]
        
        pred_prob_map = pred_y[sampleIndex,:,:,0]
        
        true_prob_map = true_y[sampleIndex,:,:,0]
        
        comp = pred_prob_map - true_prob_map

        origFig.imshow(true_x[sampleIndex,:,:,0])
        predFig.imshow(pred_prob_map) 
        trueFig.imshow(true_prob_map)
        compFig.imshow(comp)
        
        origFig.set_title('Image', fontsize=18)
        predFig.set_title('Prediction', fontsize=18)
        trueFig.set_title('Truth', fontsize=18)
        compFig.set_title('Errors (MSE)', fontsize=18)

        predFig.axis('off')
        trueFig.axis('off')
        compFig.axis('off')
        
        plt.savefig(out_dir + label + '_' + str(sampleIndex) + '_vis' + '.' + out_format, format=out_format)
        classNames = ['background', 'boundary']
        
        # write mean squared error
        ce = sklearn.metrics.mean_squared_error(y_pred = pred_prob_map.flatten(), y_true = true_prob_map.flatten())
        
        f = open(out_dir + '/' + label + '_' + str(sampleIndex) + '.txt', 'w')
        f.write('Mean Squared Error: ' + str(ce) + '\n')
        f.close()

def visualize_learning_stats(statistics, out_dir, metrics):
    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(statistics.history["loss"])
    plt.plot(statistics.history["val_loss"])
    plt.legend(["Training", "Validation"])

    plt.savefig(out_dir + "plot_loss" + '.' + out_format, format=out_format)
    
    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(statistics.history["categorical_accuracy"])
    plt.plot(statistics.history["val_categorical_accuracy"])
    plt.legend(["Training", "Validation"])

    plt.savefig(out_dir + "plot_accuracy" + '.' + out_format, format=out_format)
    
def visualize_learning_stats_boundary_hard(statistics, out_dir, metrics):
    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(statistics.history["loss"])
    plt.plot(statistics.history["val_loss"])
    plt.legend(["Training", "Validation"])

    plt.savefig(out_dir + "plot_loss" + '.' + out_format, format=out_format)
    
    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(statistics.history["binary_accuracy"])
    plt.plot(statistics.history["val_binary_accuracy"])
    plt.legend(["Training", "Validation"])

    plt.savefig(out_dir + "plot_accuracy" + '.' + out_format, format=out_format)
    
def visualize_learning_stats_boundary_soft(statistics, out_dir, metrics):
    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(statistics.history["loss"])
    plt.plot(statistics.history["val_loss"])
    plt.legend(["Training", "Validation"])

    plt.savefig(out_dir + "plot_loss" + '.' + out_format, format=out_format)
    
    plt.figure()