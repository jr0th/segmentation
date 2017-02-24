
import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt

import skimage.io
import sklearn.metrics

import numpy as np

def visualize(pred_y, true_x, true_y, out_dir='./', label=''):

    # TODO
    skimage.io.imsave(out_dir + label + '_' + 'img_probmap_boundary_test.png', pred_y[1,:,:,2])

    plt.figure()
    plt.hist(pred_y[1,:,:,2].flatten())
    plt.savefig(out_dir + label + '_' + 'hist_probmap_boundary')

    # print all samples for visual inspection
    nSamples = pred_y.shape[0]

    for sampleIndex in range(nSamples):
        nCols = 4
        figure, axes = plt.subplots(ncols=nCols, nrows=1, figsize=(nCols*5, 5))
        figure.tight_layout(pad = 1)

        predFig = axes[0]
        trueFig = axes[1]
        compFig = axes[2]
        cmatFig = axes[3]
        
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
        
        predFig.set_title('Prediction')
        trueFig.set_title('Truth')
        compFig.set_title('Errors')

        predFig.axis('off')
        trueFig.axis('off')
        compFig.axis('off')
        
        cmatFig.set_ylabel('truth')
        cmatFig.set_xlabel('prediction')
        
        # annotate share of pred
        for x in range(3):
            for y in range(3):
                cmatFig.annotate(str(np.round(cmat[x,y],2)), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center', fontsize = 15)
        
        plt.savefig(out_dir + label + '_' + str(sampleIndex) + '_vis')
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

def visualize_learning_stats(callback_batch_stats, statistics, out_dir, metrics):
    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(statistics.history["loss"])
    plt.plot(statistics.history["val_loss"])
    plt.legend(["Training", "Validation"])

    plt.savefig(out_dir + "plot_loss")
    
    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(statistics.history["categorical_accuracy"])
    plt.plot(statistics.history["val_categorical_accuracy"])
    plt.legend(["Training", "Validation"])

    plt.savefig(out_dir + "plot_accuracy")

    plt.figure()

    plt.xlabel("Batch")
    plt.ylabel("Metric")
    plt.plot(callback_batch_stats.losses['loss'])
    for metric in metrics:
        plt.plot(callback_batch_stats.losses[metric])
    plt.legend(['loss'] + metrics)

    plt.savefig(out_dir + "plot_batch_metrics")

    plt.figure()

    plt.xlabel("Batch")
    plt.ylabel("Size")
    plt.plot(callback_batch_stats.losses['size'])   
    plt.legend(['size'])

    plt.savefig(out_dir + "plot_batch_size")