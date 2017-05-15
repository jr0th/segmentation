import keras.callbacks

import helper.metrics

import tensorflow as tf
import pandas as pd
import numpy as np

debug = False

boundary_boost_factor = 100
tag_over = 'val_overdetection'
tag_under = 'val_underdetection'
tag_IoU = 'val_mean_IoU'

class SplitsAndMergesLogger3Class(keras.callbacks.TensorBoard):
    
    def __init__(self, data_type, data, gen_calls = 0, log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False):
        super().__init__(log_dir, histogram_freq, write_graph, write_images)
        
        self.data_type = data_type
        self.gen_calls = gen_calls
        
        # if data_type == "images" expect generator, if "array" expect [x, y] numpy arrays.
        self.data = data

        
    def set_model(self, model):
        super().set_model(model)
        
        self.result_placeholder = tf.placeholder(tf.float16, (3))
        
        self.value_over = tf.reduce_mean(tf.slice(self.result_placeholder, [0], [1]))
        self.value_under = tf.reduce_mean(tf.slice(self.result_placeholder, [1], [1]))
        self.value_IoU = tf.reduce_mean(tf.slice(self.result_placeholder, [2], [1]))
        
        self.summary_over = tf.summary.scalar(tag_over, self.value_over)
        self.summary_under = tf.summary.scalar(tag_under, self.value_under)
        self.summary_IoU = tf.summary.scalar(tag_IoU, self.value_IoU)
        
    def on_batch_end(self, batch, logs):
        super().on_batch_end(batch, logs)
        
        
    def get_error_for_batch(self, x_batch, y_batch):

        # get probmaps from model
        y_model_probmap_batch = self.model.predict_on_batch(x_batch)

        # get predictions from probmaps
        y_model_pred_batch = helper.metrics.probmap_to_pred(y_model_probmap_batch, boundary_boost_factor)
        y_gt_pred_batch = helper.metrics.probmap_to_pred(y_batch, boundary_boost_factor)

        # buffer for all results
        results = np.empty(shape = (0, 3), dtype = np.float16)        
        
        # loop over all samples in the batch
        for index in range(len(y_model_pred_batch)):
            
            # get data
            y_model_pred = y_model_pred_batch[index]
            y_gt_pred = y_gt_pred_batch[index]

            # calculate and save error
            result = helper.metrics.splits_and_merges_3_class(y_model_pred, y_gt_pred)
            results = np.vstack((results, result))
        
        return results
        
        
    def on_epoch_end(self, epoch, logs):
        super().on_epoch_end(epoch, logs)
        
        if self.data_type == "images":
            # predict stuff from images – use generator
            
            # buffer for all results
            results = np.empty(shape = (0, 3), dtype = np.float16)        
            
            # loop over necessary calls to generator (otherwise batches don't fit in memory)
            for index in range(self.gen_calls):
                
                generated = next(self.data)
                x_batch = generated[0]
                y_batch = generated[1]
                
                result = self.get_error_for_batch(x_batch, y_batch)
                
                results = np.vstack((results, result))
            
        elif self.data_type == "array":
            # predict probmap from nummpy arrays
            
            x_batch = self.data[0]
            y_batch = self.data[1]
            
            results = self.get_error_for_batch(x_batch, y_batch)

        # DEBUG print results. Can be removed later, but it is a nice thing to see during the training.
        if debug:
            print(results)
        
        # get all the metrics staged in Tensors.
        listOfSummaries = [self.summary_over, self.summary_under, self.summary_IoU]
        listOfSummaryResults = self.sess.run(listOfSummaries, feed_dict = {self.result_placeholder : np.mean(results, 0)})

        # write to file                                  
        for summary in listOfSummaryResults:
            self.writer.add_summary(summary, global_step = epoch)
        
        # flush writer and write statistics to file
        self.writer.flush()
        
class SplitsAndMergesLoggerBoundary(keras.callbacks.TensorBoard):
    
    def __init__(self, data_type, data, gen_calls = 0, log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False):
        super().__init__(log_dir, histogram_freq, write_graph, write_images)
        
        self.data_type = data_type
        self.gen_calls = gen_calls
        
        # if data_type == "images" expect generator, if "array" expect [x, y] numpy arrays.
        self.data = data

        
    def set_model(self, model):
        super().set_model(model)
        
        self.result_placeholder = tf.placeholder(tf.float16, (3))
        
        self.value_over = tf.reduce_mean(tf.slice(self.result_placeholder, [0], [1]))
        self.value_under = tf.reduce_mean(tf.slice(self.result_placeholder, [1], [1]))
        self.value_IoU = tf.reduce_mean(tf.slice(self.result_placeholder, [2], [1]))
        
        self.summary_over = tf.summary.scalar(tag_over, self.value_over)
        self.summary_under = tf.summary.scalar(tag_under, self.value_under)
        self.summary_IoU = tf.summary.scalar(tag_IoU, self.value_IoU)
        
    def on_batch_end(self, batch, logs):
        super().on_batch_end(batch, logs)
        
        
    def get_error_for_batch(self, x_batch, y_gt_contour_batch):

        # get probmaps from model
        y_model_probmap_batch = self.model.predict_on_batch(x_batch)

        # get contour from probmaps
        y_model_contour_batch = helper.metrics.probmap_to_contour(y_model_probmap_batch)

        # buffer for all results
        results = np.empty(shape = (0, 3), dtype = np.float16)        
        
        # loop over all samples in the batch
        for index in range(len(y_model_contour_batch)):
            
            # get data
            x_image = x_batch[index]
            y_model_contour = y_model_contour_batch[index]
            y_gt_contour = y_gt_contour_batch[index]

            # calculate and save error
            result = helper.metrics.splits_and_merges_boundary(y_model_contour, y_gt_contour, x_image)
            results = np.vstack((results, result))
        
        return results
        
        
    def on_epoch_end(self, epoch, logs):
        super().on_epoch_end(epoch, logs)
        
        if self.data_type == "images":
            # predict stuff from images – use generator
            
            # buffer for all results
            results = np.empty(shape = (0, 3), dtype = np.float16)        
            
            # loop over necessary calls to generator (otherwise batches don't fit in memory)
            for index in range(self.gen_calls):
                
                generated = next(self.data)
                x_batch = generated[0]
                y_batch = generated[1]
                
                result = self.get_error_for_batch(x_batch, y_batch)
                
                results = np.vstack((results, result))
            
        elif self.data_type == "array":
            # predict probmap from nummpy arrays
            
            x_batch = self.data[0]
            y_batch = self.data[1]
            
            results = self.get_error_for_batch(x_batch, y_batch)

        # DEBUG print results. Can be removed later, but it is a nice thing to see during the training.
        if debug:
            print(results)
        
        # get all the metrics staged in Tensors.
        listOfSummaries = [self.summary_over, self.summary_under, self.summary_IoU]
        listOfSummaryResults = self.sess.run(listOfSummaries, feed_dict = {self.result_placeholder : np.mean(results, 0)})

        # write to file                                  
        for summary in listOfSummaryResults:
            self.writer.add_summary(summary, global_step = epoch)
        
        # flush writer and write statistics to file
        self.writer.flush()

        
# Watch out. Does not work with custom metrics as Pandas doesn't handle keys that are functions correctly.
class BatchLogger(keras.callbacks.Callback):

    def __init__(self, metrics, verbose):
        self.columns = ['batch', 'size', 'loss'] + metrics
        self.losses = pd.DataFrame(columns = self.columns)
        self.verbose = verbose
        
    def on_batch_end(self,batch,logs):
        new_df = pd.Series(list(logs.values()), index=logs.keys())
        self.losses = self.losses.append(new_df, ignore_index=True)
        
        if self.verbose:
            print('\n', 'a batch is done!')
            for key in logs.keys():
                print('\n', key, ': ', logs.get(key))