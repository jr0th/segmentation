import keras.callbacks
import keras.backend as K

import tensorflow as tf
import pandas as pd

class SplitsAndMergesLogger(keras.callbacks.TensorBoard):
    
    def __init__(self, log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False):
        super().__init__(log_dir, histogram_freq, write_graph, write_images)

        self.tag_splits_and_merges = 'splits_and_merges'

        
    def set_model(self, model):
        super().set_model(model)
        
        self.result_tensor = tf.placeholder(tf.float16)
        self.summary_tensor = tf.summary.scalar(self.tag_splits_and_merges, self.result_tensor)
        
    
    def on_epoch_end(self, epoch, logs):
        super().on_epoch_end(epoch, logs)
        
        # prepare log
        result = 42 + epoch # this is complicated later
        summary = self.sess.run(self.summary_tensor, feed_dict = {self.result_tensor : result})
        
        # write to file
        self.writer.add_summary(summary, global_step = epoch)
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