import keras.callbacks
import keras.backend as K

import tensorflow as tf
import pandas as pd

class SplitsAndMergesLogger(keras.callbacks.Callback):
    
    
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.tag_splits_and_merges = 'splits_and_merges'
        print('constructor done')

    def set_model(self, model):
        self.model = model
        self.writer = tf.summary.FileWriter(self.log_dir)
        
        self.result_tensor = tf.placeholder(tf.float16)
        self.summary_tensor = tf.summary.scalar(self.tag_splits_and_merges, self.result_tensor)
        
        print('set_model done')
    
    def on_epoch_end(self, epoch, logs):
        # log stuff to TensorBoard
        
        # prepare log
        result = 42 + epoch
        summary = K.get_session().run(self.summary_tensor, feed_dict = {self.result_tensor : result})
        
        # write to file
        self.writer.add_summary(summary, global_step = epoch)
        self.writer.flush()
        
    def on_train_end(self, _):
        self.writer.close()

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