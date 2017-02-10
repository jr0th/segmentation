import keras.callbacks
import pandas as pd

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