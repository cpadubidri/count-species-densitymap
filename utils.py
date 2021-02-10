from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback
import os
from keras import backend as K
import numpy as np


def create_folder(train_id):
    weights_path = os.path.join('./Training_Logs',train_id,'weights')
    logs_path = os.path.join('./Training_Logs',train_id,'log')
    os.makedirs(weights_path)
    os.makedirs(logs_path)
    return weights_path, logs_path

class LossHistory(Callback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path

    def on_train_begin(self, logs={}):
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.lr.append(K.eval(self.model.optimizer.lr))
        np.savetxt(os.path.join(self.log_path,"lr.csv"), self.lr, delimiter=",", fmt='%s', header="LR")

def call_backs(train_id):
    weights_path, logs_path = create_folder(train_id)
    best_model = train_id + '_best_model.h5'
    callbacks = [
        ModelCheckpoint(os.path.join(weights_path,best_model), save_weights_only=True, save_best_only=True, mode='min'),
        ReduceLROnPlateau(monitor="loss",
                                            patience=3,
                                            verbose=1,
                                            factor=0.2,
                                            mode="max",
                                            min_lr=0.000000001),
        CSVLogger(os.path.join(logs_path,'train_log.csv'), append=False, separator=';'),
        LossHistory(logs_path)
        ]
    return callbacks

def save_on_epoch_end(train_id, model):
    weights_path = os.path.join('./Training_Logs', train_id, 'weights')
    end_model = train_id+'_model_eend.h5'
    model.save_weights(os.path.join(weights_path,end_model))

