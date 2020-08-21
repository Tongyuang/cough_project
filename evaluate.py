from load_dict import load_dict
from model_conv_1D import conv_model_1d, res_model_1d
from scipy.io import wavfile
import pickle
import numpy as np
import os
from preprocessor import visualize,random_crop
from postprocessor import metrics,binary,normalization
from sklearn.metrics import roc_curve,auc

import tensorflow as tf
from models import model_conv_525,model_conv_complex,model_conv_525_LSTM

from dataGenerator import train_valid_spliter, DataGenerator
from metrics import F1Score

import config
def Load_model(model_dir,model_name='conv_model_1d'):
    if model_name == 'conv_model_1d':
        model = model_conv_525_LSTM()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','Precision','Recall',F1Score()]) 
        model.load_weights(model_dir)
        model.summary()
    elif model_name == 'res_model_1d':
        model = res_model_1d()
        #metrics_dict = {'acc':'accuracy','prec':'Precision','rec':'Recall'}
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
        model.load_weights(model_dir)
        model.summary()
    return model

if __name__ == "__main__":

    model_dir = './checkpoints/conv525/conv525_kernel2/conv_525_5/cp/cp.ckpt'


    # generate train_valid dict
    tvs = train_valid_spliter(config.subtype_CV_dict,0)
    train,valid = tvs.gen_train_valid_df()

    # dataloader
    #train_data = DataGenerator(TRAIN=True,subtype_dict=train,if_preprocess=True)
    valid_data = DataGenerator(TRAIN=False,subtype_dict=valid,if_preprocess=False)

    model = Load_model(model_dir)
    scores = model.evaluate_generator(valid_data,steps=1)
    print(scores)
    print(model.metrics_names)