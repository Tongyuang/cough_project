import numpy as np
import tensorflow as tf
import os
from scipy.io import wavfile
import librosa

import config
from config import config as c

import preprocessor
import pickle

class train_valid_spliter:
    '''
    generate training dataframe and testing dataframe
    '''
    
    def __init__(self,subtype_cv_sample_dict,CV_for_test=3):

        self.subtype_cv_sample_dict = subtype_cv_sample_dict
        self.CV_for_train = [0,1,2,3]
        self.CV_for_train.remove(CV_for_test)
        self.CV_for_test = CV_for_test

    def gen_train_valid_df(self):

        sound_dict = config.SOUND_DICT
        train_dict = {key:[] for key in sound_dict.keys()}
        valid_dict = {key:[] for key in sound_dict.keys()}

        for (subtype,CV) in self.subtype_cv_sample_dict.keys():
            if CV in self.CV_for_train:
                train_dict[subtype] = train_dict[subtype]+self.subtype_cv_sample_dict[(subtype,CV)]
            else:
                valid_dict[subtype] = valid_dict[subtype]+self.subtype_cv_sample_dict[(subtype,CV)]

        return train_dict,valid_dict

class DataGenerator(tf.keras.utils.Sequence):#tf.keras.utils.Sequence):

    'Load data for Keras'
    def __init__(self, TRAIN, subtype_dict,if_preprocess = True ):

        'Initialization'
        self.TRAIN=TRAIN
        self.sps = config.samps_per_subtype_32
        self.random_seed_idx = 0
        self.if_preprocess = if_preprocess
        self.subtype_dict = subtype_dict

    def __len__(self):
        '''
        number of samples in a batch
        '''
        return c['batch_num_per_epoch']

    def preprocess(self,wav,lbl):
        wav,lbl = preprocessor.random_crop(wav,lbl,extra_p=0.25)
        if (self.TRAIN == False) or (self.if_preprocess == False):
            pass
        else:
            wav = preprocessor.random_gain(wav)
            wav = preprocessor.add_background_noise(wav)

        return wav,lbl

    def __getitem__(self, index):

        X = []
        y = []
        # generate data randomly, distribution from config.py
        for key in config.samps_per_subtype_32.keys():
            # random choice
            filename_index = np.random.choice(np.arange(0,len(self.subtype_dict[key]),1),config.samps_per_subtype_32[key],replace=False)
            for idx in filename_index:

                (domain,long_name) = self.subtype_dict[key][idx]
                domain_path = 'wav_dur_{}_{}'.format(config.sr_str,config.domain_name_dict[domain])
                wav_path = config.config_data['folder_data']+'/'+domain_path+'/'+long_name +'.wav'
                pkl_path = config.config_data['folder_data']+'/'+domain_path+'/'+'label'+long_name[3:]+'.pkl'
                

                try:
                    with open(pkl_path,'rb') as pf:
                        aug = pickle.load(pf).astype("uint8")
                    _, wav = wavfile.read(wav_path) 
                except:
                    print('could not get file:',wav_path)
                    wav = np.zeros(config.sr*config.duration)
                    aug = np.zeros(config.duration*((int)(1/config.duration_t)))

                wav,aug = self.preprocess(wav,aug)

                X.append(wav)
                y.append(aug)

        # if for some reason we couldn't get a sample, just repeat the last one
        while (len(X)<c['batch_size']):
            X.append(X[-1])
            y.append(y[-1])

        X = np.expand_dims(np.array(X),-1) # (batch, length, 1)
        y = np.expand_dims(np.array(y),-1)
        return X,y