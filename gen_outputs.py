from load_dict import load_dict
from models import model_conv_525,model_conv_complex,model_conv_525_LSTM, model_conv_525_GRU, res_model_1d, My_Unet_1d
from scipy.io import wavfile
import pickle
import config
import numpy as np
import os
import matplotlib.pyplot as plt
from preprocessor import visualize,random_crop
from postprocessor import metrics,binary,normalization
from sklearn.metrics import roc_curve,auc

import random
import tensorflow as tf
from metrics import F1Score
from focal_loss import focal_loss
import argparse



def Load_model(model_dir,model_name='conv_model_1d',dropout=0.0):

    assert model_name in ['conv_model_1d','res_model_1d','conv_model_1d_LSTM','conv_model_1d_GRU','My_Unet_1d']

    if model_name == 'conv_model_1d':
        model = model_conv_525()
    elif model_name == 'res_model_1d':
        model = res_model_1d()
    elif model_name == 'conv_model_1d_LSTM':
        model = model_conv_525_LSTM(dropout=dropout)
    elif model_name == 'conv_model_1d_GRU':
        model = model_conv_525_GRU(dropout=dropout)
    elif model_name == 'My_Unet_1d':
        model = My_Unet_1d(drop_p=dropout,output_layer='LSTM')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','Precision','Recall',F1Score()]) 
    model.load_weights(model_dir)
    model.summary()

    return model

def load_data(domain_name,batch_size=32,cough=False,No_cough=False,data_dir = '../data/',):
    '''
    output:array-like wav/lbl
    cough=True: the output only consists cough
    No_cough=True: the output without cough
    cough=False and No_cough=False: random pick
    '''

    assert((cough and No_cough) == False)

    output_wav = np.zeros((batch_size,config.OUTPUT_LENGTH*config.sr,1))
    output_lbl = np.zeros((batch_size,config.OUTPUT_LENGTH*(int(1/config.duration_t)),1))

    name_list = [] # to store the names of wavs
    data_path = data_dir+'/wav_dur_16khz_'+domain_name 
    if os.path.exists(data_path):
        data_list = sorted(os.listdir(data_path))
        num_sample = int(len(data_list)/2)# num of wavs = num of lbls
        counter = 0

        if (cough or No_cough): # only contains coughs
            indxs = []
            while counter<batch_size:

                idx = random.sample(range(0,num_sample-1),1)[0]
                if idx in indxs:
                    continue
                else: 
                    indxs.append(idx)
                    # load lbl
                    lbl_path = data_path+'/'+data_list[idx]

                    with open(lbl_path,'rb') as pf:
                        lbl = pickle.load(pf).astype('uint8')
                        lbl = np.array(lbl)

                    # if it's a cough
                    if (cough and np.max(lbl)>=1) or (No_cough and np.max(lbl)<=0):
                        # load wav
                        wav_path = data_path + '/wav' + data_list[idx][5:-3] + 'wav'
                        _, wav = wavfile.read(wav_path,'w')
                        wav = np.asarray(wav)
                        # random crop
                        wav,lbl = random_crop(wav,lbl)
                        output_wav[counter] = np.expand_dims(wav,-1)
                        output_lbl[counter] = np.expand_dims(lbl,-1)
                        counter += 1
                        name_list.append(data_list[idx][5:-3])
                    else:
                        continue

        else:
            # random loading
            idxs =  random.sample(range(0,num_sample-1),batch_size)  
            for i,idx in enumerate(idxs):
                # load lbl
                lbl_path = data_path+'/'+data_list[idx]
                with open(lbl_path,'rb') as pf:
                    lbl = pickle.load(pf).astype('uint8')
                    lbl = np.array(lbl)
                # load wav
                wav_path = data_path + '/wav' + data_list[idx][5:-3] + 'wav'
                _, wav = wavfile.read(wav_path,'w')
                wav = np.asarray(wav)

                wav,lbl = random_crop(wav,lbl)
                # write into output
                output_wav[i] = np.expand_dims(wav,-1)
                output_lbl[i] = np.expand_dims(lbl,-1)
                name_list.append(data_list[idx][5:-3])
    else:
        raise Exception(' domain name does not exist: {}'.format(domain_name))

    return (output_wav,output_lbl,name_list)








if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, help="where is the model")
    parser.add_argument('--model_name', type=str, default="conv_model_1d", help="model name")
    parser.add_argument('--dropout', type=float, default=0.0, help="add drop out? only valid for some models")
    parser.add_argument('--output_dir', type=str, help="output dir of the results")
    parser.add_argument('--domain_names', type=list, default=['whosecough','southafrica','jotform','cs','audioset'], help='domain names to predict')
    args = parser.parse_args()

    model = Load_model(args.model_dir,model_name=args.model_name,dropout=args.dropout)

    save_dir = args.output_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for (i,domain) in enumerate(args.domain_names):
        if domain != 'fsd':
            (wav_cough,lbl_cough,cough_name_list) = load_data(domain,batch_size=32,cough=True)
            (wav_no_cough,lbl_no_cough,no_cough_name_list) = load_data(domain,batch_size=32,No_cough=True)
        else:
            (wav_cough,lbl_cough,cough_name_list) = load_data(domain,batch_size=32,cough=False)
            (wav_no_cough,lbl_no_cough,no_cough_name_list) = load_data(domain,batch_size=32,No_cough=False)

        preds_cough = model.predict(wav_cough)
        preds_no_cough = model.predict(wav_no_cough)
        # cough
        if not os.path.exists(save_dir+'{}'.format(domain)):
            os.mkdir(save_dir+'{}'.format(domain))

        save_path = save_dir+'{}/cough'.format(domain)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in range(32):
            pred = preds_cough[i]
            pred[pred>=0.5] = 1
            pred[pred<=0.5] = 0
            plt.figure()
            visualize(wav_cough[i],lbl_cough[i],pred,ax=plt,name=cough_name_list[i])
            plt.savefig(save_path+'/%s_%d'%(domain,i)+'.png', bbox_inches='tight')
            #plt.show()
        
        # no cough
        save_path = save_dir+'{}/no_cough'.format(domain)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in range(32):
            pred = preds_no_cough[i]
            pred[pred>=0.6] = 1
            pred[pred<=0.6] = 0
            plt.figure()
            visualize(wav_no_cough[i],lbl_no_cough[i],pred,ax=plt,name=no_cough_name_list[i])
            plt.savefig(save_path+'/%s_%d'%(domain,i)+'.png', bbox_inches='tight')
            #plt.show()
        
        print('{} done.'.format(domain))