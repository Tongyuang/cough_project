import argparse
import os
import pickle

import numpy as np
from scipy.io import wavfile

import config
import tensorflow as tf
from metrics import F1Score
from model_conv_1D import conv_model_1d, res_model_1d
from models import (My_Unet_1d, model_conv_525, model_conv_525_GRU,
                    model_conv_525_LSTM, model_conv_complex, res_model_1d)
from preprocessor import random_crop


# confusion_matrix
cm = tf.math.confusion_matrix

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

def get_array(wav_list):
    '''
    generate wav_array and lbl_array from wav_list
    wavlist:[(domain,long_name),...]
    '''
    wav_array = np.zeros((len(wav_list),int(config.sr*config.OUTPUT_LENGTH)))
    lbl_array = np.zeros((len(wav_list),int(config.OUTPUT_LENGTH / config.duration_t)))
    for (i,(domain,wav_name)) in enumerate(wav_list):
        domain_path = 'wav_dur_{}_{}'.format(config.sr_str,config.domain_name_dict[domain])
        wav_path = config.config_data['folder_data']+'/'+domain_path+'/'+wav_name +'.wav'
        pkl_path = config.config_data['folder_data']+'/'+domain_path+'/'+'label'+wav_name[3:]+'.pkl'

        try:
            with open(pkl_path,'rb') as pf:
                lbl = pickle.load(pf).astype("uint8")
            _, wav = wavfile.read(wav_path) 
            lbl,wav = np.asarray(lbl),np.asarray(wav)
        except:
            print('could not get file:',wav_path)
            wav = np.zeros(config.sr*config.duration)
            lbl = np.zeros(config.duration*((int)(1/config.duration_t)))
        
        wav,lbl = random_crop(wav,lbl)
        
        wav_array[i] = wav
        lbl_array[i] = lbl
    
    return wav_array,lbl_array

def calculate_metrics(TP,FP,TN,FN):
    '''
    calculate metrics
    acc = (TP+TN)/(TP+FP+TN+FN)
    prec = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = 2*(prec*Recall)/(prec+Recall)
    '''

    acc = (TP+TN)/(TP+FP+TN+FN)
    prec = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = 2*(prec*Recall)/(prec+Recall)

    return acc,prec,Recall,F1
def evaluate_samples(model,wav_list,num_samples=500,batch_size=config.config['batch_size']):
    '''
    evaluate the model by predicting the samples in the wav list
    para model: the model
    para wav_list: wav list
    para num_samples: number of samples being predicted, default 500
    para batch_size: size of input of the model, default as config

    confusion matrix:
    mtx = cm(true_lbls,predictions)

    mtx = 

    TN | FP
    _______
    FN | TP
    '''
    assert (num_samples <= len(wav_list))

    TP = FP = TN = FN = 0
    num_batch = 0
    while (num_batch+1)*batch_size<=num_samples:
        wav_sub_list = wav_list[num_batch*batch_size:(num_batch+1)*batch_size]
        # get wav array
        wav_array,lbl_array = get_array(wav_sub_list)

        wav_array = np.expand_dims(wav_array,-1)# 32,80000,1
        lbl_array = np.expand_dims(lbl_array,-1)# 32,500,1
        # predictions
        preds = model.predict(wav_array)
        preds[preds>=0.5] = 1
        preds[preds<0.5] = 0

        # flatten
        preds = preds.flatten()
        lbl_array = lbl_array.flatten()
        # confusion_mtx
        mtx = np.asarray(cm(lbl_array,preds,2))
        TP += mtx[1][1]
        FP += mtx[0][1]
        TN += mtx[0][0]
        FN += mtx[1][0]

        num_batch += 1 # !!!!!
    if num_batch*batch_size < num_samples:
        wav_sub_list = wav_list[num_batch*batch_size:num_samples]
        # get wav array
        wav_array,lbl_array = get_array(wav_sub_list)

        wav_array = np.expand_dims(wav_array,-1)# 32,80000,1
        lbl_array = np.expand_dims(lbl_array,-1)# 32,500,1
        
        # append
        missing_rows = batch_size-wav_array.shape[0]
        wav_array = np.concatenate((wav_array,np.zeros((missing_rows,wav_array.shape[1],wav_array.shape[2]))),0)

        preds = model.predict(wav_array)
        preds[preds>=0.5] = 1
        preds[preds<0.5] = 0

        # flatten
        preds = preds[0:lbl_array.shape[0]].flatten()
        lbl_array = lbl_array.flatten()

        # confusion_mtx
        mtx = np.asarray(cm(lbl_array,preds,2))
        TP += mtx[1][1]
        FP += mtx[0][1]
        TN += mtx[0][0]
        FN += mtx[1][0]
    
    acc,prec,Recall,F1 = calculate_metrics(TP,FP,TN,FN)
    return (acc,prec,Recall,F1)

def calculate_error(model,cough_wav_list,
                    batch_size=config.config['batch_size'],
                    duration_t=config.duration_t,
                    output_file='./evaluations/duration_error.txt'):
    '''
    calculate the error of cough length between labels and predictions
    
    return a txt file:
    index   |   domain   |  long-name   |   ground_truth_coughs     |   pred_coughs     |   time_error_ms

    para cough_wav_list: list of cough samples
    para batch_size: batch_size
    para duration_t: time interval

    '''
    num_batch = 0
    try:
        f = open(output_file,'w')
    except:
        raise Exception('could not open file: {}'.format(output_file))
    
    time_interval = 1e3*duration_t # ms
    # all the wav:
    while (num_batch+1)*batch_size<=len(cough_wav_list):
        
        wav_sub_list = wav_list[num_batch*batch_size:(num_batch+1)*batch_size]
        # get wav array
        wav_array,lbl_array = get_array(wav_sub_list)

        wav_array = np.expand_dims(wav_array,-1)# 32,80000,1
        lbl_array = np.expand_dims(lbl_array,-1)# 32,500,1
        # predictions
        preds = model.predict(wav_array)
        preds = preds[0:lbl_array.shape[0]]
        preds[preds>=0.5] = 1
        preds[preds<0.5] = 0

        for (i,(domain,wav_name)) in enumerate(wav_sub_list):
            truth_duration = int(np.sum(lbl_array[i].flatten()))
            pred_duration = int(np.sum(preds[i].flatten()))

            time_error = (truth_duration-pred_duration)*time_interval
            f.writelines('%s,%s,%d,%d,%d'%(domain,wav_name,truth_duration,pred_duration,time_error))
            f.writelines('\n')
        
        num_batch += 1 # !!!!!
        print('file predicted:%d / %d'%(num_batch*batch_size,len(cough_wav_list)))
    
    if num_batch*batch_size < len(cough_wav_list):
        wav_sub_list = wav_list[num_batch*batch_size:len(cough_wav_list)]
        # get wav array
        wav_array,lbl_array = get_array(wav_sub_list)

        wav_array = np.expand_dims(wav_array,-1)# 32,80000,1
        lbl_array = np.expand_dims(lbl_array,-1)# 32,500,1
        
        # append
        missing_rows = batch_size-wav_array.shape[0]
        wav_array = np.concatenate((wav_array,np.zeros((missing_rows,wav_array.shape[1],wav_array.shape[2]))),0)

        preds = model.predict(wav_array)
        preds[preds>=0.5] = 1
        preds[preds<0.5] = 0

        for (i,(domain,wav_name)) in enumerate(wav_sub_list):
            truth_duration = int(np.sum(lbl_array[i].flatten()))
            #print(lbl_array[i].shape,lbl_array[i])
            #print(truth_duration)
            pred_duration = int(np.sum(preds[i].flatten()))

            time_error = (truth_duration-pred_duration)*time_interval
            f.writelines('%s,%s,%d,%d,%d'%(domain,wav_name,truth_duration,pred_duration,time_error))
            f.writelines('\n')
        
        print('done!')
        f.close()
    
    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default='./checkpoints/conv525_kernel2/conv_525_5/model/model_weights/model_weights', help="where is the model")
    parser.add_argument('--model_name', type=str, default="conv_model_1d_LSTM", help="model name")
    parser.add_argument('--dropout', type=float, default=0.0, help="add drop out? only valid for some models")
    parser.add_argument('--CV',type=int, default=0, help="CV for validation, default 0")
    parser.add_argument('--sound',type=str, default='cough', help='sound category for evaluating')
    parser.add_argument('--num_samples',type=int, default=32, help='number of samples for evaluating')
    parser.add_argument('--output_file',type=str,default='./evaluations/duration_error.txt', help='where is the output')

    args = parser.parse_args()

    assert args.sound in list(config.SOUND_DICT.keys())
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    

    num_samples= args.num_samples

    wav_list = config.subtype_CV_dict[(args.sound,args.CV)]
    #num_samples = args.num_samples
    model = Load_model(args.model_dir,model_name=args.model_name)

    calculate_error(model,wav_list,output_file=args.output_file)
    
    #(acc,prec,Recall,F1) = evaluate_samples(model,wav_list,num_samples)

    #print('done!')
    #print('='*20)
    #print( "CV:{}    |   sound:{}    |   length of sample list:{}    |   number of samples:{}".format(args.CV,args.sound,len(wav_list),num_samples))

    #print('Accuracy:    %.4f'%(acc))
    #print('Precision:    %.4f'%(prec))
    #print('Recall:    %.4f'%(Recall))
    #print('F1:    %.4f'%(F1))
