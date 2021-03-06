import numpy as np
import os
import time
import argparse

import config
from config import config as c
from config import config_data as c_d

from dataGenerator import train_valid_spliter,DataGenerator
from load_dict import load_dict

from models import model_conv_525,res_model_1d,model_conv_525_LSTM,model_conv_525_GRU,My_Unet_1d

from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import socket
from focal_loss import focal_loss

from list_to_csv import list_to_csv
from metrics import F1Score

save_weights_on_load = False#True

def lr_scheduler(epoch):
  if epoch < 1500:
    return(1e-4)
  else:
    return(float(1e-4 * K.exp(0.00025 * (1500 - epoch)))) # .001

class epoch_callback(tf.keras.callbacks.Callback):

    def __init__(self, model, val_gen, result_list):
        self.model = model
        self.val_gen = val_gen
        self.result_list = result_list

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):

        if epoch % 10 != 0:
            return

        scores = self.model.evaluate_generator(self.val_gen, steps=10, verbose=True) # ['loss', 'accuracy', 'precision', 'recall', 'F1']
        self.result_list.append((epoch,scores))


def train(args):

    CV = args.CV
    folder = args.folder
    model_name = args.model_name
    dropout = args.dropout
    # deal with folder
    if not os.path.exists(folder):
        raise Exception('folder path not exists!')

    cp_path = folder + '/cp'
    model_path = folder + '/model'
    log_path = folder + '/logs'
    

    for path in [cp_path,model_path,log_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    
    
    # generate train_valid dict
    tvs = train_valid_spliter(config.subtype_CV_dict,CV)
    train,valid = tvs.gen_train_valid_df()

    use_loud_noise = args.use_loud_noise
    samps_per_subtype_idx = args.samps_per_subtype_idx

    # dataloader
    train_data = DataGenerator(TRAIN=True,subtype_dict=train,if_preprocess=True,use_loud_noise=use_loud_noise,samps_per_subtype_idx=samps_per_subtype_idx)
    valid_data = DataGenerator(TRAIN=False,subtype_dict=valid,if_preprocess=False,use_loud_noise=False,samps_per_subtype_idx=samps_per_subtype_idx)

    # model
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
    
    assert args.loss in ['binary_crossentropy', 'focal_loss']
    if args.loss == 'binary_crossentropy':
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','Precision','Recall',F1Score()]) 
    elif args.loss == 'focal_loss':
        model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy','Precision','Recall',F1Score()]) 
    # callback

    output_list = []
    cp_file = cp_path+'/cp.ckpt'
    cp_callback = [ tf.keras.callbacks.ModelCheckpoint(filepath=cp_file,monitor='Recall',save_weights_only=True,save_best_only=True,verbose=2),
                    tf.keras.callbacks.TensorBoard(log_dir = log_path),
                    tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
                    epoch_callback(model,valid_data,output_list)]


    model.summary()

    initial_epoch = args.initial_epoch
    if len(os.listdir(cp_path))>0 and args.continue_training:
        # continue
        try:
            model.load_weights(cp_file)
        except:
            initial_epoch = 0
            raise Exception('failed to load checkpoint from current folder, will start training without initial checkpoints.')
            

    model.fit_generator(generator=train_data, steps_per_epoch=args.steps_each_epoch, epochs=args.epochs,validation_data=valid_data, validation_steps=10,callbacks=cp_callback,initial_epoch=initial_epoch,verbose=2)
                        #validation_freq=10)    
                        #callbacks=callbacks, initial_epoch=initial_epoch)  
    
    savefile = model_path+'/model.h5'
    model.save(savefile)

    savefile_path = model_path+'/model_weights'
    if not os.path.exists(savefile_path):
        os.mkdir(savefile_path)
    
    model.save_weights(savefile_path+'/model_weights')

    output_csv_file = folder + '/metrics.csv'
    list_to_csv(output_list,output_csv_file)
    print('Done')
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_list', default="0", help="Which gpu numbers to use. Should be in format: 1,2 ")
    parser.add_argument('--CV', type=int, default="0", help="Which cross-validation set to use. 0, 1, 2, or 3")
    parser.add_argument("--use_loud_noise", default=False, help="whether to use loud noise when generating data")
    parser.add_argument("--loss", default='binary_crossentropy', help="loss function, must be in [binary_crossentropy, focal_loss]")
    parser.add_argument("--samps_per_subtype_idx", type=int, default=0,help="index to samps_per_subtype")
    parser.add_argument('--folder', type=str, default="./checkpoints/model", help="where to store the model and checkpoints")
    parser.add_argument('--model_name', type=str, default="conv_model_1d", help="model name")
    parser.add_argument('--dropout', type=float, default=0.0, help="add drop out? only valid for some models")
    parser.add_argument('--steps_each_epoch', type=int, default=50, help="steps each epoch")
    parser.add_argument('--epochs', type=int,default=2500, help="number of epochs")
    parser.add_argument('--continue_training',type=bool, default=True, help='if continue training from previous checkpoints')
    parser.add_argument('--initial_epoch',type=int, default=0, help='initial epoch')
    args = parser.parse_args()

    if args.CV not in [0,1,2,3]:
        raise ValueError("CV arg must be 0,1,2,3")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    
    #num_gpus = len(args.gpu_list.split(','))
    #num_gpus = max(1, num_gpus)

    #if socket.gethostname() == 'area51.cs.washington.edu':
    #    if num_gpus == 1:
    #        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    #        print("Setting gpus to:", args.gpu_list)
    #    else:
    #        print("incorrect number of gpus for area51. args.gpu_list:", args.gpu_list)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # use cpu in case using gpu to train

    train(args)
