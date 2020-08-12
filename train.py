import numpy as np
import os
import time
import argparse

import config
from config import config as c
from config import config_data as c_d

from dataGenerator import train_valid_spliter,DataGenerator
from load_dict import load_dict

from model_conv_1D import conv_model_1d,res_model_1d

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import tensorflow as tf

import socket
from focal_loss import focal_loss

#from metrics import F1Score

# in case GPU memory exceeds
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
from metrics import F1Score

save_weights_on_load = False#True

def lr_scheduler(epoch):
  if epoch < 1500:
    return(1e-4)
  else:
    return(float(1e-4 * K.exp(0.00025 * (1500 - epoch))))

class epoch_calback(tf.keras.callbacks.Callback):

    def __init__(self, model, val_gen, val_gen_full, file_writer, args):
        self.model = model
        self.val_gen = val_gen
        self.val_gen_full = val_gen_full
        self.filewriter = file_writer

    def on_epoch_begin(self, epoch, logs=None):

        #Set back to 0 for testing during this epoch
        self.val_gen.random_seed_idx = 0
        self.val_gen_full.idx = 0

        # Beginning of epoch, set to something random for training
        np.random.seed(int(time.time()))


    def on_epoch_end(self, epoch, logs=None):

        #if 'hyak.local' in socket.gethostname() and epoch < 2:
        #    os.system('nvidia-smi')

        if epoch % 10 != 0:
            return

        loss, acc, prec, recall, f1 = self.model.evaluate(self.val_gen_full, steps=len(self.val_gen_full)-2, verbose=True)
        tf.summary.scalar('epoch_loss', loss, step=epoch)
        tf.summary.scalar('epoch_accuracy', acc, step=epoch)
        tf.summary.scalar('epoch_Precision', prec, step=epoch)
        tf.summary.scalar('epoch_Recall', recall, step=epoch)
        tf.summary.scalar('epoch_F1', f1, step=epoch)

def get_model(args):

    initial_epoch=0

    if args.load_folder is not None:
        folder_head = os.path.dirname(c_d['folder_data'])
        load_folder = os.path.join(folder_head,args.load_folder)
        if args.load_epoch is None:
            checkpoints = np.array([[int(f.split('.')[1]), f] for f in os.listdir(load_folder) if f.endswith('.h5')], dtype=object)
            initial_epoch, model_name = checkpoints[np.argsort(checkpoints[:, 0])][-1]
        else:
            initial_epoch = int(args.load_epoch)
            model_name = 'model.{}.h5'.format(args.load_epoch)

        load_path = os.path.join(load_folder, model_name)
        model = load_model(load_path, {'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D,
                                           'AudioVarianceScaling': AudioVarianceScaling, 'F1Score': F1Score})

        print("Loaded model from:",load_path)
    else:
        if args.model_type == 'conv_model_1D':
            model = conv_model_1d()
        else:
            raise ValueError('Invalid model type specified in input arguments')

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Starting New Model Training")

    model.summary()

    return(model, initial_epoch)

def get_folders(args, folder_head):

    checkpoints_folder = os.path.join(folder_head,'checkpoints')
    logs_folder = os.path.join(folder_head, 'logs')
    os.makedirs(checkpoints_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    i = 0
    model_name = '_'.join([args.model_type, "CV-"+str(args.CV), args.name, str(i)])
    path = os.path.join(checkpoints_folder, model_name)
    while os.path.exists(path):
        i += 1
        model_name = '_'.join([args.model_type, "CV-"+str(args.CV), args.name, str(i)])
        path = os.path.join(checkpoints_folder, model_name)

    ckpt_folder = os.path.join(checkpoints_folder, model_name)
    log_folder = os.path.join(logs_folder, model_name)
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    ckpt_path = os.path.join(ckpt_folder, 'model.{epoch:03d}.h5')

    # file_writer = tf.summary.create_file_writer(log_folder + "/metrics")
    file_writer = tf.summary.create_file_writer(log_folder + "/full_file")
    file_writer.set_as_default()

    return(log_folder, file_writer, ckpt_path)

def train(args):

    CV = args.CV
    folder = args.folder
    model_name = args.model_name
    # deal with folder
    if not os.path.exists(folder):
        raise Exception('folder path not exists!')

    cp_path = folder + '/cp'
    model_path = folder + '/model'
    log_path = folder + '/logs'

    for path in [cp_path,model_path,log_path]:
        if not os.path.exists(path):
            os.mkdir(path)


    dict1 = load_dict()
    # generate train_valid dict
    tvs = train_valid_spliter(dict1,CV)
    train,valid = tvs.gen_train_valid_df()

    # dataloader
    train_data = DataGenerator(TRAIN=True,subtype_dict=train,if_preprocess=True)
    valid_data = DataGenerator(TRAIN=False,subtype_dict=valid,if_preprocess=False)

    # model
    assert model_name in ['conv_model_1d','res_model_1d']

    if model_name == 'conv_model_1d':
        model = conv_model_1d()
    elif model_name == 'res_model_1d':
        model = res_model_1d()

    model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy','Precision','Recall',F1Score()]) 

    # callback

    cp_file = cp_path+'cp.ckpt'
    cp_callback = [ tf.keras.callbacks.ModelCheckpoint(filepath=cp_file,monitor='val_loss',save_best_only=True,verbose=1),
                    tf.keras.callbacks.TensorBoard(log_dir = log_path),
                    tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]


    model.summary()

    model.fit_generator(generator=train_data, steps_per_epoch=50, epochs=2500,validation_data=valid_data, validation_steps=10,callbacks=cp_callback)
                        #validation_freq=10)    
                        #callbacks=callbacks, initial_epoch=initial_epoch)  
    
    savefile = model_path+'.h5'
    model.save(savefile)

    savefile_path = model_path+'/model_weights'
    if not os.path.exists(savefile_path):
        os.mkdir(savefile_path)
    
    model.save_weights(savefile_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_folder', default=None, help="Path to folder where model is - leave blank if doing new training")
    parser.add_argument('--load_epoch', default=None, help="Model epoch to load - leave blank if want most recent")
    parser.add_argument('--gpu_list', default="0", help="Which gpu numbers to use. Should be in format: 1,2 ")
    parser.add_argument('--model_type', default="conv-model", help="Type of model to use. Currently just 'resnet' or 'conv'")
    parser.add_argument('--CV', type=int, default="0", help="Which cross-validation set to use. 0, 1, 2, or 3")
    parser.add_argument('--name', default='no-name', help="use to name this model run")
    parser.add_argument("--llf_pretrain", action='store_true', default=False)
    parser.add_argument("--ft_cs", action='store_true', default=False)
    parser.add_argument('--folder', type=str, default="./checkpoints/model", help="where to store the model and checkpoints")
    parser.add_argument('--model_name', type=str, default="conv_model_1d", help="model name")
    args = parser.parse_args()

    if args.CV not in [0,1,2,3]:
        raise ValueError("CV arg must be 0,1,2,3")

    #config = ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = InteractiveSession(config=config)
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    #tf.device('./gpu:0')
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
