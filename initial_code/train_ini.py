import numpy as np
import os
import time
import socket
import argparse

import config
from config import config as c
from config import config_data as c_d
from model_conv import conv_model
from model_mobilenet import MobileNetv2, MobileNetv2_simple, relu6, DepthwiseConv2D
from model_vggish import vggish_model
from model_samp_cnn import SampleCNN, ModelConfig, AudioVarianceScaling
from data_generator import DataGenerator
from evaluate_full_file import DataGeneratorFullFile
from cross_validation import get_typ_and_domain_probabilities_dicts, get_file_list
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import tensorflow as tf

from metrics import F1Score

save_weights_on_load = False#True

def lr_scheduler(epoch):
  if epoch < 2500:
    return(0.001)
  else:
    return(float(0.001 * K.exp(0.00025 * (2500 - epoch))))

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
        # print("set random seed random")

    def on_epoch_end(self, epoch, logs=None):

        if 'hyak.local' in socket.gethostname() and epoch < 2:
            os.system('nvidia-smi')

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
        if args.model_type == 'mobilenet-simple':
            model = MobileNetv2_simple(batch_size=None, llf_pretrain=args.llf_pretrain)
            print("Model = MobileNet Simple")
        elif args.model_type == 'mobilenet':
            model = MobileNetv2(batch_size=None, llf_pretrain=args.llf_pretrain)
            print("Model = MobileNet")
        elif args.model_type == 'conv-model':
            model = conv_model(batch_size=None, llf_pretrain=args.llf_pretrain)
        elif args.model_type == 'sample-cnn':
            model = SampleCNN(ModelConfig(block='basic', multi=False, num_blocks=7, init_features=16))
        elif args.model_type == 'vggish':
            model = vggish_model()
        else:
            raise ValueError('Invalid model type specified in input arguments')

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', F1Score()])

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

    domains, subtypes_dict, percentages_dict = get_typ_and_domain_probabilities_dicts()
    files_tr, files_te = get_file_list(CV=args.CV, TEST=True)

    tr_generator = DataGenerator(args.model_type, True, domains, subtypes_dict, percentages_dict, files_tr)
    val_generator = DataGenerator(args.model_type, False, domains, subtypes_dict, percentages_dict, files_te)
    val_generator_full = DataGeneratorFullFile(args.model_type, args.CV)

    folder_head = os.path.dirname(c_d['folder_data'])
    if 'hyak.local' in socket.gethostname():
        folder_head = folder_head.replace('scr', 'gscratch')

    model, initial_epoch = get_model(args)

    # model.summary()
    log_folder, file_writer, ckpt_path = get_folders(args, folder_head)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=0, save_best_only=False,
                                                    save_weights_only=False, mode='auto', period=50),
                 tf.keras.callbacks.TensorBoard(log_dir=log_folder, profile_batch=0),
                 tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
                 epoch_calback(model, val_generator, val_generator_full, file_writer, args)]
                 # keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=30, min_lr=.000008,min_delta=.005)]

    model.fit_generator(generator=tr_generator, steps_per_epoch=100, epochs=100000,
                        validation_data=val_generator, validation_steps=500,
                        validation_freq=10,
                        callbacks=callbacks, initial_epoch=initial_epoch)

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
    args = parser.parse_args()

    if args.CV not in [0,1,2,3]:
        raise ValueError("CV arg must be 0,1,2,3")

    num_gpus = len(args.gpu_list.split(','))
    num_gpus = max(1, num_gpus)

    if socket.gethostname() == 'area51.cs.washington.edu':
        if num_gpus == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
            print("Setting gpus to:", args.gpu_list)
        else:
            print("incorrect number of gpus for area51. args.gpu_list:", args.gpu_list)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # use cpu in case using gpu to train

    train(args)
