from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense
import tensorflow as tf
import config
from config import config as c

import numpy as np
import pickle

def save_llf_pretrain_dict():
    import torch
    path2 = r'C:\Users\mattw12\Downloads\AclNet_training\audio_classification\apps\audioset_models\aclnet_2_distill_from_aclres18\checkpoint.dat'
    chkpt = torch.load(open(path2, 'rb'))

    #print the llf weights
    # for k, v in chkpt['state_dict'].items():
    #     if "low_level_features" in k:
    #         print(k, v.shape)

    weights = {}
    for i, (name, k) in enumerate([('llf_conv0', 'low_level_features.0.0.weight'),
                                   ('llf_conv1', 'low_level_features.1.0.weight'),
                                   ('llf_bn0_gamma', 'low_level_features.0.1.weight'),
                                   ('llf_bn0_beta', 'low_level_features.0.1.bias'),
                                   ('llf_bn0_moving-mean', 'low_level_features.0.1.running_mean'),
                                   ('llf_bn0_moving-var', 'low_level_features.0.1.running_var'),
                                   ('llf_bn1_gamma', 'low_level_features.1.1.weight'),
                                   ('llf_bn1_beta', 'low_level_features.1.1.bias'),
                                   ('llf_bn1_moving-mean', 'low_level_features.1.1.running_mean'),
                                   ('llf_bn1_moving-var', 'low_level_features.1.1.running_var')]):
        val = chkpt['state_dict']['module.' + k].cpu().data.numpy()
        if i == 0:
            # This is the first conv llf kernel
            val = np.transpose(val, (1, 3, 2, 0))
        elif i == 1:
            # This is the second conv llf kernel
            val = np.transpose(val, (2, 3, 1, 0))
        weights[name] = val.copy()

    with open(c['llf_pretrain_dict'], 'wb') as handle:
        pickle.dump(weights, handle)

def get_llf_pretrain_weights(pretrain=False):

    msg = "Using pretraining for low level features" if pretrain else "No pretraining for low level features"
    print(msg)

    with open(c['llf_pretrain_dict'], 'rb') as handle:
        weights = pickle.load(handle)

    for k in weights.keys():
        weights[k] = tf.constant_initializer(weights[k]) if pretrain else None
    return(weights)

def llf_model(x, pretrain=False):

    weights = get_llf_pretrain_weights(pretrain=pretrain)

    # input: (None, 15648)
    x = Reshape([1] + x.shape.as_list()[1:] + [1])(x) #out: (None, 1, 15648, 1)

    # These are basically just 1D convs (there's no support for 1D  conv in tflite)
    x = Conv2D(filters=64, kernel_size=(1, 33), strides=(1,config.llf_stride_1), padding='same', name='llf-conv0',
               use_bias=False,kernel_initializer=weights['llf_conv0'], trainable=not(pretrain))(x) #out: (None, 1, 978, 64)
    x = BatchNormalization(axis=-1, momentum=(1-.1), epsilon=1e-5, name='llf-bn0',
                           gamma_initializer=weights['llf_bn0_gamma'],
                           beta_initializer=weights['llf_bn0_beta'],
                           moving_mean_initializer=weights['llf_bn0_moving-mean'],
                           moving_variance_initializer=weights['llf_bn0_moving-var'],
                           trainable=not(pretrain))(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=128, kernel_size=(1, 5), strides=(1,config.llf_stride_2), padding='same', name='llf-conv1',
               use_bias=False, kernel_initializer=weights['llf_conv1'],trainable=not(pretrain))(x) #out: (None, 1, 489, 128)
    x = BatchNormalization(axis=-1, momentum=(1-.1), epsilon=1e-5, name='llf-bn1',
                           gamma_initializer=weights['llf_bn1_gamma'],
                           beta_initializer=weights['llf_bn1_beta'],
                           moving_mean_initializer=weights['llf_bn1_moving-mean'],
                           moving_variance_initializer=weights['llf_bn1_moving-var'],
                           trainable=not(pretrain))(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(1, config.llf_max_pool))(x) #out:(None, 1, 48, 128)

    x = Reshape(x.shape.as_list()[2:] + [1])(x)  # out:(None, 48, 128, 1)

    return(x)

def llf_fc(batch_size=None):
    X_input = Input(shape=(config.MAX_SAMPS,), batch_size=batch_size)
    X = llf_model(X_input, pretrain=True)
    X = Flatten()(X)
    X = Dense(128, activation='relu', name='fc_1')(X)
    X = Dense(1, activation='sigmoid', name='fc_sig')(X)

    model = Model(inputs=X_input, outputs=X, name='llf_fc')
    return(model)


if __name__ == '__main__':

    # save_llf_pretrain_dict()

    # pretrain=True
    #
    # X_input = Input(shape=(config.MAX_SAMPS,))
    # x = llf_model(X_input, pretrain)
    # model = Model(inputs=X_input, outputs=x, name='conv_model')
    # model.summary()
    #
    # #Check correcty being set
    # print([(v.name, v.shape, v[0,0,0,:5]) for v in model.variables if 'conv' in v.name])
    # print([(v.name, v.shape, v[:5]) for v in model.variables if 'bn' in v.name])

    model = llf_fc()
    model.summary()


