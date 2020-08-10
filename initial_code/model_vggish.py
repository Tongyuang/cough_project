from __future__ import print_function
from functools import partial
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, BatchNormalization, Activation, Add,MaxPooling1D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import os

import config
from config import config as c
from config import config_data as c_d
from model_llf import llf_model
from vggish import mel_features
from vggish import vggish_params as vp

frames =96
hop_param=.01
window_param = .025
n_fft = 512
nmels = 64
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01

MIN_SAMPS = 15648 #int((((frames * hop_param) + window_param)) * c['sr']) - 1  # min samps for 96 mel spec frames

def mel_spec(wavs):
    window = int(round(c['sr'] * window_param))
    hop = int(round(c['sr'] * hop_param))

    spectrograms = tf.abs(tf.signal.stft(wavs, frame_length=window, frame_step=hop, fft_length=n_fft,
                                         window_fn=tf.signal.hann_window))
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=nmels,
                                                                        num_spectrogram_bins=spectrograms.shape[-1],
                                                                        sample_rate=c['sr'],
                                                                        lower_edge_hertz=MEL_MIN_HZ,
                                                                        upper_edge_hertz=MEL_MAX_HZ)

    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    return tf.math.log(mel_spectrograms + LOG_OFFSET)

def vggish_model(use_mel_spec=True, llf_pretrain=True, pretrain=True):
    with tf.name_scope('vggish'):

        # inputs = Input(shape=(config.MAX_SAMPS),batch_size=None, name='input_1')
        #
        # llf = tf.expand_dims(mel_spec(inputs),-1) if use_mel_spec else tf.transpose(llf_model(inputs, llf_pretrain),(0,2,1,3))
        #
        inputs = Input(shape=(96,64,1), name='input_1')

        # setup layer params
        conv = partial(Conv2D, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')
        conv_l2 = partial(Conv2D, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',kernel_regularizer=l2(0.01))

        maxpool = partial(MaxPooling2D, pool_size=(2, 2), strides=(2, 2), padding='same')
        dense = partial(Dense, activation='relu', kernel_regularizer=l2(0.01))

        # Block 1
        x = conv(64, name='conv1')(inputs) #(llf)
        x = maxpool(name='pool1')(x)

        # Block 2
        x = conv(128, name='conv2')(x)
        x = maxpool(name='pool2')(x)

        # Block 3
        x = conv(256, name='conv3/conv3_1')(x)
        x = conv(256, name='conv3/conv3_2')(x)
        x = maxpool(name='pool3')(x)

        # Block 4
        x = conv(512, name='conv4/conv4_1_nopt')(x)
        x = conv(512, name='conv4/conv4_2_nopt')(x)
        x = maxpool(name='pool4')(x)


        # FC block
        x = Flatten(name='flatten_')(x)
        x = dense(4096, name='fc1/fc1_1_nopt')(x)
        x = dense(4096, name='fc1/fc1_2_nopt')(x)
        x = dense(128, name='fc2_nopt')(x)

        # Extra layers (not included in vggish pretrained model)
        # nopt = no pre-train
        #x = dense(64, name='fc3_nopt')(x)
        x = Dense(1, activation='sigmoid', name='fc_sigmoid_nopt')(x)

        # Create model
        model = Model(inputs, x, name='model')


        if pretrain:
            model = load_weights(model)

            for layer in model.layers:
                if 'nopt'in layer.name:
                    continue
                layer.trainable=False

    return model

def load_weights(model):

    from tensorflow.python import pywrap_tensorflow

    checkpoint_path = os.path.join(os.path.dirname(c_d['folder_data']), 'vggish_model.ckpt')
    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)

    # If you want to view layer names
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor_name: ", key)
    #     print(reader.get_tensor(key).shape)  # Remove this is you want to print only variable names


    # How we convert a tensorflow checkpoint to keras model
    for layer in model.layers:
        layer_name = layer.name

        # Skip the layers that don't have weights or aren't pretrained
        if 'input' in layer_name or 'pool' in layer_name or 'flatten_' in layer_name or 'sigmoid' in layer.name or 'tf_op_layer' in layer_name:
            continue

        layer_name_ckpt = 'vggish/'+layer.name.replace('_nopt','')
        biases_name = layer_name_ckpt + '/biases'
        weights_name = layer_name_ckpt+'/weights'
        biases = reader.get_tensor(biases_name)
        weights = reader.get_tensor(weights_name)
        model.get_layer(layer_name).set_weights([weights, biases])

    return(model)



if __name__ == '__main__':
    import numpy as np
    samp = np.ones(MIN_SAMPS,dtype=np.float32)
    samp = mel_features.log_mel_spectrogram(samp, audio_sample_rate=c['sr'], log_offset=vp.LOG_OFFSET,
                                            window_length_secs=vp.STFT_WINDOW_LENGTH_SECONDS, hop_length_secs=vp.STFT_HOP_LENGTH_SECONDS,
                                            num_mel_bins=vp.NUM_MEL_BINS, lower_edge_hertz=vp.MEL_MIN_HZ,
                                            upper_edge_hertz=vp.MEL_MAX_HZ)
    samp = np.expand_dims(samp, axis=-1)

    # samp = np.zeros((2, 96,64,1), dtype=np.float32)
    # ms = mel_spec(samp)
    # print(ms.shape)
    model = vggish_model()
    res = model.predict(samp)
    print(res.shape)