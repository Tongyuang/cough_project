from __future__ import print_function
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Conv1D, MaxPooling1D,GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import config

### SIMPLE CONV MODEL

def conv_block_1d(X, filters, kernel_size,BN = True):
    # input shape: (batch, steps, channels)
    # 1d Conv
    X = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(X)
    # BN
    if BN:
    # BatchNormalization
        X = BatchNormalization(axis=-1, )(X)
    # activation
    X = Activation('relu')(X)
    
    return X

class add_Tensor(tf.keras.layers.Layer):
    def __init__(self,X1):
      super(add_Tensor, self).__init__()
      self.X1 = X1

    def call(self, inputs):
      return inputs+self.X1

def res_block_1d(X,filters,kernel_size,shortcut=True):
    # input shape: (batch, steps, channels)
    # 
    channels = int(X.shape[2])
    #print(channels)

    X1 = BatchNormalization(axis=-1, )(X)

    X1 = Conv1D(filters=filters,kernel_size=1, strides=1, padding='same')(X1) # output shape: (batch, steps, filters)
    X1 = Activation('relu')(X1)
   
    X1 = Conv1D(filters=filters,kernel_size=kernel_size, strides=1, padding='same')(X1) # output shape: (batch, steps, filters)
    X1 = Activation('relu')(X1)

    X1 = Conv1D(filters=channels, kernel_size=1, strides=1, padding='same')(X1) # (batch, steps, channels)

    if shortcut:
        X1 = add_Tensor(X)(X1)

    X1 = Activation('relu')(X1)
    return X1 # (batch,steps,channels) , won't change channels

def conv_model_1d(batch_size=config.config['batch_size'],dropout=True,drop_p=0.8):

    X_input = Input(shape=(config.MAX_SAMPS,1), batch_size=batch_size)# (batch, max_samps,1)

    X = conv_block_1d(X_input,filters=32,kernel_size=5,BN=False)
    X = MaxPooling1D(pool_size=5,strides=5)(X)

    for i in range(4):
        X = conv_block_1d(X,filters=(2**(i+6)),kernel_size=5,BN=True)
        X = conv_block_1d(X,filters=(2**(i+6)),kernel_size=5,BN=False)
        X = MaxPooling1D(pool_size=2,strides=2)(X)
    X = conv_block_1d(X,filters=512,kernel_size=5,BN=True)
    X = MaxPooling1D(pool_size=2,strides=2)(X)
    X = Dropout(drop_p)(X)
    X = Dense( units=1, activation = 'sigmoid')(X)
    model = Model(inputs=X_input, outputs=X, name='conv_model_1D')

    return model

def res_model_1d(batch_size=config.config['batch_size']):

    X_input = Input(shape=(config.MAX_SAMPS,1), batch_size=batch_size)# (batch, max_samps,1)

    X = conv_block_1d(X_input,filters=64,kernel_size=5,BN=False)
    X = MaxPooling1D(pool_size=5,strides=5)(X)

    X = conv_block_1d(X, filters=128, kernel_size=2, BN=False)
    X = MaxPooling1D(pool_size=2,strides=2)(X)

    X = res_block_1d(X,filters=64,kernel_size=2)
    X = conv_block_1d(X, filters=256, kernel_size=1, BN=False)
    X = MaxPooling1D(pool_size=2,strides=2)(X)

    X = res_block_1d(X,filters=128,kernel_size=2)
    X = conv_block_1d(X, filters=512, kernel_size=1, BN=False)
    X = MaxPooling1D(pool_size=2,strides=2)(X)

    X = res_block_1d(X,filters=256,kernel_size=2)
    X = conv_block_1d(X, filters=1024, kernel_size=1, BN=False)
    X = MaxPooling1D(pool_size=2,strides=2)(X)

    X = res_block_1d(X,filters=512,kernel_size=2)
    X = conv_block_1d(X, filters=1024, kernel_size=1, BN=False)
    X = MaxPooling1D(pool_size=2,strides=2)(X)

    X = Dense(units=1, activation = 'sigmoid')(X)

    model = Model(inputs=X_input, outputs=X, name='res_model_1D')

    return model


if __name__ == '__main__':

    import numpy as np
    model = res_model_1d()
    model.summary()

    #model.predict(tf.Variable(tf.random_normal([32,80000,1])),steps=1)