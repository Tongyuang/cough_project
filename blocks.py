from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Conv1D, MaxPooling1D,GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import config


class add_Tensor(tf.keras.layers.Layer):
    def __init__(self,X1):
      super(add_Tensor, self).__init__()
      self.X1 = X1

    def call(self, inputs):
      return inputs+self.X1


def conv_block_1d(X, filters, kernel_size,BN = True, padding='same',strides=1,activation='relu'):
    # input shape: (batch, steps, channels)
    # 1d Conv
    X = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(X)
    # BN
    if BN:
    # BatchNormalization
        X = BatchNormalization(axis=-1, )(X)
    # activation
    X = Activation(activation)(X)
    
    return X

def res_block_1d(X,filters,kernel_size,shortcut=True,padding='same',strides=1,activation='relu'):
    # input shape: (batch, steps, channels)
    # 
    channels = int(X.shape[2])
    #print(channels)

    X1 = BatchNormalization(axis=-1, )(X)

    X1 = Conv1D(filters=filters,kernel_size=1, strides=strides, padding=padding)(X1) # output shape: (batch, steps, filters)
    X1 = Activation(activation)(X1)
   
    X1 = Conv1D(filters=filters,kernel_size=kernel_size, strides=strides, padding=padding)(X1) # output shape: (batch, steps, filters)
    X1 = Activation(activation)(X1)

    X1 = Conv1D(filters=channels, kernel_size=1, strides=strides, padding=padding)(X1) # (batch, steps, channels)

    if shortcut:
        X1 = add_Tensor(X)(X1)

    X1 = Activation(activation)(X1)

    return X1 # (batch,steps,channels) , won't change channels



