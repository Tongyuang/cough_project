from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Conv1D, MaxPooling1D,GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from blocks import conv_block_1d,res_block_1d
import config

def model_conv_525(batch_size=config.config['batch_size'],dropout=True,drop_p=0.8):

    X_input = Input(shape=(config.MAX_SAMPS,1), batch_size=batch_size)# (batch, max_samps,1)
    X = conv_block_1d(X_input,filters=8,kernel_size=5,BN=False)
    X = MaxPooling1D(pool_size=5,strides=5)(X)

    for i in range(4):
        X = conv_block_1d(X,filters=(2**(i+4)),kernel_size=5,BN=True)
        X = MaxPooling1D(pool_size=2,strides=2)(X)

    X = conv_block_1d(X,filters=128,kernel_size=5,BN=True)
    X = MaxPooling1D(pool_size=2,strides=2)(X)

    X = Dropout(drop_p)(X)
    X = Dense( units=1, activation = 'sigmoid')(X)
    model = Model(inputs=X_input, outputs=X, name='conv_model_1D')

    return model

def model_conv_complex(batch_size=config.config['batch_size'],dropout=True,drop_p=0.8):

    X_input = Input(shape=(config.MAX_SAMPS,1), batch_size=batch_size)# (batch, max_samps,1)

    X = conv_block_1d(X_input,filters=32,kernel_size=5,BN=False)
    X = MaxPooling1D(pool_size=5,strides=5)(X)

    for i in range(4):
        X = conv_block_1d(X,filters=(2**(i+6)),kernel_size=5,BN=True)
        X = conv_block_1d(X,filters=(2**(i+6)),kernel_size=5,BN=False)
        X = MaxPooling1D(pool_size=2,strides=2)(X)
    X = conv_block_1d(X,filters=512,kernel_size=5,BN=True)
    X = MaxPooling1D(pool_size=2,strides=2)(X)
    # add a bidirectional layer
    #
    #
    #
    
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

    model = model_conv_525()
    model.summary()