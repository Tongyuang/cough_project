from __future__ import print_function
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from model_llf import llf_model
import config

### SIMPLE CONV MODEL
def conv_block(X, filters, stage, conv_num):

    X = BatchNormalization(axis=-1, name='bn_conv{}_{}'.format(stage,conv_num))(X)
    X = Conv2D(filters=filters, kernel_size=(3,3), strides=1, padding='same', name='conv{}_{}'.format(stage,conv_num))(X)
    X = Activation('relu')(X)
    return(X)


def conv_model(batch_size=None, llf_pretrain=False):

    X_input = Input(shape=(config.MAX_SAMPS,), batch_size=batch_size)
    X = llf_model(X_input, llf_pretrain)

    for i in range(3):
        X = conv_block(X, 8, 1, i)
    X = MaxPooling2D(pool_size=2)(X)

    for i in range(3):
        X = conv_block(X, 8, 2, i)
    X = MaxPooling2D(pool_size=2)(X)

    for i in range(3):
        X = conv_block(X, 8, 3, i)

    X = GlobalAveragePooling2D(name="avg_pool")(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='conv_model')

    return model

def conv_model_525(batch_size=None, llf_pretrain = False):
    X_input = Input(shape=(config.MAX_SAMPS,), batch_size=batch_size)

if __name__ == '__main__':

    import numpy as np
    model = conv_model()
    model.summary()

    model.predict(np.array([np.zeros(config.MAX_SAMPS,)]*32))



