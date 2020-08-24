from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input, Conv1D, MaxPooling1D,GlobalAveragePooling1D, Dropout, LSTM, GRU, UpSampling1D, Cropping1D, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
from blocks import conv_block_1d,res_block_1d
import config

def model_conv_525(batch_size=config.config['batch_size'],dropout=True,drop_p=0.2):

    X_input = Input(shape=(config.MAX_SAMPS,1), batch_size=batch_size)# (batch, max_samps,1)
    X = conv_block_1d(X_input,filters=8,kernel_size=5,BN=False)
    X = MaxPooling1D(pool_size=5,strides=5)(X)

    for i in range(4):
        X = conv_block_1d(X,filters=(2**(i+4)),kernel_size=2,BN=True)
        X = MaxPooling1D(pool_size=2,strides=2)(X)

    X = conv_block_1d(X,filters=128,kernel_size=2,BN=True)
    X = MaxPooling1D(pool_size=2,strides=2)(X)

    X = Dropout(drop_p)(X)
    X = Dense( units=1, activation = 'sigmoid')(X)
    model = Model(inputs=X_input, outputs=X, name='conv_model_1D')

    return model

def model_conv_complex(batch_size=config.config['batch_size'],dropout=True,drop_p=0.2):

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

def model_conv_525_LSTM(batch_size=config.config['batch_size'],units=1,dropout=0.2,Train=True):
    # LSTM as output layer
    X_input = Input(shape=(config.MAX_SAMPS,1), batch_size=batch_size)# (batch, max_samps,1)
    X = conv_block_1d(X_input,filters=8,kernel_size=5,BN=False)
    X = MaxPooling1D(pool_size=5,strides=5)(X)

    for i in range(4):
        X = conv_block_1d(X,filters=(2**(i+4)),kernel_size=2,BN=True)
        X = MaxPooling1D(pool_size=2,strides=2)(X)

    X = conv_block_1d(X,filters=128,kernel_size=2,BN=True)
    X = MaxPooling1D(pool_size=2,strides=2)(X)

    if not Train:
        lstm = LSTM(units=units,return_sequences=True,return_state=True,activation="sigmoid")
    else:
        lstm = LSTM(units=units,return_sequences=True,return_state=True,activation="sigmoid",dropout=dropout)
    whole_seq_X, final_memory_state, final_carry_state = lstm(X)

    model = Model(inputs=X_input, outputs=whole_seq_X, name='conv_model_1D_LSTM')

    return model

def model_conv_525_GRU(batch_size=config.config['batch_size'],units=1,dropout=0.2):
    # LSTM as output layer
    X_input = Input(shape=(config.MAX_SAMPS,1), batch_size=batch_size)# (batch, max_samps,1)
    X = conv_block_1d(X_input,filters=8,kernel_size=5,BN=False)
    X = MaxPooling1D(pool_size=5,strides=5)(X)

    for i in range(4):
        X = conv_block_1d(X,filters=(2**(i+4)),kernel_size=2,BN=True)
        X = MaxPooling1D(pool_size=2,strides=2)(X)

    X = conv_block_1d(X,filters=128,kernel_size=2,BN=True)
    X = MaxPooling1D(pool_size=2,strides=2)(X)

   
    gru = GRU(units=units,return_sequences=True,return_state=True,activation="sigmoid",dropout=dropout)
    whole_seq_X, final_state = gru(X)
    model = Model(inputs=X_input, outputs=whole_seq_X, name='conv_model_1D_GRU')

    return model

def My_Unet_1d(batch_size=config.config['batch_size'],drop_p=0.0,output_layer=None,units=1):

    # Encoder 544
    # Output layer: LSTM
    def Encoder(X_input,drop_p=drop_p):
        # (batch, max_samps,1)
        X = conv_block_1d(X_input,filters=8,kernel_size=5,BN=False)
        X = MaxPooling1D(pool_size=5,strides=5)(X)


        X = conv_block_1d(X,filters=16,kernel_size=4,BN=True)
        X = MaxPooling1D(pool_size=4,strides=4)(X)

        X = conv_block_1d(X,filters=32,kernel_size=4,BN=True)
        X = MaxPooling1D(pool_size=4,strides=4)(X)

        X = conv_block_1d(X,filters=32,kernel_size=2,BN=True)
        X = MaxPooling1D(pool_size=2,strides=2)(X)

        X = Dropout(drop_p)(X) if drop_p>0 else X
        return X # 32,500,32

    def Decoder(X,features=64,BN=True,drop_p = drop_p,depth=4):
        # Encoder output:(32,500,64)
        # U-net Decoder
        def down_samp(X,filters,BN,dropout=drop_p):
            X = conv_block_1d(X,filters=filters,kernel_size=2,BN=BN)
            X = Dropout(drop_p)(X) if drop_p>0 else X
            return MaxPooling1D(pool_size=2,strides=2,padding='same')(X),X

        def up_samp(X,skip_tensor_X,filters,BN,dropout=drop_p):
            X = UpSampling1D(size=2)(X) # 32,500,64 -> 32,1000,64 

            s_width = skip_tensor_X.shape[1]
            width = X.shape[1]
            #print(s_width,width)
            assert (width-s_width >= 0 and width-s_width <= 1)
            if width - s_width == 0 :
                Y = X
            else:
                Y = Cropping1D(cropping=(1,0))(X)

            X = Concatenate(axis=-1)([skip_tensor_X,Y])

            X = conv_block_1d(X,filters=filters,kernel_size=2,BN=BN)
            X = Dropout(drop_p)(X) if drop_p>0 else X
            return X
        
        skips = []
        filters = features
        for i in range(depth):
            X,X0 = down_samp(X,filters=filters,BN=BN)
            skips.append(X0)
            filters *= 2
        
        X = conv_block_1d(X,filters=filters,kernel_size=2,BN=BN)

        for i in range(depth):
            filters //= 2
            X = up_samp(X,skips[depth-i-1],filters=filters,BN=BN)

        return X

    X_input = Input(shape=(config.MAX_SAMPS,1), batch_size=batch_size)# (batch, max_samps,1)
    X = Encoder(X_input,drop_p=0)
    X = Decoder(X,features=32,BN=True,drop_p=0)

    if output_layer is None:
        X = Dense(units=1, activation = 'sigmoid')(X)
        model = Model(inputs=X_input, outputs=X, name='My_Unet_1d')
    else:
        assert output_layer in ['LSTM','GRU']
        if output_layer == 'LSTM':
            lstm = LSTM(units=units,return_sequences=True,return_state=True,activation="sigmoid",dropout=drop_p)
            whole_seq_X, final_memory_state, final_carry_state = lstm(X)
            model = Model(inputs=X_input, outputs=whole_seq_X, name='My_Unet_1d')
        elif output_layer == 'GRU':
            gru = GRU(units=units,return_sequences=True,return_state=True,activation="sigmoid",dropout=drop_p)
            whole_seq_X, final_state = gru(X)
            model = Model(inputs=X_input, outputs=whole_seq_X, name='My_Unet_1d')

    return model



if __name__ == '__main__':

    model = My_Unet_1d(output_layer='LSTM')
    model.summary()