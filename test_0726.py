import tensorflow as tf

from model_conv import conv_block_1d

from tensorflow.keras.layers import MaxPooling1D

input_tensor = tf.Variable(tf.random_normal([32,80000,16]))
print(input_tensor.shape)

output_tensor = conv_block_1d(input_tensor,32,5)
print(output_tensor.shape)

output_tensor = MaxPooling1D(pool_size=5,strides=5)(output_tensor)
print(output_tensor.shape)