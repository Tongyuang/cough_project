import tensorflow as tf

a = tf.random.normal([32,125,128])
print(a.shape)

#a = tf.keras.layers.UpSampling1D(size=2)(a)
#print(a.shape)

b = tf.keras.layers.MaxPooling1D(pool_size=2,strides=2,padding='same')(a)
print(b.shape)

b = tf.keras.layers.UpSampling1D(size=2)(b)
print(b.shape)

b = tf.keras.layers.Cropping1D(cropping=(1,0))(b)
print(b.shape)

c = tf.keras.layers.Concatenate(axis=-1)([a,b])
print(c.shape)