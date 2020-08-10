import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Multiply, GlobalMaxPool1D,
                                     Dense, Dropout, Activation, Reshape, Concatenate, Add, Input)
from depthwise_conv1D import DepthwiseConv1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Initializer
from tensorflow.keras import backend as K
import config


def _compute_audio_fans(shape):
  assert len(shape) == 3, 'This initialization is for Conv1D.'

  len_filter, in_channels, out_channels = shape

  receptive_field_size = len_filter * in_channels  # 원래는 len_filter 여야함!!
  fan_in = in_channels * receptive_field_size
  fan_out = out_channels * receptive_field_size

  return fan_in, fan_out


class AudioVarianceScaling(Initializer):
  """VarianceScaling for Audio"""

  def __init__(self,
               scale=1.0,
               mode="fan_in",
               distribution="truncated_normal",
               seed=None,
               dtype=tf.float32):
    if scale <= 0.:
      raise ValueError("`scale` must be positive float.")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
      raise ValueError("Invalid `mode` argument:", mode)
    distribution = distribution.lower()
    if distribution not in {"uniform", "truncated_normal", "untruncated_normal"}:
      raise ValueError("Invalid `distribution` argument:", distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed
    self.dtype = tf.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    scale = self.scale
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape
    fan_in, fan_out = _compute_audio_fans(scale_shape)
    if self.mode == "fan_in":
      scale /= max(1., fan_in)
    elif self.mode == "fan_out":
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == "normal" or self.distribution == "truncated_normal":
      # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = math.sqrt(scale) / .87962566103423978
      return tf.random.truncated_normal(
        shape, 0.0, stddev, dtype, seed=self.seed)
    elif self.distribution == "untruncated_normal":
      stddev = math.sqrt(scale)
      return tf.random.normal(
        shape, 0.0, stddev, dtype, seed=self.seed)
    else:
      limit = math.sqrt(3.0 * scale)
      return tf.random.uniform(
        shape, -limit, limit, dtype, seed=self.seed)

  def get_config(self):
    return {
      "scale": self.scale,
      "mode": self.mode,
      "distribution": self.distribution,
      "seed": self.seed,
      "dtype": self.dtype.name
    }

def relu6(x):
  """Relu 6
  """
  return K.relu(x, max_value=6.0)

def _conv_block(inputs, filters, kernel, strides, name=''):
  """Convolution Block
  This function defines a 2D convolution operation with BN and relu6.
  # Arguments
      inputs: Tensor, input tensor of conv layer.
      filters: Integer, the dimensionality of the output space.
      kernel: An integer or tuple/list of 2 integers, specifying the
          width and height of the 2D convolution window.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
  # Returns
      Output tensor.
  """
  x = Conv1D(filters, kernel, padding='same', strides=strides,name=f'{name}_conv')(inputs)
  x = BatchNormalization(name=f'{name}_norm0')(x)
  return Activation(relu6, name = f'{name}_relu6-0')(x)


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False, name=''):
  """Bottleneck
  This function defines a basic bottleneck structure.
  # Arguments
      inputs: Tensor, input tensor of conv layer.
      filters: Integer, the dimensionality of the output space.
      kernel: An integer or tuple/list of 2 integers, specifying the
          width and height of the 2D convolution window.
      t: Integer, expansion factor.
          t is always applied to the input size.
      s: An integer or tuple/list of 2 integers,specifying the strides
          of the convolution along the width and height.Can be a single
          integer to specify the same value for all spatial dimensions.
      alpha: Integer, width multiplier.
      r: Boolean, Whether to use the residuals.
  # Returns
      Output tensor.
  """
  # Depth
  tchannel = K.int_shape(inputs)[-1] * t
  # Width
  cchannel = int(filters * alpha)

  x = _conv_block(inputs, tchannel, 1, 1, name)

  x = DepthwiseConv1D(cchannel, kernel, depth_multiplier=1, padding='same', name=f'{name}_dwconv')(x)
  x = BatchNormalization(name= f'{name}_norm1')(x)
  x = Activation(relu6, name=f'{name}_relu6_1')(x)

  x = Conv1D(cchannel, kernel_size=1, strides=1, padding='same', name=f'{name}_ptconv')(x)
  x = BatchNormalization(name= f'{name}_norm2')(x)

  if r:
    x = Add(name = f'{name}_add')([x, inputs])

  return x

#(inputs, filters, kernel, t, alpha, strides, n)
def inverted_residual_block(x, num_features, cfg, name):
  """Inverted Residual Block
  This function defines a sequence of 1 or more identical layers.
  # Arguments
      inputs: Tensor, input tensor of conv layer.
      filters: Integer, the dimensionality of the output space.
      kernel: An integer or tuple/list of 2 integers, specifying the
          width and height of the 2D convolution window.
      t: Integer, expansion factor.
          t is always applied to the input size.
      alpha: Integer, width multiplier.
      s: An integer or tuple/list of 2 integers,specifying the strides
          of the convolution along the width and height.Can be a single
          integer to specify the same value for all spatial dimensions.
      n: Integer, layer repeat times.
  # Returns
      Output tensor.
  """

  x = _bottleneck(x, filters=num_features, kernel=3, t=6, alpha=1, s=1, name=name)
  # x = MaxPool1D(pool_size=3, name=f'{name}_pool')(x)

  # for i in range(1, n):
  #   x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

  return x


def taejun_uniform(scale=2., seed=None):
  return AudioVarianceScaling(scale=scale, mode='fan_in', distribution='uniform', seed=seed)

def squeeze_excitation(x, amplifying_ratio, name):
  num_features = x.shape[-1]
  x = GlobalAvgPool1D(name=f'{name}_squeeze')(x)
  x = Reshape((1, num_features), name=f'{name}_reshape')(x)
  x = Dense(num_features * amplifying_ratio, activation='relu',
            kernel_initializer='glorot_uniform', name=f'{name}_ex0')(x)
  x = Dense(num_features, activation='sigmoid', kernel_initializer='glorot_uniform', name=f'{name}_ex1')(x)
  return x


def basic_block(x, num_features, cfg, name):
  """Block for basic models."""

  x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
             kernel_regularizer=l2(cfg.weight_decay), kernel_initializer=taejun_uniform(), name=f'{name}_conv')(x)
  x = BatchNormalization(name=f'{name}_norm')(x)
  x = Activation('relu', name=f'{name}_relu')(x)
  x = MaxPool1D(pool_size=3, name=f'{name}_pool')(x)
  return x


def se_block(x, num_features, cfg, name):
  """Block for SE models."""
  x = basic_block(x, num_features, cfg, name)
  x = Multiply(name=f'{name}_scale')([x, squeeze_excitation(x, cfg.amplifying_ratio, name)])
  return x


def rese_block(x, num_features, cfg, name):
  """Block for Res-N & ReSE-N models."""
  if num_features != x.shape[-1]:
    shortcut = Conv1D(num_features, kernel_size=1, padding='same', use_bias=True, name=f'{name}_scut_conv',
                      kernel_regularizer=l2(cfg.weight_decay), kernel_initializer='glorot_uniform')(x)
    shortcut = BatchNormalization(name=f'{name}_scut_norm')(shortcut)
  else:
    shortcut = x

  for i in range(cfg.num_convs):
    if i > 0:
      x = Activation('relu', name=f'{name}_relu{i-1}')(x)
      x = Dropout(0.2, name=f'{name}_drop{i-1}')(x)
    x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
               kernel_regularizer=l2(cfg.weight_decay), kernel_initializer=taejun_uniform(), name=f'{name}_conv{i}')(x)
    x = BatchNormalization(name=f'{name}_norm{i}')(x)

  # Add SE if it is ReSE block.
  if cfg.amplifying_ratio:
    x = Multiply(name=f'{name}_scale')([x, squeeze_excitation(x, cfg.amplifying_ratio, name)])

  x = Add(name=f'{name}_scut')([shortcut, x])
  x = Activation('relu', name=f'{name}_relu1')(x)
  x = MaxPool1D(pool_size=3, name=f'{name}_pool')(x)
  return x


def SampleCNN(cfg):
  """Build a SampleCNN model."""
  # Variable-length input for feature visualization.
  x_in = Input(batch_shape=(cfg.batch_size, config.MAX_SAMPS, 1), name='input')

  num_features = cfg.init_features
  x = Conv1D(num_features, kernel_size=3, strides=3, padding='same', use_bias=True,
             kernel_regularizer=l2(cfg.weight_decay), kernel_initializer=taejun_uniform(scale=1.), name='conv0')(x_in)
  x = BatchNormalization(name='norm0')(x)
  x = Activation('relu', name='relu0')(x)

  # Stack convolutional blocks.
  layer_outputs = []
  for i in range(cfg.num_blocks):
    num_features *= 2 if (i == 2 or i == (cfg.num_blocks - 1)) else 1
    x = cfg.block_fn(x, num_features, cfg, f'block{i}')
    layer_outputs.append(x)

  if cfg.multi:  # Use multi-level feature aggregation or not.
    x = Concatenate(name='multi')([GlobalMaxPool1D(name=f'final_pool{i}')(output)
                                   for i, output in enumerate(layer_outputs[-3:])])
  else:
    x = GlobalMaxPool1D(name='final_pool')(x)

  # The final two FCs.
  x = Dense(x.shape[-1], kernel_initializer='glorot_uniform', name='final_fc')(x)
  x = BatchNormalization(name='final_norm')(x)
  x = Activation('relu', name='final_relu')(x)
  if cfg.dropout > 0.:
    x = Dropout(cfg.dropout, name='final_drop')(x)
  x = Dense(cfg.num_classes, kernel_initializer='glorot_uniform', name='logit')(x)
  x = Activation(cfg.activation, name='pred')(x)

  return Model(inputs=[x_in], outputs=[x], name='sample_cnn')


class ModelConfig:
  """
  The default setting is for MTT with se-multi.
  """

  def __init__(self, block='se', multi=True, num_blocks=9, init_features=128, num_convs=1,
               amplifying_ratio=0.125, dropout=0.5, activation='sigmoid', num_classes=1, weight_decay=0.,
               separable=False, batch_size=None):

    # Configure block specific settings.
    if block == 'basic':
      block_fn = basic_block
    elif block.startswith('rese'):
      num_convs = int(block[-1])
      block_fn = rese_block
    elif block.startswith('res'):
      num_convs = int(block[-1])
      amplifying_ratio = None
      block_fn = rese_block
    elif block == 'se':
      block_fn = se_block
    elif block == 'ires':
      block_fn = inverted_residual_block
    else:
      raise Exception(f'Unknown block name: {block}')

    # Overall architecture configurations.
    self.multi = multi
    self.init_features = init_features

    # Block configurations.
    self.block = block
    self.block_fn = block_fn
    self.num_blocks = num_blocks
    self.amplifying_ratio = amplifying_ratio
    self.num_convs = num_convs

    # Training related configurations.
    self.dropout = dropout
    self.activation = activation
    self.num_classes = num_classes
    self.weight_decay = weight_decay
    self.separable = separable
    self.batch_size = batch_size

  def get_signature(self):
    s = self.block
    if self.multi:
      s += '_multi'
    return s

  def print_summary(self):
    print(f'''=> {self.get_signature()} properties:
      block             : {self.block}
      multi             : {self.multi}
      num_blocks        : {self.num_blocks}
      amplifying_ratio  : {self.amplifying_ratio}
      dropout           : {self.dropout}
      activation        : {self.activation}
      num_classes       : {self.num_classes}''')

if __name__ == '__main__':
    # import numpy as np
    # x = np.ones((2,13122,4))

    block = 'ires' #'basic' 'rese' 'res' 'se', 'ires'
    multi = False #True
    num_blocks = 7 #9
    init_features = 16

    cfg = ModelConfig(block=block, multi=multi, num_blocks=num_blocks, init_features=init_features)
    model = SampleCNN(cfg)
    model.summary()

    # model = SampleCNN(ModelConfig(block=block, multi=multi, num_blocks=num_blocks, init_features=init_features))