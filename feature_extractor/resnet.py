from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import squeeze

layers = tf.keras.layers


def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return tf.keras.regularizers.L2(l2_weight_decay) if use_l2_regularizer else None


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_l2_regularizer=True,
                   batch_norm_decay=0.9,
                   batch_norm_epsilon=1e-5):
    """The identity block is the block that has no conv layer at shortcut.
    Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.
      batch_norm_decay: Moment of batch norm layers.
      batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
      Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters1, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(
            input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2a')(
            x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2b')(
            x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2c')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2c')(
            x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_l2_regularizer=True,
               batch_norm_decay=0.9,
               batch_norm_epsilon=1e-5):
    """A block that has a conv layer at shortcut.
    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the second conv layer in the block.
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.
      batch_norm_decay: Moment of batch norm layers.
      batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
      Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters1, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(
            input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2a')(
            x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2b')(
            x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2c')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2c')(
            x)

    shortcut = layers.Conv2D(
        filters3, (1, 1),
        strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '1')(
            input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '1')(
            shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet50(input_tensor,
             use_l2_regularizer=True,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-5):

    x = input_tensor

    if tf.keras.backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
        squeeze_axis = 2
    else:  # channels_last
        bn_axis = 3
        squeeze_axis = 1

    block_config = dict(
        use_l2_regularizer=use_l2_regularizer,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv1')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name='bn_conv1')(
            x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', **block_config)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', **block_config, strides=(2, 1))
    x = identity_block(x, 3, [128, 128, 512], stage=3,
                       block='b', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3,
                       block='c', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3,
                       block='d', **block_config)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', **block_config, strides=(2, 1))
    x = identity_block(x, 3, [256, 256, 1024],
                       stage=4, block='b', **block_config)
    x = identity_block(x, 3, [256, 256, 1024],
                       stage=4, block='c', **block_config)
    x = identity_block(x, 3, [256, 256, 1024],
                       stage=4, block='d', **block_config)
    x = identity_block(x, 3, [256, 256, 1024],
                       stage=4, block='e', **block_config)
    x = identity_block(x, 3, [256, 256, 1024],
                       stage=4, block='f', **block_config)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', **block_config, strides=(2, 1))
    x = identity_block(x, 3, [512, 512, 2048],
                       stage=5, block='b', **block_config)
    x = identity_block(x, 3, [512, 512, 2048],
                       stage=5, block='c', **block_config)

    x = layers.Conv2D(
        512, (3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='compression_layer')(x)
    x = tf.squeeze(x, axis=squeeze_axis)
    return x

def resnet50_slim(input_tensor,
             use_l2_regularizer=True,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-5):
    x = input_tensor
    if tf.keras.backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
        squeeze_axis = 2
    else:  # channels_last
        bn_axis = 3
        squeeze_axis = 1

    block_config = dict(
        use_l2_regularizer=use_l2_regularizer,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(
        16, (7, 7),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv1')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name='bn_conv1')(
            x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1), **block_config)
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='b', **block_config)
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='c', **block_config)

    x = conv_block(x, 3, [32, 32, 128], stage=3, block='a', **block_config, strides=(2, 1))
    x = identity_block(x, 3, [32, 32, 128], stage=3,
                       block='b', **block_config)
    x = identity_block(x, 3, [32, 32, 128], stage=3,
                       block='c', **block_config)
    x = identity_block(x, 3, [32, 32, 128], stage=3,
                       block='d', **block_config)
    x = conv_block(x, 3, [64, 64, 256], stage=4, block='a', **block_config, strides=(2, 1))

    x = identity_block(x, 3, [64, 64, 256],
                       stage=4, block='b', **block_config)
    x = identity_block(x, 3, [64, 64, 256],
                       stage=4, block='c', **block_config)
    x = identity_block(x, 3, [64, 64, 256],
                       stage=4, block='d', **block_config)
    x = identity_block(x, 3, [64, 64, 256],
                       stage=4, block='e', **block_config)
    x = identity_block(x, 3, [64, 64, 256],
                       stage=4, block='f', **block_config)

    x = conv_block(x, 3, [128, 128, 512], stage=5, block='a', **block_config, strides=(2, 1))
    x = identity_block(x, 3, [128, 128, 512],
                       stage=5, block='b', **block_config)
    x = identity_block(x, 3, [128, 128, 512],
                       stage=5, block='c', **block_config)

    x = tf.squeeze(x, axis=squeeze_axis)
    return x

def resnet18(input_tensor,
             use_l2_regularizer=True,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-5):

    x = input_tensor

    if tf.keras.backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
        squeeze_axis = 2
    else:  # channels_last
        bn_axis = 3
        squeeze_axis = 1

    block_config = dict(
        use_l2_regularizer=use_l2_regularizer,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv1')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name='bn_conv1')(
            x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', **block_config)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', **block_config, strides=(2, 1))
    x = identity_block(x, 3, [128, 128, 512], stage=3,
                       block='b', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3,
                       block='c', **block_config)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', **block_config, strides=(2, 1))
    x = identity_block(x, 3, [256, 256, 1024],
                       stage=4, block='b', **block_config)
    x = identity_block(x, 3, [256, 256, 1024],
                       stage=4, block='c', **block_config)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', **block_config, strides=(2, 1))
    x = identity_block(x, 3, [512, 512, 2048],
                       stage=5, block='b', **block_config)
    x = identity_block(x, 3, [512, 512, 2048],
                       stage=5, block='c', **block_config)

    x = layers.Conv2D(
        512, (3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='compression_layer')(x)
    x = tf.squeeze(x, axis=squeeze_axis)

    return x

def resnet18_slim(input_tensor,
             use_l2_regularizer=True,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-5):

    x = input_tensor

    if tf.keras.backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
        squeeze_axis = 2
    else:  # channels_last
        bn_axis = 3
        squeeze_axis = 1

    block_config = dict(
        use_l2_regularizer=use_l2_regularizer,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(
        16, (7, 7),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv1')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name='bn_conv1')(
            x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1), **block_config)
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='b', **block_config)
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='c', **block_config)

    x = conv_block(x, 3, [32, 32, 128], stage=3, block='a', **block_config, strides=(2, 1))
    x = identity_block(x, 3, [32, 32, 128], stage=3,
                       block='b', **block_config)
    x = identity_block(x, 3, [32, 32, 128], stage=3,
                       block='c', **block_config)

    x = conv_block(x, 3, [64, 64, 256], stage=4, block='a', **block_config, strides=(2, 1))
    x = identity_block(x, 3, [64, 64, 256],
                       stage=4, block='b', **block_config)
    x = identity_block(x, 3, [64, 64, 256],
                       stage=4, block='c', **block_config)

    x = conv_block(x, 3, [128, 128, 512], stage=5, block='a', **block_config, strides=(2, 1))
    x = identity_block(x, 3, [128, 128, 512],
                       stage=5, block='b', **block_config)
    x = identity_block(x, 3, [128, 128, 512],
                       stage=5, block='c', **block_config)

    x = tf.squeeze(x, axis=squeeze_axis)

    return x

if __name__ == "__main__":
    input_tensor = tf.zeros(shape=[64, 32, 320, 3], dtype=tf.float32)

    features = resnet18(input_tensor)
    print(features.shape)

    image_input = tf.keras.layers.Input(shape=[32, 320, 3])
    output = resnet18(image_input)
    model = tf.keras.Model(inputs=image_input, outputs=output, name='resnet')
    print('ResNet-18 parameters: ', model.count_params())

    image_input = tf.keras.layers.Input(shape=[32, 320, 3])
    output = resnet18_slim(image_input)
    model = tf.keras.Model(inputs=image_input, outputs=output, name='resnet')
    print('ResNet-18 slim parameters: ', model.count_params())

    image_input = tf.keras.layers.Input(shape=[32, 320, 3])
    output = resnet50(image_input)
    model = tf.keras.Model(inputs=image_input, outputs=output, name='resnet')
    print('ResNet-50 parameters: ', model.count_params())

    image_input = tf.keras.layers.Input(shape=[32, 320, 3])
    output = resnet50_slim(image_input)
    model = tf.keras.Model(inputs=image_input, outputs=output, name='resnet')
    print('ResNet-50 slim parameters: ', model.count_params())
