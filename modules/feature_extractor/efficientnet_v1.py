from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from typing import Any, Dict, Optional, Text, Tuple

from absl import logging
from dataclasses import dataclass
import tensorflow as tf

from modules.parameters import base_config


@dataclass
class BlockConfig(base_config.Config):
    """Config for a single MB Conv Block."""
    input_filters: int = 0
    output_filters: int = 0
    kernel_size: int = 3
    num_repeat: int = 1
    expand_ratio: int = 1
    strides: Tuple[int, int] = (1, 1)
    se_ratio: Optional[float] = None
    id_skip: bool = True
    fused_conv: bool = False
    conv_type: str = 'depthwise'


@dataclass
class ModelConfig(base_config.Config):
    """Default Config for Efficientnet-B0."""
    width_coefficient: float = 1.0
    depth_coefficient: float = 1.0
    resolution: int = 224
    dropout_rate: float = 0.2
    blocks: Tuple[BlockConfig, ...] = (
        # (input_filters, output_filters, kernel_size, num_repeat,
        #  expand_ratio, strides, se_ratio)
        # pylint: disable=bad-whitespace
        BlockConfig.from_args(32, 16, 3, 1, 1, (1, 1), 0.25),
        BlockConfig.from_args(16, 24, 3, 2, 6, (2, 2), 0.25),
        BlockConfig.from_args(24, 40, 5, 2, 6, (2, 1), 0.25),
        BlockConfig.from_args(40, 80, 3, 3, 6, (2, 1), 0.25),
        BlockConfig.from_args(80, 112, 5, 3, 6, (1, 1), 0.25),
        BlockConfig.from_args(112, 192, 5, 4, 6, (2, 1), 0.25),
        BlockConfig.from_args(192, 320, 3, 1, 6, (1, 1), 0.25),
        # pylint: enable=bad-whitespace
    )
    stem_base_filters: int = 32
    top_base_filters: int = 1280
    activation: str = 'simple_swish'
    batch_norm: str = 'default'
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-3
    # While the original implementation used a weight decay of 1e-5,
    # tf.nn.l2_loss divides it by 2, so we halve this to compensate in Keras
    weight_decay: float = 5e-6
    drop_connect_rate: float = 0.2
    depth_divisor: int = 8
    min_depth: Optional[int] = None
    use_se: bool = True
    input_channels: int = 3
    num_classes: int = 1000
    model_name: str = 'efficientnet'
    rescale_input: bool = True
    data_format: str = 'channels_last'
    dtype: str = 'float32'

@dataclass
class ModelConfigSlim(base_config.Config):
    """Default Config for Efficientnet-B0."""
    width_coefficient: float = 1.0
    depth_coefficient: float = 1.0
    resolution: int = 224
    dropout_rate: float = 0.2
    blocks: Tuple[BlockConfig, ...] = (
        # (input_filters, output_filters, kernel_size, num_repeat,
        #  expand_ratio, strides, se_ratio)
        # pylint: disable=bad-whitespace
        BlockConfig.from_args(32, 16, 3, 1, 1, (1, 1), 0.25),
        BlockConfig.from_args(16, 24, 3, 2, 4, (2, 2), 0.25),
        BlockConfig.from_args(24, 40, 5, 2, 4, (2, 1), 0.25),
        BlockConfig.from_args(40, 80, 3, 2, 4, (2, 1), 0.25),
        BlockConfig.from_args(80, 112, 5, 2, 4, (1, 1), 0.25),
        BlockConfig.from_args(112, 192, 5, 2, 4, (2, 1), 0.25),
        BlockConfig.from_args(192, 320, 3, 1, 4, (1, 1), 0.25),
        # pylint: enable=bad-whitespace
    )
    stem_base_filters: int = 32
    top_base_filters: int = 512
    activation: str = 'simple_swish'
    batch_norm: str = 'default'
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-3
    # While the original implementation used a weight decay of 1e-5,
    # tf.nn.l2_loss divides it by 2, so we halve this to compensate in Keras
    weight_decay: float = 5e-6
    drop_connect_rate: float = 0.2
    depth_divisor: int = 8
    min_depth: Optional[int] = None
    use_se: bool = True
    input_channels: int = 3
    num_classes: int = 1000
    model_name: str = 'efficientnet'
    rescale_input: bool = True
    data_format: str = 'channels_last'
    dtype: str = 'float32'

@dataclass
class ModelConfigTiny(base_config.Config):
    """Default Config for Efficientnet-B0."""
    width_coefficient: float = 1.0
    depth_coefficient: float = 1.0
    resolution: int = 224
    dropout_rate: float = 0.2
    blocks: Tuple[BlockConfig, ...] = (
        # (input_filters, output_filters, kernel_size, num_repeat,
        #  expand_ratio, strides, se_ratio)
        # pylint: disable=bad-whitespace
        BlockConfig.from_args(32, 16, 3, 1, 1, (1, 1), 0.25),
        BlockConfig.from_args(16, 24, 3, 2, 4, (2, 2), 0.25),
        BlockConfig.from_args(24, 40, 3, 2, 1, (2, 1), 0.25),
        BlockConfig.from_args(40, 80, 3, 2, 1, (2, 1), 0.25),
        BlockConfig.from_args(80, 112, 5, 2, 4, (1, 1), 0.25),
        BlockConfig.from_args(112, 192, 3, 2, 1, (2, 1), 0.25),
        BlockConfig.from_args(192, 320, 3, 1, 4, (1, 1), 0.25),
        # pylint: enable=bad-whitespace
    )
    stem_base_filters: int = 32
    top_base_filters: int = 512
    activation: str = 'simple_swish'
    batch_norm: str = 'default'
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-3
    # While the original implementation used a weight decay of 1e-5,
    # tf.nn.l2_loss divides it by 2, so we halve this to compensate in Keras
    weight_decay: float = 5e-6
    drop_connect_rate: float = 0.2
    depth_divisor: int = 8
    min_depth: Optional[int] = None
    use_se: bool = True
    input_channels: int = 3
    num_classes: int = 1000
    model_name: str = 'efficientnet'
    rescale_input: bool = True
    data_format: str = 'channels_last'
    dtype: str = 'float32'


MODEL_CONFIGS = {
    # (width, depth, resolution, dropout)
    'efficientnet-b0': ModelConfig.from_args(1.0, 1.0, 224, 0.2),
    'efficientnet-b1': ModelConfig.from_args(1.0, 1.1, 240, 0.2),
    'efficientnet-b2': ModelConfig.from_args(1.1, 1.2, 260, 0.3),
    'efficientnet-b3': ModelConfig.from_args(1.2, 1.4, 300, 0.3),
    'efficientnet-b4': ModelConfig.from_args(1.4, 1.8, 380, 0.4),
    'efficientnet-b5': ModelConfig.from_args(1.6, 2.2, 456, 0.4),
    'efficientnet-b6': ModelConfig.from_args(1.8, 2.6, 528, 0.5),
    'efficientnet-b7': ModelConfig.from_args(2.0, 3.1, 600, 0.5),
    'efficientnet-b8': ModelConfig.from_args(2.2, 3.6, 672, 0.5),
    'efficientnet-l2': ModelConfig.from_args(4.3, 5.3, 800, 0.5),
}

MODEL_CONFIGS_SLIM = {
    # (width, depth, resolution, dropout)
    'efficientnet-b0s': ModelConfigSlim.from_args(1.0, 1.0, 224, 0.2),
    'efficientnet-b1s': ModelConfigSlim.from_args(1.0, 1.1, 240, 0.2),
    'efficientnet-b2s': ModelConfigSlim.from_args(1.1, 1.2, 260, 0.3),
    'efficientnet-b3s': ModelConfigSlim.from_args(1.2, 1.3, 300, 0.3),
    'efficientnet-b4s': ModelConfigSlim.from_args(1.4, 1.8, 380, 0.4),
    'efficientnet-b5s': ModelConfigSlim.from_args(1.6, 2.2, 456, 0.4),
    'efficientnet-b6s': ModelConfigSlim.from_args(1.8, 2.6, 528, 0.5),
    'efficientnet-b7s': ModelConfigSlim.from_args(2.0, 3.1, 600, 0.5),
    'efficientnet-b8s': ModelConfigSlim.from_args(2.2, 3.6, 672, 0.5),
    'efficientnet-l2s': ModelConfigSlim.from_args(4.3, 5.3, 800, 0.5),
}

MODEL_CONFIGS_TINY = {
    # (width, depth, resolution, dropout)
    'efficientnet-b0t': ModelConfigTiny.from_args(1.0, 1.0, 224, 0.2),
    'efficientnet-b1t': ModelConfigTiny.from_args(1.0, 1.1, 240, 0.2),
    'efficientnet-b2t': ModelConfigTiny.from_args(1.1, 1.2, 260, 0.3),
    'efficientnet-b3t': ModelConfigTiny.from_args(1.2, 1.3, 300, 0.3),
    'efficientnet-b4t': ModelConfigTiny.from_args(1.4, 1.8, 380, 0.4),
    'efficientnet-b5t': ModelConfigTiny.from_args(1.6, 2.2, 456, 0.4),
    'efficientnet-b6t': ModelConfigTiny.from_args(1.8, 2.6, 528, 0.5),
    'efficientnet-b7t': ModelConfigTiny.from_args(2.0, 3.1, 600, 0.5),
    'efficientnet-b8t': ModelConfigTiny.from_args(2.2, 3.6, 672, 0.5),
    'efficientnet-l2t': ModelConfigTiny.from_args(4.3, 5.3, 800, 0.5),
}

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # Note: this is a truncated normal distribution
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1 / 3.0,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def round_filters(filters: int, config: ModelConfig) -> int:
    """Round number of filters based on width coefficient."""
    width_coefficient = config.width_coefficient
    min_depth = config.min_depth
    divisor = config.depth_divisor
    orig_filters = filters

    if not width_coefficient:
        return filters

    filters *= width_coefficient
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(
        filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    logging.info('round_filter input=%s output=%s', orig_filters, new_filters)
    return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
    """Round number of repeats based on depth coefficient."""
    return int(math.ceil(depth_coefficient * repeats))


def conv2d_block(inputs: tf.Tensor,
                 conv_filters: Optional[int],
                 config: ModelConfig,
                 kernel_size: Any = (1, 1),
                 strides: Any = (1, 1),
                 use_batch_norm: bool = True,
                 use_bias: bool = False,
                 activation: Any = None,
                 depthwise: bool = False,
                 name: Text = None):
    """A conv2d followed by batch norm and an activation."""
    batch_norm = tf.keras.layers.BatchNormalization
    bn_momentum = config.bn_momentum
    bn_epsilon = config.bn_epsilon
    data_format = tf.keras.backend.image_data_format()
    weight_decay = config.weight_decay

    name = name or ''

    # Collect args based on what kind of conv2d block is desired
    init_kwargs = {
        'kernel_size': kernel_size,
        'strides': strides,
        'use_bias': use_bias,
        'padding': 'same',
        'name': name + '_conv2d',
        'kernel_regularizer': tf.keras.regularizers.l2(weight_decay),
        'bias_regularizer': tf.keras.regularizers.l2(weight_decay),
    }

    if depthwise:
        conv2d = tf.keras.layers.DepthwiseConv2D
        init_kwargs.update({'depthwise_initializer': CONV_KERNEL_INITIALIZER})
    else:
        conv2d = tf.keras.layers.Conv2D
        init_kwargs.update({
            'filters': conv_filters,
            'kernel_initializer': CONV_KERNEL_INITIALIZER
        })

    x = conv2d(**init_kwargs)(inputs)

    if use_batch_norm:
        bn_axis = 1 if data_format == 'channels_first' else -1
        x = batch_norm(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + '_bn')(
                x)

    if activation is not None:
        x = tf.keras.layers.Activation(
            activation, name=name + '_activation')(x)
    return x


def mb_conv_block(inputs: tf.Tensor,
                  block: BlockConfig,
                  config: ModelConfig,
                  prefix: Text = None):
    """Mobile Inverted Residual Bottleneck.
    Args:
      inputs: the Keras input to the block
      block: BlockConfig, arguments to create a Block
      config: ModelConfig, a set of model parameters
      prefix: prefix for naming all layers
    Returns:
      the output of the block
    """
    use_se = config.use_se
    activation = tf.keras.layers.Activation("swish")
    drop_connect_rate = config.drop_connect_rate
    data_format = tf.keras.backend.image_data_format()
    use_depthwise = block.conv_type != 'no_depthwise'
    prefix = prefix or ''

    if block.strides == (2, 1):
        block.fused_conv = True

    filters = block.input_filters * block.expand_ratio

    x = inputs

    if block.fused_conv:
        # If we use fused mbconv, skip expansion and use regular conv.
        x = conv2d_block(
            x,
            filters,
            config,
            kernel_size=block.kernel_size,
            strides=block.strides,
            activation=activation,
            name=prefix + 'fused')
    else:
        if block.expand_ratio != 1:
            # Expansion phase
            kernel_size = (1, 1) if use_depthwise else (3, 3)
            x = conv2d_block(
                x,
                filters,
                config,
                kernel_size=kernel_size,
                activation=activation,
                name=prefix + 'expand')

        # Depthwise Convolution
        if use_depthwise:
            x = conv2d_block(
                x,
                conv_filters=None,
                config=config,
                kernel_size=block.kernel_size,
                strides=block.strides,
                activation=activation,
                depthwise=True,
                name=prefix + 'depthwise')

    # Squeeze and Excitation phase
    if use_se:
        assert block.se_ratio is not None
        assert 0 < block.se_ratio <= 1
        num_reduced_filters = max(1, int(block.input_filters * block.se_ratio))

        if data_format == 'channels_first':
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)

        se = tf.keras.layers.GlobalAveragePooling2D(
            name=prefix + 'se_squeeze')(x)
        se = tf.keras.layers.Reshape(se_shape, name=prefix + 'se_reshape')(se)

        se = conv2d_block(
            se,
            num_reduced_filters,
            config,
            use_bias=True,
            use_batch_norm=False,
            activation=activation,
            name=prefix + 'se_reduce')
        se = conv2d_block(
            se,
            filters,
            config,
            use_bias=True,
            use_batch_norm=False,
            activation='sigmoid',
            name=prefix + 'se_expand')
        x = tf.keras.layers.multiply([x, se], name=prefix + 'se_excite')

    # Output phase
    x = conv2d_block(
        x, block.output_filters, config, activation=None, name=prefix + 'project')

    # Add identity so that quantization-aware training can insert quantization
    # ops correctly.
    x = tf.keras.layers.Activation("linear", name=prefix + 'id')(x)

    if (block.id_skip and all(s == 1 for s in block.strides) and
            block.input_filters == block.output_filters):
        if drop_connect_rate and drop_connect_rate > 0:
            # Apply dropconnect
            # The only difference between dropout and dropconnect in TF is scaling by
            # drop_connect_rate during training. See:
            # https://github.com/keras-team/keras/pull/9898#issuecomment-380577612
            x = tf.keras.layers.Dropout(
                drop_connect_rate, noise_shape=(None, 1, 1, 1), name=prefix + 'drop')(
                    x)

        x = tf.keras.layers.add([x, inputs], name=prefix + 'add')

    return x


def efficientnet(image_input: tf.keras.layers.Input, config: ModelConfig):
    depth_coefficient = config.depth_coefficient
    blocks = config.blocks
    stem_base_filters = config.stem_base_filters
    top_base_filters = config.top_base_filters
    activation = tf.keras.layers.Activation("swish")
    drop_connect_rate = config.drop_connect_rate

    x = image_input

    # Build stem
    x = conv2d_block(
        x,
        round_filters(stem_base_filters, config),
        config,
        kernel_size=[3, 3],
        strides=[2, 2],
        activation=activation,
        name='stem')

    # Build blocks
    num_blocks_total = sum(
        round_repeats(block.num_repeat, depth_coefficient) for block in blocks)
    block_num = 0

    for stack_idx, block in enumerate(blocks):
        assert block.num_repeat > 0
        # Update block input and output filters based on depth multiplier
        block = block.replace(
            input_filters=round_filters(block.input_filters, config),
            output_filters=round_filters(block.output_filters, config),
            num_repeat=round_repeats(block.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        config = config.replace(drop_connect_rate=drop_rate)
        block_prefix = 'stack_{}/block_0/'.format(stack_idx)
        x = mb_conv_block(x, block, config, block_prefix)
        block_num += 1
        if block.num_repeat > 1:
            block = block.replace(
                input_filters=block.output_filters, strides=[1, 1])

            for block_idx in range(block.num_repeat - 1):
                drop_rate = drop_connect_rate * \
                    float(block_num) / num_blocks_total
                config = config.replace(drop_connect_rate=drop_rate)
                block_prefix = 'stack_{}/block_{}/'.format(
                    stack_idx, block_idx + 1)
                x = mb_conv_block(x, block, config, prefix=block_prefix)
                block_num += 1

    # Build top
    # x = conv2d_block(
    #     x,
    #     round_filters(top_base_filters, config),
    #     config,
    #     activation=activation,
    #     name='top')

    x = conv2d_block(
        x,
        top_base_filters,
        config,
        activation=activation,
        name='top')
    x = tf.squeeze(x, axis=1)
    return x

def EfficientNet(image_input, model_name):
    if 's' in model_name:
        model_configs = MODEL_CONFIGS_SLIM[model_name]
    elif 't' in model_name:
        model_configs = MODEL_CONFIGS_TINY[model_name]
    else:
        model_configs = MODEL_CONFIGS[model_name]

    return efficientnet(image_input, model_configs)

def EfficientNetB0(image_input):
    model_configs = MODEL_CONFIGS['efficientnet-b0']
    return efficientnet(image_input, model_configs)

def EfficientNetB1(image_input):
    model_configs = MODEL_CONFIGS['efficientnet-b1']
    return efficientnet(image_input, model_configs)
    
def EfficientNetB2(image_input):
    model_configs = MODEL_CONFIGS['efficientnet-b2']
    return efficientnet(image_input, model_configs)

def EfficientNetB3(image_input):
    model_configs = MODEL_CONFIGS['efficientnet-b3']
    return efficientnet(image_input, model_configs)

def EfficientNetB4(image_input):
    model_configs = MODEL_CONFIGS['efficientnet-b4']
    return efficientnet(image_input, model_configs)

def EfficientNetB0s(image_input):
    model_configs = MODEL_CONFIGS_SLIM['efficientnet-b0s']
    return efficientnet(image_input, model_configs)

def EfficientNetB1s(image_input):
    model_configs = MODEL_CONFIGS_SLIM['efficientnet-b1s']
    return efficientnet(image_input, model_configs)
    
def EfficientNetB2s(image_input):
    model_configs = MODEL_CONFIGS_SLIM['efficientnet-b2s']
    return efficientnet(image_input, model_configs)

def EfficientNetB3s(image_input):
    model_configs = MODEL_CONFIGS_SLIM['efficientnet-b3s']
    return efficientnet(image_input, model_configs)

def EfficientNetB4s(image_input):
    model_configs = MODEL_CONFIGS_SLIM['efficientnet-b4s']
    return efficientnet(image_input, model_configs)

def EfficientNetB0t(image_input):
    model_configs = MODEL_CONFIGS_TINY['efficientnet-b0t']
    return efficientnet(image_input, model_configs)

def EfficientNetB1t(image_input):
    model_configs = MODEL_CONFIGS_TINY['efficientnet-b1t']
    return efficientnet(image_input, model_configs)

def EfficientNetB2t(image_input):
    model_configs = MODEL_CONFIGS_TINY['efficientnet-b2t']
    return efficientnet(image_input, model_configs)

def EfficientNetB3t(image_input):
    model_configs = MODEL_CONFIGS_TINY['efficientnet-b3t']
    return efficientnet(image_input, model_configs)

def EfficientNetB4t(image_input):
    model_configs = MODEL_CONFIGS_TINY['efficientnet-b4t']
    return efficientnet(image_input, model_configs)

if __name__ == "__main__":
    pass
