from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from typing import Any, Dict, Optional, Text, Tuple

from absl import logging
from dataclasses import dataclass
from numpy.core.numeric import False_
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
    """Default Config for EfficientnetV2-Base."""
    width_coefficient: float = 1.0
    depth_coefficient: float = 1.0
    resolution: int = 224
    dropout_rate: float = 0.2
    blocks: Tuple[BlockConfig, ...] = None
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
    model_name: str = 'efficientnetv2'
    rescale_input: bool = True
    data_format: str = 'channels_last'
    dtype: str = 'float32'

base_blocks: Tuple[BlockConfig, ...] = (
        # (input_filters, output_filters, kernel_size, num_repeat,
        #  expand_ratio, strides, se_ratio, id_skip, fused_conv)
        # pylint: disable=bad-whitespace
        BlockConfig.from_args(32, 16, 3, 1, 1, (1, 1), 0, True, True),
        BlockConfig.from_args(16, 32, 3, 2, 4, (2, 2), 0, True, True),
        BlockConfig.from_args(32, 48, 3, 2, 4, (2, 1), 0, True, True),
        BlockConfig.from_args(48, 96, 3, 3, 4, (2, 1), 0.25, True, False),
        BlockConfig.from_args(96, 112, 3, 5, 6, (1, 1), 0.25, True, False),
        BlockConfig.from_args(112, 192, 3, 8, 6, (2, 1), 0.25, True, False),
        # pylint: enable=bad-whitespace
    )

s_blocks: Tuple[BlockConfig, ...] = (
        # (input_filters, output_filters, kernel_size, num_repeat,
        #  expand_ratio, strides, se_ratio, id_skip, fused_conv)
        # pylint: disable=bad-whitespace
        BlockConfig.from_args(24, 24, 3, 2, 1, (1, 1), 0, True, True),
        BlockConfig.from_args(24, 48, 3, 4, 4, (2, 2), 0, True, True),
        BlockConfig.from_args(48, 64, 3, 4, 4, (2, 1), 0, True, True),
        BlockConfig.from_args(64, 128, 3, 6, 4, (2, 1), 0.25, True, False),
        BlockConfig.from_args(128, 160, 3, 9, 6, (1, 1), 0.25, True, False),
        BlockConfig.from_args(160, 256, 3, 15, 6, (2, 1), 0.25, True, False),
        # pylint: enable=bad-whitespace
    )

m_blocks: Tuple[BlockConfig, ...] = (
        # (input_filters, output_filters, kernel_size, num_repeat,
        #  expand_ratio, strides, se_ratio, id_skip, fused_conv)
        # pylint: disable=bad-whitespace
        BlockConfig.from_args(24, 24, 3, 3, 1, (1, 1), 0, True, True),
        BlockConfig.from_args(24, 48, 3, 5, 4, (2, 2), 0, True, True),
        BlockConfig.from_args(48, 80, 3, 5, 4, (2, 1), 0, True, True),
        BlockConfig.from_args(80, 160, 3, 7, 4, (2, 1), 0.25, True, False),
        BlockConfig.from_args(160, 176, 3, 14, 6, (1, 1), 0.25, True, False),
        BlockConfig.from_args(176, 304, 3, 18, 6, (2, 1), 0.25, True, False),
        BlockConfig.from_args(304, 512, 3, 5, 6, (1, 1), 0.25, True, False),
        # pylint: enable=bad-whitespace
    )

slim_blocks: Tuple[BlockConfig, ...] = (
        # (input_filters, output_filters, kernel_size, num_repeat,
        #  expand_ratio, strides, se_ratio, id_skip, fused_conv)
        # pylint: disable=bad-whitespace
        BlockConfig.from_args(24, 24, 3, 1, 1, (1, 1), 0, True, True),
        BlockConfig.from_args(24, 48, 3, 2, 4, (2, 2), 0, True, True),
        BlockConfig.from_args(48, 64, 3, 2, 4, (2, 1), 0, True, True),
        BlockConfig.from_args(64, 128, 3, 3, 4, (2, 1), 0.25, True, False),
        BlockConfig.from_args(128, 160, 3, 4, 4, (1, 1), 0.25, True, False),
        BlockConfig.from_args(160, 256, 3, 7, 4, (2, 1), 0.25, True, False),
        # pylint: enable=bad-whitespace
    )

tiny_blocks: Tuple[BlockConfig, ...] = (
        # (input_filters, output_filters, kernel_size, num_repeat,
        #  expand_ratio, strides, se_ratio, id_skip, fused_conv)
        # pylint: disable=bad-whitespace
        BlockConfig.from_args(24, 24, 3, 1, 1, (1, 1), 0, True, True),
        BlockConfig.from_args(24, 48, 3, 2, 4, (2, 2), 0, True, True),
        BlockConfig.from_args(48, 64, 3, 2, 2, (2, 1), 0, True, True),
        BlockConfig.from_args(64, 128, 3, 3, 2, (2, 1), 0.25, True, False),
        BlockConfig.from_args(128, 160, 3, 3, 4, (1, 1), 0.25, True, False),
        BlockConfig.from_args(160, 256, 3, 4, 2, (2, 1), 0.25, True, False),
        # pylint: enable=bad-whitespace
    )

MODEL_CONFIGS = {
    # (width, depth, resolution, dropout, blocks, stem, head)
    'efficientnetv2-s': ModelConfig.from_args(1.0, 1.0, 300, 0.2, s_blocks, 24, 1280),
    'efficientnetv2-m': ModelConfig.from_args(1.0, 1.0, 384, 0.3, m_blocks, 24, 1280),
    'efficientnetv2-b0': ModelConfig.from_args(1.0, 1.0, 192, 0.2, base_blocks, 32, 1280),
    'efficientnetv2-b1': ModelConfig.from_args(1.0, 1.1, 192, 0.2, base_blocks, 32, 1280),
    'efficientnetv2-b2': ModelConfig.from_args(1.1, 1.2, 208, 0.3, base_blocks, 32, 1280),
    'efficientnetv2-b3': ModelConfig.from_args(1.2, 1.4, 240, 0.3, base_blocks, 32, 1280),
    'efficientnetv2-slim': ModelConfig.from_args(1.0, 1.0, 300, 0.2, slim_blocks, 24, 512),
    'efficientnetv2-tiny': ModelConfig.from_args(1.0, 1.0, 300, 0.2, tiny_blocks, 24, 256),
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
    use_se = config.use_se and (block.se_ratio > 0)
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
    x = conv2d_block(
        x,
        round_filters(top_base_filters, config),
        config,
        activation=activation,
        name='top')

    # x = conv2d_block(
    #     x,
    #     top_base_filters,
    #     config,
    #     activation=activation,
    #     name='top')
    x = tf.squeeze(x, axis=1)
    return x

def EfficientNetV2B0(image_input):
    model_configs = MODEL_CONFIGS['efficientnetv2-b0']
    return efficientnet(image_input, model_configs)

def EfficientNetV2B1(image_input):
    model_configs = MODEL_CONFIGS['efficientnetv2-b1']
    return efficientnet(image_input, model_configs)
    
def EfficientNetV2B2(image_input):
    model_configs = MODEL_CONFIGS['efficientnetv2-b2']
    return efficientnet(image_input, model_configs)

def EfficientNetV2B3(image_input):
    model_configs = MODEL_CONFIGS['efficientnetv2-b3']
    return efficientnet(image_input, model_configs)

def EfficientNetV2S(image_input):
    model_configs = MODEL_CONFIGS['efficientnetv2-s']
    return efficientnet(image_input, model_configs)

def EfficientNetV2M(image_input):
    model_configs = MODEL_CONFIGS['efficientnetv2-m']
    return efficientnet(image_input, model_configs)

def EfficientNetV2Slim(image_input):
    model_configs = MODEL_CONFIGS['efficientnetv2-slim']
    return efficientnet(image_input, model_configs)

def EfficientNetV2Tiny(image_input):
    model_configs = MODEL_CONFIGS['efficientnetv2-tiny']
    return efficientnet(image_input, model_configs)

if __name__ == "__main__":
    pass
