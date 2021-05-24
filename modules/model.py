from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import logging

from modules.feature_extractor.resnet import resnet18_slim, resnet50_slim, resnet50, resnet18
from modules.feature_extractor.efficientnet_v1 import EfficientNet
from modules.sequence_modeling.bilstm import Attention_BiLSTM, BiLSTM, Attention_BiLSTM_v2, BiLSTM_v2
from modules.model_head.attention import Attention
from modules.model_head.ctc import CTC

backbone_factory = {
    'resnet18_slim': resnet18_slim,
    'resnet18': resnet18,
    'resnet50_slim': resnet50_slim,
    'resnet50': resnet50,
    'efficientnet': EfficientNet
}

seq_modeling_factory = {
    'bilstm': BiLSTM,
    'abilstm': Attention_BiLSTM,
    'bilstmv2': BiLSTM_v2,
    'abilstmv2': Attention_BiLSTM_v2,
}

model_head_factory = {
    'ctc': CTC,
    'attention': Attention
}

def get_backbone(opt):
    name = opt.model_build.backbone.lower()

    if name not in backbone_factory.keys():
        logging.DEBUG('Unsupported backbone %s', name)
    return backbone_factory[name]

def get_sequence_modeling(opt):
    name = opt.model_build.sequence_modeling.lower()

    if name not in seq_modeling_factory.keys():
        logging.DEBUG('Unsupported sequence modeling %s', name)
    return seq_modeling_factory[name]

def get_model_head(opt):
    name = opt.model_build.model_head.lower()

    if name not in model_head_factory.keys():
        logging.DEBUG('Unsupported model head %s', name)
    return model_head_factory[name]

def get_crnn_attention_model(image_tensor, text_tensor, is_train, opt=None):
    image_features = resnet18_slim(image_tensor)
    # image_features = EfficientNet(image_tensor, 'efficientnet-b2s')
    seq_features = Attention_BiLSTM(image_features, hidden_units=opt.hidden_size)
    logits = Attention(opt.hidden_size, opt.num_classes)(seq_features, text_tensor, is_train=is_train, batch_max_length=opt.max_len)

    return logits

if __name__ == "__main__":
    image_input = tf.keras.layers.Input(shape=[32, 280, 3])
    output = EfficientNet(image_input, 'efficientnet-b0')
    print(output.get_shape)
    model = tf.keras.Model(inputs=image_input, outputs=output, name='effnet0')
    print('Effnet-b0 parameters: ', model.count_params())

    image_input = tf.keras.layers.Input(shape=[32, 280, 3])
    output = EfficientNet(image_input, 'efficientnet-b0s')
    print(output.get_shape)
    model = tf.keras.Model(inputs=image_input, outputs=output, name='effnet0s')
    print('Effnet-b0s parameters: ', model.count_params())

    image_input = tf.keras.layers.Input(shape=[32, 280, 3])
    output = EfficientNet(image_input, 'efficientnet-b2')
    print(output.get_shape)
    model = tf.keras.Model(inputs=image_input, outputs=output, name='effnet2')
    print('Effnet-b2 parameters: ', model.count_params())

    image_input = tf.keras.layers.Input(shape=[32, 280, 3])
    output = EfficientNet(image_input, 'efficientnet-b2s')
    print(output.get_shape)
    model = tf.keras.Model(inputs=image_input, outputs=output, name='effnet2s')
    print('Effnet-b2s parameters: ', model.count_params())


