from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse

from feature_extractor.resnet import resnet18_slim, resnet50_slim, resnet50, resnet18
from sequence_modeling.bilstm import Attention_BiLSTM, BiLSTM
from model_head.attention import Attention
from data_helper.data_generator import Dataset
from data_helper.data_utils import AttnLabelConverter, NormalizePAD

def get_crnn_attention_model(image_tensor, text_tensor, is_train, opt=None):
    image_features = resnet50_slim(image_tensor)
    seq_features = Attention_BiLSTM(image_features)
    logits = Attention(opt.hidden_size, opt.num_classes)(seq_features, text_tensor, is_train=is_train, batch_max_length=opt.max_len)

    return logits

if __name__ == "__main__":
    pass

