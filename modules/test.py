from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import argparse
import os
import numpy as np
from modules.feature_extractor.resnet import resnet18_slim, resnet50_slim, resnet50, resnet18
from modules.sequence_modeling.bilstm import Attention_BiLSTM, BiLSTM
from modules.model_head.attention import Attention
from modules.data_helper.data_generator import Dataset
from modules.data_helper.data_utils import AttnLabelConverter, NormalizePAD
from modules.lnr_factory import CosineDecayWithWarmup

from modules.model import get_crnn_attention_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_set', type=str, required=True, help='path to train set tfrecords file')
    # parser.add_argument('--valid_set', type=str, required=True, help='path to valid set tfrecords file')
    parser.add_argument('--test_set', type=str, required=True, help='path to test set tfrecords file')
    parser.add_argument('--augment_level', type=int, default=5, help='level of RandAugment')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=280, help='the width of the input image')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--max_len', type=int, default=12, help='maximum-label-length')
    parser.add_argument('--character', type=str, default='0123456789', help='character label')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')

    parser.add_argument('--log_dir', type=str, required=True, help='path to save checkpoint')
    opt = parser.parse_args()

    opt.num_channels = 1
    if opt.rgb:
        opt.num_channels = 3

    text_converter = AttnLabelConverter(opt)
    image_converter = NormalizePAD(opt, is_training=False)
    opt.num_classes = len(text_converter.character)

    print(opt)
    image_tensor = tf.keras.layers.Input(shape=[opt.imgH, opt.imgW, opt.num_channels])
    text_tensor = tf.keras.layers.Input(shape=[opt.max_len + 1], dtype=tf.int32)
    logits = get_crnn_attention_model(image_tensor, text_tensor, False, opt)

    model = tf.keras.Model(inputs=[image_tensor, text_tensor], outputs=logits)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(opt.log_dir), max_to_keep=3)
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if not latest_checkpoint:
        logging.info('No checkpoint found. Exiting.....')
        exit(0)
    else:
        logging.info('Checkpoint file %s found and restoring from ''checkpoint', latest_checkpoint)
        checkpoint.restore(latest_checkpoint)

    idnum = Dataset(opt.test_set, epochs=1, text_converter=text_converter, image_converter=image_converter)
    num_corrects = 0
    num_totals = 0
        
    text_for_pred = tf.zeros([32, opt.max_len + 1], dtype=tf.int32)
    for image, label, length in idnum.gen(32):
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)

        logits = model((image, text_for_pred), training=False)
        logits = tf.argmax(logits, axis=2)

        logits = logits.numpy()
        labels = label[:, 1:].numpy()

        for i in range(logits.shape[0]):
            num_totals += 1
            if np.array_equal(logits[i], labels[i]):
                num_corrects += 1

        print('Current accuracy: ', num_corrects * 1.0 / num_totals)
