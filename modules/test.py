from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import argparse
import os
import numpy as np

from modules.data_helper.data_generator import Dataset
from modules.data_helper.data_utils import AttnLabelConverter, NormalizePAD
from modules.model import get_crnn_attention_model
from modules.parameters.base_config import Config

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
# tf.config.experimental.enable_tensor_float_32_execution(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='path to config file')
    args = parser.parse_args()

    opt = Config.from_yaml(args.config_file)

    text_converter = AttnLabelConverter(opt)
    image_converter = NormalizePAD(opt, is_training=True)
    opt.model_params.num_classes = len(text_converter.character)

    image_tensor = tf.keras.layers.Input(shape=[opt.model_params.imgH, opt.model_params.imgW, opt.model_params.num_channels])
    text_tensor = tf.keras.layers.Input(shape=[opt.model_params.max_len + 1], dtype=tf.int32)
    logits = get_crnn_attention_model(image_tensor, text_tensor, is_train=False, opt=opt)
    model = tf.keras.Model(inputs=[image_tensor, text_tensor], outputs=logits)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(opt.training_params.log_dir), max_to_keep=3)
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if not latest_checkpoint:
        logging.info('No checkpoint found. Exiting.....')
        exit(0)
    else:
        logging.info('Checkpoint file %s found and restoring from ''checkpoint', latest_checkpoint)
        checkpoint.restore(latest_checkpoint)

    idnum = Dataset(opt.dataset.test_set, epochs=1, text_converter=text_converter, image_converter=image_converter)
    num_corrects = 0
    num_totals = 0
        
    text_for_pred = tf.zeros([opt.training_params.batch_size, opt.model_params.max_len + 1], dtype=tf.int32)
    for image, label, length in idnum.gen(opt.training_params.batch_size):
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)

        logits = model((image, text_for_pred), training=False)
        logits = tf.argmax(logits, axis=2)

        logits = logits.numpy()
        labels = label[:, 1:].numpy()
        num_totals += len(logits)
        num_corrects += np.sum(np.sum(logits - labels, axis=1) == 0)

        print('Current accuracy: ', num_corrects * 1.0 / num_totals)
