from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import argparse
import os
import numpy as np
import cv2

from modules.data_helper.data_generator import Dataset
from modules.data_helper.data_utils import AttnLabelConverter, NormalizePAD
from modules.model import get_crnn_attention_model
from modules.parameters.base_config import Config

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
# tf.config.experimental.enable_tensor_float_32_execution(False)

def evaluate(infer_fn, opt):
    generator = Dataset(opt.dataset.test_set, epochs=1, text_converter=text_converter, image_converter=image_converter)
    num_corrects = 0
    num_totals = 0
        
    text_for_pred = tf.zeros([opt.training_params.batch_size, opt.model_params.max_len + 1], dtype=tf.int32)
    for image, label, length in generator.gen(opt.training_params.batch_size):
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)

        logits = infer_fn(image, text_for_pred)['tf.compat.v1.transpose']
        logits = tf.argmax(logits, axis=2)

        logits = logits.numpy()
        labels = label[:, 1:].numpy()
        num_totals += len(logits)
        num_corrects += np.sum(np.sum(logits - labels, axis=1) == 0)

        print('Current accuracy: ', num_corrects * 1.0 / num_totals)

def predict(infer_fn, image_path, opt):
    image = cv2.imread(image_path)
    processed_image = tf.convert_to_tensor([image_converter(image)])
    text_for_pred = tf.zeros([1, opt.model_params.max_len + 1], dtype=tf.int32)

    logits = infer_fn(processed_image, text_for_pred)['tf.compat.v1.transpose']
    logits = tf.argmax(logits, axis=2)

    return text_converter.decode(logits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='path to config file')
    parser.add_argument('--export_path', type=str, required=True, help='path to savedmodel')
    args = parser.parse_args()

    opt = Config.from_yaml(args.config_file)

    text_converter = AttnLabelConverter(opt)
    image_converter = NormalizePAD(opt, is_training=False)

    exported_path = args.export_path

    loaded = tf.saved_model.load(exported_path)
    infer = loaded.signatures["serving_default"]
    infer._num_positional_args = 2

    # evaluate(infer, opt)
    predict(infer, '/media/cuongvc/UBUNTU/home/cuongvc/cuongltb/aocr/datasets/idcards/idnum/image/2020-02-03-15-28-38_211442911_VO-THI-CHANH_True.jpg', opt)