from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import logging
import os
import numpy as np
import time

from tensorflow.python.training.checkpoint_management import latest_checkpoint

from data_helper.data_generator import Dataset
from data_helper.data_utils import AttnLabelConverter, NormalizePAD
from lnr_factory import CosineDecayWithWarmup
from model import get_crnn_attention_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

def validator(model, generator, opt):
    num_corrects = 0
    num_totals = 0
    for image, label, length in generator.gen(opt.batch_size):
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)

        logits = model((image, label[:, :-1]), training=False)
        logits = tf.argmax(logits, axis=2)

        logits = logits.numpy()
        labels = label[:, 1:].numpy()
        num_totals += len(logits)
        num_corrects += np.sum(np.sum(logits - labels, axis=1) == 0)
    return num_corrects * 100.0 / num_totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', type=str, required=True, help='path to train set tfrecords file')
    parser.add_argument('--valid_set', type=str, required=True, help='path to valid set tfrecords file')

    parser.add_argument('--augment_level', type=int, default=5, help='level of RandAugment')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_warmup_steps', type=int, default=100, help='number of warmup steps')
    parser.add_argument('--initial_lnr', type=float, default=1.0, help='number of warmup steps')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=280, help='the width of the input image')
    parser.add_argument('--max_len', type=int, default=12, help='maximum-label-length')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--character', type=str, default='0123456789', help='character label')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')


    parser.add_argument('--log_dir', type=str, required=True, help='path to save checkpoint')
    parser.add_argument('--log_interval', type=int, default=10, help='number of steps to log')
    parser.add_argument('--save_interval', type=int, default=10, help='number of steps to save model')
    opt = parser.parse_args()

    opt.num_channels = 1
    if opt.rgb:
        opt.num_channels = 3

    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)

    text_converter = AttnLabelConverter(opt)
    image_converter = NormalizePAD(opt, is_training=True)
    opt.num_classes = len(text_converter.character)

    train_generator = Dataset(opt.train_set, epochs=opt.num_epochs, text_converter=text_converter, image_converter=image_converter)
    num_steps = int(300000 * opt.num_epochs / opt.batch_size) + 1

    image_tensor = tf.keras.layers.Input(shape=[opt.imgH, opt.imgW, opt.num_channels])
    text_tensor = tf.keras.layers.Input(shape=[opt.max_len + 1], dtype=tf.int32)
    logits = get_crnn_attention_model(image_tensor, text_tensor, is_train=True, opt=opt)

    model = tf.keras.Model(inputs=[image_tensor, text_tensor], outputs=logits)
    logging.info('Total number of model parameters: ' + str(model.count_params()))

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    learning_rate = CosineDecayWithWarmup(base_lr=opt.initial_lnr, total_steps=num_steps, warmup_steps=opt.num_warmup_steps)
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(opt.log_dir), max_to_keep=3)
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if not latest_checkpoint:
        logging.info('No checkpoint found. Create model with fresh parameters')
    else:
        logging.info('Checkpoint file %s found and restoring from ''checkpoint', latest_checkpoint)
        checkpoint.restore(latest_checkpoint)

    train_generator = Dataset(opt.train_set, epochs=opt.num_epochs, text_converter=text_converter, image_converter=image_converter)

    num_corrects = 0
    num_totals = 0
    for image, label, length in train_generator.gen(opt.batch_size):
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)

        with tf.GradientTape() as tape:
            logits = model((image, label[:, :-1]), training=True)
            loss = loss_func(y_true=label[:, 1:], y_pred=logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            logits = tf.argmax(logits, axis=2)
            logits = logits.numpy()
            labels = label[:, 1:].numpy()
            num_totals += len(logits)
            num_corrects += np.sum(np.sum(logits - labels, axis=1) == 0)

        cur_step = optimizer.iterations.numpy()
        if cur_step % opt.log_interval == 0:
            logging.info("Step %d\tLearning rate: %.3f\tLoss: %.3f"%(cur_step, learning_rate.learning_rate.numpy(), loss.numpy()))

        if cur_step % opt.save_interval == 0:
            save_path = checkpoint_manager.save()
            logging.info("Saved checkpoint for step {}: {}".format(cur_step, save_path))
            logging.info("Training accuracy: %.3f" % (num_corrects * 100.0 / num_totals))
            valid_generator = Dataset(opt.valid_set, epochs=1, text_converter=text_converter, image_converter=NormalizePAD(opt, is_training=False))
            logging.info("Valid accuracy: %.3f" % validator(model, valid_generator, opt))
            num_corrects = 0
            num_totals = 0

        

