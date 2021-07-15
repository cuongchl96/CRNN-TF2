from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import L

import tensorflow as tf
import argparse
import logging
import os
import numpy as np
import time

from tensorflow.python.training.checkpoint_management import latest_checkpoint

from modules.data_helper.data_generator import Dataset
from modules.data_helper.data_utils import AttnLabelConverter, NormalizePAD, CTCLabelConverter
from modules.lnr_factory import CosineDecayWithWarmup
from modules.model import get_crnn_attention_model, get_crnn_transformer_ctc_model
from modules.parameters.base_config import Config

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
tf.config.experimental.enable_tensor_float_32_execution(False)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 6)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def validator(model, generator, opt):
    num_corrects = 0
    num_totals = 0
    total_loss = 0
    num_steps = 0
    for image, label, length in generator.gen(opt.training_params.batch_size):
        num_steps += 1
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)
        length = tf.expand_dims(tf.convert_to_tensor(length), 1)

        logits = model(image, training=False)
        logits_length = tf.shape(logits)[1] * tf.ones((tf.shape(logits)[0], 1), tf.int32)
        loss = tf.keras.backend.ctc_batch_cost(y_true=label, y_pred=logits, label_length=length, input_length=logits_length)
        loss = tf.math.reduce_sum(loss)

        logits_length = tf.squeeze(logits_length, axis=1)
        output = tf.keras.backend.ctc_decode(logits, input_length=logits_length, greedy=True)[0][0].numpy()

        length = length.numpy()
        label = label.numpy()
        num_totals += len(length)
        for i in range(len(length)):
            true_char = 0
            for j in range(length[i][0]):
                if output[i][j] == label[i][j]:
                    true_char += 1
            if true_char == length[i]:
                num_corrects += 1
        total_loss += loss

    return num_corrects * 100.0 / num_totals, total_loss / num_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        required=True, help='path to config file')
    args = parser.parse_args()

    opt = Config.from_yaml(args.config_file)

    if not os.path.isdir(opt.training_params.log_dir):
        os.makedirs(opt.training_params.log_dir)

    text_converter = CTCLabelConverter(opt)
    image_converter = NormalizePAD(opt, is_training=True)
    opt.model_params.num_classes = len(text_converter.character)

    train_generator = Dataset(opt.dataset.train_set, epochs=opt.training_params.num_epochs,
                              text_converter=text_converter, image_converter=image_converter)
    total_examples = train_generator.ds_len
    num_steps = int(total_examples / opt.training_params.batch_size) + 1

    image_tensor = tf.keras.layers.Input(
        shape=[opt.model_params.imgH, opt.model_params.imgW, opt.model_params.num_channels])
    
    logits = get_crnn_transformer_ctc_model(image_tensor, is_train=True, opt=opt)

    model = tf.keras.Model(inputs=image_tensor, outputs=logits)
    logging.info('Total number of model parameters: ' +
                 str(model.count_params()))

    learning_rate = CosineDecayWithWarmup(base_lr=opt.training_params.initial_lnr,
                                          total_steps=num_steps, warmup_steps=opt.training_params.num_warmup_steps)
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, os.path.join(opt.training_params.log_dir), max_to_keep=3)
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if not latest_checkpoint:
        logging.info('No checkpoint found. Create model with fresh parameters')
    else:
        logging.info(
            'Checkpoint file %s found and restoring from ''checkpoint', latest_checkpoint)
        checkpoint.restore(latest_checkpoint)

    summary_writer = tf.summary.create_file_writer(opt.training_params.log_dir)

    num_corrects = 0
    num_totals = 0
    training_loss = 0

    for image, label, length in train_generator.gen(opt.training_params.batch_size):
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)
        length = tf.expand_dims(tf.convert_to_tensor(length), 1)
        with tf.GradientTape() as tape:
            logits = model(image, training=True)
            logits_length = tf.shape(logits)[1] * tf.ones((tf.shape(logits)[0], 1), tf.int32)

            loss = tf.keras.backend.ctc_batch_cost(y_true=label, y_pred=logits, label_length=length, input_length=logits_length)
            loss = tf.math.reduce_mean(loss)

            training_loss += loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            logits_length = tf.squeeze(logits_length, axis=1)
            output = tf.keras.backend.ctc_decode(logits, input_length=logits_length, greedy=True)[0][0].numpy()

            length = length.numpy()
            label = label.numpy()
            num_totals += len(length)
            for i in range(len(length)):
                true_char = 0
                for j in range(length[i][0]):
                    if output[i][j] == label[i][j]:
                        true_char += 1
                if true_char == length[i]:
                    num_corrects += 1

        cur_step = optimizer.iterations.numpy()
        if cur_step % opt.training_params.log_interval == 0:
            logging.info("Step %d\tLearning rate: %.3f\tLoss: %.3f" % (
                cur_step, learning_rate.learning_rate.numpy(), loss.numpy()))

        if cur_step % opt.training_params.save_interval == 0:
            save_path = checkpoint_manager.save()
            logging.info("Saved checkpoint for step {}: {}".format(cur_step, save_path))

            training_loss = training_loss / opt.training_params.save_interval
            training_accuracy = num_corrects * 100.0 / num_totals
            logging.info("Training accuracy: %.3f\tTraining loss: %.3f" % (training_accuracy, training_loss))

            valid_generator = Dataset(opt.dataset.valid_set, epochs=1, text_converter=text_converter,
                                      image_converter=NormalizePAD(opt, is_training=False))
            valid_accuracy, valid_loss = validator(model, valid_generator, opt)
            logging.info("Valid accuracy: %.3f\tValid loss: %.3f" % (valid_accuracy, valid_loss))

            with summary_writer.as_default():
                tf.summary.scalar('train_loss', training_loss, step=cur_step)
                tf.summary.scalar('train_accuracy', training_accuracy / 100, step=cur_step)
                tf.summary.scalar('valid_loss', valid_loss, step=cur_step)
                tf.summary.scalar('valid_accuracy', valid_accuracy / 100, step=cur_step)

            num_corrects = 0
            num_totals = 0
            training_loss = 0
