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

from modules.data_helper.data_generator import Dataset
from modules.data_helper.data_utils import AttnLabelConverter, NormalizePAD
from modules.lnr_factory import CosineDecayWithWarmup
from modules.model import get_crnn_attention_model
from modules.parameters.base_config import Config

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
tf.config.experimental.enable_tensor_float_32_execution(False)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2.5)])
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

        logits = model((image, label[:, :-1]), training=False)
        loss = loss_func(y_true=label[:, 1:], y_pred=logits)
        logits = tf.argmax(logits, axis=2)

        logits = logits.numpy()
        labels = label[:, 1:].numpy()
        num_totals += len(logits)
        num_corrects += np.sum(np.sum(logits - labels, axis=1) == 0)
        total_loss += loss

        if num_steps == 1:
            prediction = text_converter.decode(tf.convert_to_tensor(logits))
            true_label = text_converter.decode(tf.convert_to_tensor(labels))
            logging.info('Samples prediction.............')
            for i in range(5):
                logging.info('Prediction: %s' %(prediction[i]))
                logging.info('Label: %s\n' %(true_label[i]))


    return num_corrects * 100.0 / num_totals, total_loss / num_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        required=True, help='path to config file')
    args = parser.parse_args()

    opt = Config.from_yaml(args.config_file)

    if not os.path.isdir(opt.training_params.log_dir):
        os.makedirs(opt.training_params.log_dir)

    text_converter = AttnLabelConverter(opt)
    image_converter = NormalizePAD(opt, is_training=True)
    opt.model_params.num_classes = len(text_converter.character)

    train_generator = Dataset(opt.dataset.train_set, epochs=opt.training_params.num_epochs,
                              text_converter=text_converter, image_converter=image_converter)
    total_examples = train_generator.ds_len
    num_steps = int(total_examples / opt.training_params.batch_size) + 1

    image_tensor = tf.keras.layers.Input(
        shape=[opt.model_params.imgH, opt.model_params.imgW, opt.model_params.num_channels])
    text_tensor = tf.keras.layers.Input(
        shape=[opt.model_params.max_len + 1], dtype=tf.int32)
    logits = get_crnn_attention_model(
        image_tensor, text_tensor, is_train=True, opt=opt)

    model = tf.keras.Model(inputs=[image_tensor, text_tensor], outputs=logits)
    logging.info('Total number of model parameters: ' +
                 str(model.count_params()))

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    learning_rate = CosineDecayWithWarmup(base_lr=opt.training_params.initial_lnr,
                                          total_steps=num_steps, warmup_steps=opt.training_params.num_warmup_steps)
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, os.path.join(opt.training_params.log_dir), max_to_keep=3)
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if not latest_checkpoint:
        logging.info('No checkpoint found. Finding pretrained model........')
        if opt.training_params.pretrain_ckpt is not None:
            logging.info('Found pretrained model %s', opt.training_params.pretrain_ckpt)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(opt.training_params.pretrain_ckpt)
        else:
            logging.info('No pretrained model found. Creating model with new parameters...')
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

        with tf.GradientTape() as tape:
            logits = model((image, label[:, :-1]), training=True)
            loss = loss_func(y_true=label[:, 1:], y_pred=logits)
            training_loss += loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            logits = tf.argmax(logits, axis=2)
            logits = logits.numpy()
            labels = label[:, 1:].numpy()
            num_totals += len(logits)
            num_corrects += np.sum(np.sum(logits - labels, axis=1) == 0)

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
