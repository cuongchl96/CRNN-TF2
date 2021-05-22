from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import numpy as np
from feature_extractor.resnet import resnet18_slim, resnet50_slim, resnet50, resnet18
from sequence_modeling.bilstm import Attention_BiLSTM, BiLSTM
from model_head.attention import Attention
from data_helper.data_generator import Dataset
from data_helper.data_utils import AttnLabelConverter, NormalizePAD
from model import get_crnn_attention_inference_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=280, help='the width of the input image')
    parser.add_argument('--max_len', type=int, default=12, help='maximum-label-length')
    parser.add_argument('--hidden_size', type=int, default=512, help='the size of the LSTM hidden state')
    parser.add_argument('--character', type=str, default='0123456789', help='character label')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
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
    logits = get_crnn_attention_inference_model(image_tensor, text_tensor, opt)

    model = tf.keras.Model(inputs=[image_tensor, text_tensor], outputs=logits)
    print(model.count_params())
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0)

    model.load_weights('checkpoints/model')

    idnum = Dataset('datasets/IDNum/idnum_test.tfrecords', epochs=1, text_converter=text_converter, image_converter=image_converter)
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


        # with tf.GradientTape() as tape:
        #     logits = model((image, label[:, :-1]), training=True)
        #     loss = loss_func(y_true=label[:, 1:], y_pred=logits)
        #     print(loss)
        #     grads = tape.gradient(loss, model.trainable_variables)

        #     optimizer.apply_gradients(zip(grads, model.trainable_variables))

