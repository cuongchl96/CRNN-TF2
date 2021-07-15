from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import tensorflow as tf
import math
import random
import cv2

from PIL import Image

from modules.data_helper import augmenter


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, opt):
        self.opt = opt
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']
        list_character = list(self.opt.model_params.character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length = self.opt.model_params.max_len + 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = np.zeros((len(text), batch_max_length + 1), dtype=int)
        for i, t in enumerate(text):
            cur_text = list(t)
            cur_text.append('[s]')
            cur_text = [self.dict[char] for char in cur_text]
            batch_text[i][1:1 + len(cur_text)] = cur_text
        return tf.convert_to_tensor(batch_text, dtype=tf.int32), tf.convert_to_tensor(length, dtype=tf.int32)

    def decode(self, text_index):
        """ convert text-index into text-label. """
        texts = []
        for idx, text_idx in enumerate(text_index):
            text = ''
            text_idx = text_idx.numpy()
            for i in range(len(text_idx)):
                if text_idx[i] == 1:
                    break
                else:
                    text += self.character[text_idx[i]]
            texts.append(text)
        return texts

class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        return img


class NormalizePAD(object):
    def __init__(self, opt, is_training):
        self.opt = opt
        self.max_size = (opt.model_params.imgH, opt.model_params.imgW, opt.model_params.num_channels)
        self.is_training = is_training
        self.augment = augmenter.RandAugment(magnitude=opt.data_augmentation.augment_level)

    def __call__(self, img):
        h, w, c = img.shape
        ratio = w / float(h)
        if math.ceil(self.max_size[0] * ratio) > self.max_size[1]:
            resized_w = self.max_size[1]
        else:
            resized_w = math.ceil(self.max_size[0] * ratio)

        img = cv2.resize(img, (resized_w, self.max_size[0]))

        h, w = img.shape[:2]
        pad_img = np.zeros((self.max_size[0], self.max_size[1], 3), dtype=np.uint8)
        if self.is_training:
            start_w = random.randint(0, self.max_size[1] - w)
        else:
            start_w = 0
        
        pad_img[:, start_w:start_w + w, :] = img

        if self.is_training and random.random() < self.opt.data_augmentation.prob:
            pad_img = tf.convert_to_tensor(pad_img)
            pad_img = self.augment.distort(pad_img)
            pad_img = pad_img.numpy().astype(np.uint8)

        if self.max_size[-1] == 1:
            pad_img = cv2.cvtColor(pad_img, cv2.COLOR_BGR2GRAY)
            pad_img = np.expand_dims(pad_img, -1)

        pad_img = pad_img.astype(np.float32) / 255.
        pad_img = (pad_img - 0.5) / 0.5
        return tf.convert_to_tensor(pad_img)

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, opt):
        # character (str): set of the possible characters.
        self.opt = opt

        list_token = ['[CTCblank]']
        list_character = list(self.opt.model_params.character)
        self.character = list_character + list_token

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        batch_max_length = self.opt.model_params.max_len
        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = np.zeros((len(text), batch_max_length), dtype=int)
        for i, t in enumerate(text):
            cur_text = list(t)
            cur_text = [self.dict[char] for char in cur_text]
            batch_text[i][:len(cur_text)] = cur_text

        return tf.convert_to_tensor(batch_text, dtype=tf.int32), tf.convert_to_tensor(length, dtype=tf.int32)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts

if __name__ == "__main__":
    pass
