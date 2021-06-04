from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import numpy as np
import time

from PIL import Image
from six import BytesIO as IO


class Dataset(object):
    def __init__(self, annotation_fn, epochs, text_converter, image_converter):
        self.epochs = epochs
        self.text_converter = text_converter
        self.image_converter = image_converter
        self.ds_len = 0

        dataset = tf.data.TFRecordDataset(filenames=[annotation_fn])
        dataset = dataset.map(self._parse_record)

        counter = dataset.repeat(1)
        counter = dataset.batch(1)
        for ex in counter:
            self.ds_len += 1
        self.ds_len *= self.epochs

        dataset = dataset.shuffle(buffer_size=500000)
        self.dataset = dataset.repeat(self.epochs)

    def get_length(self):
        return tf.data.experimental.cardinality(self.dataset).numpy()

    def gen(self, batch_size):
        self.dataset = self.dataset.batch(batch_size)
        for batch in self.dataset:
            start = time.time()
            image, label, comment = batch

            image = [np.array(Image.open(IO(i)))[..., ::-1] for i in image.numpy()]
            label = [t.decode('utf-8') for t in label.numpy()]
            comment = comment.numpy()

            converted_label, length = self.text_converter.encode(label)
            converted_image = [self.image_converter(im) for im in image]
            yield converted_image, converted_label, length

    @staticmethod
    def _parse_record(example_proto):
        features = tf.io.parse_single_example(
            example_proto,
            features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.string),
                'comment': tf.io.FixedLenFeature([], tf.string, default_value=''),
            })
        return features['image'], features['label'], features['comment']

if __name__ == "__main__":
    pass

    
