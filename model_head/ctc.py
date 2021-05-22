from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

layers = tf.keras.layers

def CTC(input_tensor, num_class):
    x = layers.Dense(num_class)