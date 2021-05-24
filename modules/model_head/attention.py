from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch

layers = tf.keras.layers

class AttentionCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size):
        super(AttentionCell, self).__init__()
        self.i2h = layers.Dense(hidden_size, use_bias=False)
        self.h2h = layers.Dense(hidden_size, use_bias=False)
        self.score = layers.Dense(1, use_bias=False)
        self.hidden_size = hidden_size
        self.rnn = layers.LSTMCell(hidden_size)

    def __call__(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = tf.expand_dims(self.h2h(prev_hidden[0]), 1)

        e = self.score(tf.math.tanh(batch_H_proj + prev_hidden_proj))
        alpha = tf.nn.softmax(e)

        context_vector = tf.matmul(layers.Permute((2, 1))(alpha), batch_H)
        context_vector = tf.squeeze(context_vector, axis=1)
        context_vector_concat = tf.concat(
            [context_vector, char_onehots], axis=1)

        output, cur_hidden = self.rnn(context_vector_concat, prev_hidden)
        return cur_hidden, alpha


class Attention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.generator = layers.Dense(self.num_classes)
        self.attention_cell = AttentionCell(self.hidden_size)

    def _char_to_onehot(self, input_char, onehot_dim):
        return tf.one_hot(indices=input_char, depth=onehot_dim)

    def __call__(self, batch_H, text, is_train, batch_max_length):
        batch_size = tf.shape(batch_H)[0]
        num_steps = batch_max_length + 1

        output_hiddens = tf.zeros([num_steps, batch_size, self.hidden_size])
        hidden = (tf.zeros([batch_size, self.hidden_size]),
                  tf.zeros([batch_size, self.hidden_size]))

        if is_train:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    text[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(
                    hidden, batch_H, char_onehots)
                # Assign to probs
                indices = [[i]]
                output_hiddens = tf.tensor_scatter_nd_update(output_hiddens, indices, tf.expand_dims(hidden[0], axis=0))
            output_hiddens = tf.transpose(output_hiddens, perm=[1, 0, 2])
            probs = self.generator(output_hiddens)
        else:
            targets = tf.zeros([batch_size], dtype=tf.int32)
            probs = tf.zeros([num_steps, batch_size, self.num_classes])
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(
                    hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])

                # Assign to probs
                indices = [[i]]
                probs = tf.tensor_scatter_nd_update(probs, indices, tf.expand_dims(probs_step, axis=0))
                next_input = tf.argmax(probs_step, axis=1)
                targets = tf.cast(next_input, tf.int32)

            probs = tf.transpose(probs, perm=[1, 0, 2])
        return probs


if __name__ == "__main__":
    # batch_H = tf.zeros(shape=[64, 80, 512])
    # char_onehots = tf.zeros(shape=[64, 25])
    # hidden = (tf.zeros(shape=[64, 512]), tf.zeros(shape=[64, 512]))

    # attn = AttentionCell(512)

    # output = attn(hidden, batch_H, char_onehots)

    attn = Attention(512, 25)
    batch_H = tf.keras.layers.Input(shape=[80, 512])
    # batch_H = tf.zeros(shape=[3, 80, 512])
    text = tf.constant(
        [[0, 1, 2, 11, 3, 4], [0, 1, 2, 1, 2, 1], [0, 14, 15, 16, 17, 18]], dtype=tf.int32)
    attn(batch_H, text, True, 5)
