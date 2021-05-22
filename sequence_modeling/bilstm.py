from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

layers = tf.keras.layers

def BiLSTM(input_tensor, hidden_units=128):
    x = input_tensor
    x = layers.Bidirectional(
        layers.LSTM(units=hidden_units, return_sequences=True), name='bi_lstm1')(x)
    x = layers.Bidirectional(
        layers.LSTM(units=hidden_units, return_sequences=True), name='bi_lstm2')(x)

    return x

def attention_rnn(inputs):
    input_dim = int(inputs.shape[2])
    timestep = int(inputs.shape[1])
    a = layers.Permute((2, 1))(inputs)
    a = layers.Dense(timestep, activation='softmax')(a)
    a = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1), name='dim_reduction')(a)
    a = layers.RepeatVector(input_dim)(a)
    a_probs = layers.Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = layers.multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def Attention_BiLSTM(input_tensor, hidden_units=256):
    x = attention_rnn(input_tensor)
    x = BiLSTM(x, hidden_units)

    return x

if __name__ == "__main__":
    input_tensor = tf.zeros(shape=[64, 80, 512], dtype=tf.float32)

    features = BiLSTM(input_tensor)
    attention = attention_rnn(input_tensor)
    print(features.shape)
    print(attention.shape)

    image_input = tf.keras.layers.Input(shape=[80, 512])
    output = BiLSTM(image_input)
    model = tf.keras.Model(inputs=image_input, outputs=output, name='seq_modeling')
    print('BiLSTM parameters: ', model.count_params())

    image_input = tf.keras.layers.Input(shape=[80, 512])
    output = Attention_BiLSTM(image_input)
    model = tf.keras.Model(inputs=image_input, outputs=output, name='seq_modeling')
    print('Attention_BiLSTM parameters: ', model.count_params())