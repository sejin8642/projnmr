import numpy as np
import tensorflow as tf
from tensorflow import keras

# list of NMR NN models
def model_NMR(input_length, GRU_unit, first_filter_num, second_filter_num):
    # check your input arguments are powers of 2
    l2 = np.log2
    assert l2(input_length).is_integer(), "input_length is not a power of 2"
    assert l2(GRU_unit).is_integer(), "GRU_unit is not a power of 2"
    assert l2(first_filter_num).is_integer(), "first_filter_num is not a power of 2"
    assert l2(second_filter_num).is_integer(), "second_filter_num is not a power of 2"

    seq_input = keras.layers.Input(shape=[input_length])

    expand_layer = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
    expand_output = expand_layer(seq_input)

    GRU_output = keras.layers.Bidirectional(
        keras.layers.GRU(GRU_unit, return_sequences=True))(expand_output)

    expand_output2 = expand_layer(GRU_output)

    cnn_layer1 = keras.layers.Conv2D(
        filters=first_filter_num,
        kernel_size=(1, 2*GRU_unit),
        activation='elu') # elu
    cnn_output1 = cnn_layer1(expand_output2)

    transpose_fn = lambda x: tf.transpose(x, perm=[0, 1, 3, 2])
    transpose_layer = keras.layers.Lambda(transpose_fn)
    transpose_output = transpose_layer(cnn_output1)

    cnn_layer2 = keras.layers.Conv2D(
        filters=second_filter_num,
        kernel_size=(1, first_filter_num),
        activation='selu') # selu
    cnn2_output = cnn_layer2(transpose_output)

    transpose2_output = transpose_layer(cnn2_output)

    cnn_layer3 = keras.layers.Conv2D(
        filters=1,
        kernel_size=(1, second_filter_num),
        activation='LeakyReLU') # selu
    cnn3_output = cnn_layer3(transpose2_output)

    flat_output = keras.layers.Flatten()(cnn3_output)

    model_output = keras.layers.Add()([seq_input, flat_output])
    return keras.Model(inputs=[seq_input], outputs=[model_output])
