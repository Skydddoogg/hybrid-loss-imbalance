import tensorflow as tf
from tensorflow import keras

from config import LOSS, METRICS, OPTIMIZER

def make_model(input_shape, loss, metrics = METRICS, optimizer = OPTIMIZER, output_bias=None, factor = 2.0):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    n_layers = 6

    model = keras.Sequential([
        keras.layers.Dense(int(factor * input_shape[0]), activation='relu', input_shape=input_shape),
        keras.layers.Dense(int(factor * input_shape[0] * ((n_layers - 1)/n_layers)), activation='relu'),
        keras.layers.Dense(int(factor * input_shape[0] * ((n_layers - 2)/n_layers)), activation='relu'),
        keras.layers.Dense(int(factor * input_shape[0] * ((n_layers - 3)/n_layers)), activation='relu'),
        keras.layers.Dense(int(factor * input_shape[0] * ((n_layers - 4)/n_layers)), activation='relu'),
        keras.layers.Dense(int(factor * input_shape[0] * ((n_layers - 5)/n_layers)), activation='relu'),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    return model
