import tensorflow as tf
from tensorflow import keras

from config import LOSS, METRICS, OPTIMIZER

def make_model(input_shape, loss, metrics = METRICS, optimizer = OPTIMIZER, output_bias=None, factor = 0.25):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = keras.Sequential([
        keras.layers.Dense(int(factor * input_shape[0]), activation='relu', input_shape=input_shape),
        keras.layers.Dense(int(factor * input_shape[0] * (2 / 3)), activation='relu'),
        keras.layers.Dense(int(factor * input_shape[0] * (1 / 3)), activation='relu'),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    return model