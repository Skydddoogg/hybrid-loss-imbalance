import tensorflow as tf
from tensorflow import keras

def make_model(input_shape, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = keras.Sequential([
        keras.layers.Dense((input_shape[0] + 2) // 2, activation='relu', input_shape=input_shape),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])

    return model
