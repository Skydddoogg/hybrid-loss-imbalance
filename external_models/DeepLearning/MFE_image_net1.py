import tensorflow as tf
from tensorflow import keras
import numpy as np

def make_model(input_shape, output_bias = None):

    def get_prediction(model, X_test):
        y_pred = model.predict_classes(X_test).T[0]
        return y_pred

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])

    # model.compile(
    #     optimizer=optimizer,
    #     loss=loss,
    #     metrics=metrics)

    return model, get_prediction