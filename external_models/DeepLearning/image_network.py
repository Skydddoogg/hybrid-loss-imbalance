import tensorflow as tf
from tensorflow import keras

from config import LOSS, METRICS, OPTIMIZER

def make_model(input_shape, loss, metrics = METRICS, optimizer = OPTIMIZER):

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(2),
    ])

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    return model
