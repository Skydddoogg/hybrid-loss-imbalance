import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model

def make_model(input_shape, num_classes = 1):

    def get_prediction(model, X_test):
        y_pred = model.predict(X_test)
        y_pred = np.reshape(y_pred, (y_pred.shape[0],))

        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        return y_pred

    x = tf.keras.applications.densenet.DenseNet121(include_top=False, input_shape=input_shape, pooling='avg', weights=None)

    inputs = x.input
    flatten = Flatten()(x.output)
    outputs = Dense(num_classes, activation='sigmoid')(flatten)
    model = Model(inputs, outputs)

    return model, get_prediction