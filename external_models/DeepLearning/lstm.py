from tensorflow import keras
from tensorflow.keras import layers

def make_model(max_features):

    filters = 250
    kernel_size = 3
    hidden_dims = 250
    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1)(x)

    x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(hidden_dims)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Activation('relu')(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    # model = keras.models.Sequential()
    # model.add(layers.Embedding(max_features, 128))
    # model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # model.add(layers.Dense(1, activation='sigmoid'))


    return model