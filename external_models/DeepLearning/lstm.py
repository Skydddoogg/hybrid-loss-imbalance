from tensorflow import keras
from tensorflow.keras import layers

def make_model(max_features):

    embedding_dims = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    maxlen = 200

    model = keras.models.Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(layers.Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(layers.Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(layers.Conv1D(filters,
                    kernel_size,
                    padding='valid',
                    activation='relu',
                    strides=1))
    # we use max pooling:
    model.add(layers.GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(layers.Dense(hidden_dims))
    model.add(layers.Dropout(0.2))
    model.add(layers.Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    # model = keras.models.Sequential()
    # model.add(layers.Embedding(max_features, 128))
    # model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # model.add(layers.Dense(1, activation='sigmoid'))


    return model