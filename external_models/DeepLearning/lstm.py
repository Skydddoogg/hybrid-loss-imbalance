from tensorflow import keras
from tensorflow.keras import layers

def make_model(max_features):

    # Embedding
    maxlen = 200
    embedding_size = 128

    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 70

    model = keras.models.Sequential()
    model.add(layers.Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv1D(filters,
                    kernel_size,
                    padding='valid',
                    activation='relu',
                    strides=1))
    model.add(layers.MaxPooling1D(pool_size=pool_size))
    model.add(layers.LSTM(lstm_output_size))
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    return model
