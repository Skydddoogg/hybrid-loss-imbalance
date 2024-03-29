import sys
sys.path.append("../")

import tensorflow_datasets as tfds
import numpy as np
from config_path import train_set_path, test_set_path
import os
from tensorflow import keras

def get_data_for_one_class(X, y, _classes):

    _class_indices = []
    for _class in _classes:

        _class_indices += list(np.where(y == _class)[0])

    desired_X = X[_class_indices]
    desired_y = y[_class_indices]

    return desired_X, desired_y

def make_new_label(y, classes, target):

    for _class in classes:
        y[y == _class] = target

    return y

def get_reduced_data(X, y, reduction_ratio):

    desired_indices = np.random.choice([i for i in range(y.shape[0])], (y.shape[0] * reduction_ratio) // 100)

    desired_X, desired_y = X[desired_indices], y[desired_indices]

    return desired_X, desired_y

def shuffle_Xy(X, y):
    indices = np.array([i for i in range(y.shape[0])])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y

if __name__ == "__main__":

    dataset_name = 'imdb_reviews'

    max_features = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

    train_folder_path = os.path.join(train_set_path, dataset_name)
    test_folder_path = os.path.join(test_set_path, dataset_name)

    if not os.path.isdir(train_folder_path):
        os.mkdir(train_folder_path)
        os.mkdir(train_folder_path + '/X')
        os.mkdir(train_folder_path + '/y')
    if not os.path.isdir(test_folder_path):
        os.mkdir(test_folder_path)
        os.mkdir(test_folder_path + '/X')
        os.mkdir(test_folder_path + '/y')

    IMB_RATIO = [20, 10, 5]

    negative_class = [0]
    positive_class = [1]

    negative_class_X_train, negative_class_y_train = get_data_for_one_class(X_train, y_train, negative_class)
    negative_class_X_test, negative_class_y_test = get_data_for_one_class(X_test, y_test, negative_class)

    positive_class_X_train, positive_class_y_train = get_data_for_one_class(X_train, y_train, positive_class)
    positive_class_X_test, positive_class_y_test = get_data_for_one_class(X_test, y_test, positive_class)

    in_X_train = positive_class_X_train
    in_X_test = positive_class_X_test
    in_y_train = positive_class_y_train
    in_y_test = positive_class_y_test

    for ratio in IMB_RATIO:
        print("{0}: Reducing representation of positive class to {1}%".format(dataset_name, ratio))
        reduced_positive_class_X_train, reduced_positive_class_y_train = get_reduced_data(positive_class_X_train, positive_class_y_train, ratio)
        reduced_positive_class_X_test, reduced_positive_class_y_test = get_reduced_data(positive_class_X_test, positive_class_y_test, ratio)

        in_X_train = reduced_positive_class_X_train
        in_X_test = reduced_positive_class_X_test
        in_y_train = reduced_positive_class_y_train
        in_y_test = reduced_positive_class_y_test

        combined_X_train, combined_y_train = np.concatenate((negative_class_X_train, reduced_positive_class_X_train), axis=0), np.concatenate((negative_class_y_train, reduced_positive_class_y_train), axis=0)
        combined_X_test, combined_y_test = np.concatenate((negative_class_X_test, reduced_positive_class_X_test), axis=0), np.concatenate((negative_class_y_test, reduced_positive_class_y_test), axis=0)

        combined_X_train, combined_y_train = shuffle_Xy(combined_X_train, combined_y_train)
        combined_X_test, combined_y_test = shuffle_Xy(combined_X_test, combined_y_test)

        np.save(train_folder_path + '/X/' + str(ratio) + '.npy', combined_X_train)
        np.save(test_folder_path + '/X/' + str(ratio) + '.npy', combined_X_test)
        np.save(train_folder_path + '/y/' + str(ratio) + '.npy', combined_y_train)
        np.save(test_folder_path + '/y/' + str(ratio) + '.npy', combined_y_test)
