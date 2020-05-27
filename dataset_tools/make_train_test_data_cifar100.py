import sys
sys.path.append("../")

import tensorflow_datasets as tfds
import numpy as np
from config_path import train_set_path, test_set_path
import os

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

def get_reduced_data(X, y, n_desired):

    desired_indices = np.random.choice([i for i in range(y.shape[0])], n_desired)

    desired_X, desired_y = X[desired_indices], y[desired_indices]

    return desired_X, desired_y

def shuffle_Xy(X, y):
    indices = np.array([i for i in range(y.shape[0])])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y

if __name__ == "__main__":

    dataset_name = 'cifar100'

    train_ds = tfds.load(dataset_name, split=tfds.Split.TRAIN, batch_size=-1)
    test_ds = tfds.load(dataset_name, split=tfds.Split.TEST, batch_size=-1)
    numpy_train_ds = tfds.as_numpy(train_ds)
    numpy_test_ds = tfds.as_numpy(test_ds)
    X_train, y_train, X_test, y_test = numpy_train_ds["image"] / 255.0, numpy_train_ds["label"], numpy_test_ds['image'] / 255.0, numpy_test_ds['label']

    ex_dataset = {
        'Tree1': {
            'positive_class': [47], # 47 is maple tree
            'negative_class': [52] # 52 is oak tree
        },
        'Tree2': {
            'positive_class': [47], # 47 is maple tree
            'negative_class': [56] # 56 is palm tree
        },
        'Household':{
            'positive_class': [22, 39, 40, 86, 87],
            'negative_class': [5, 20, 25, 84, 94]
        }
    }

    for ex in ex_dataset:

        negative_class = ex_dataset[ex]['negative_class']
        positive_class = ex_dataset[ex]['positive_class']

        negative_class_X_train, negative_class_y_train = get_data_for_one_class(X_train, y_train, negative_class)
        negative_class_X_test, negative_class_y_test = get_data_for_one_class(X_test, y_test, negative_class)
        negative_class_y_train = make_new_label(negative_class_y_train, negative_class, 0)
        negative_class_y_test = make_new_label(negative_class_y_test, negative_class, 0)

        positive_class_X_train, positive_class_y_train = get_data_for_one_class(X_train, y_train, positive_class)
        positive_class_X_test, positive_class_y_test = get_data_for_one_class(X_test, y_test, positive_class)
        positive_class_y_train = make_new_label(positive_class_y_train, positive_class, 1)
        positive_class_y_test = make_new_label(positive_class_y_test, positive_class, 1)

        train_folder_path = os.path.join(train_set_path, ex + '_' + dataset_name)
        test_folder_path = os.path.join(test_set_path, ex + '_' + dataset_name)

        if not os.path.isdir(train_folder_path):
            os.mkdir(train_folder_path)
            os.mkdir(train_folder_path + '/X')
            os.mkdir(train_folder_path + '/y')
        if not os.path.isdir(test_folder_path):
            os.mkdir(test_folder_path)
            os.mkdir(test_folder_path + '/X')
            os.mkdir(test_folder_path + '/y')

        REDUCTION_RATIO = [20, 10, 5]

        in_X_train = positive_class_X_train
        in_X_test = positive_class_X_test
        in_y_train = positive_class_y_train
        in_y_test = positive_class_y_test

        for ratio in REDUCTION_RATIO:

            train_n_desired = (positive_class_y_train.shape[0] * ratio) // 100
            test_n_desired = (positive_class_y_test.shape[0] * ratio) // 100

            print("{0}: Reducing representation of positive class to {1}% ({2}, {3})".format(ex, ratio, train_n_desired, test_n_desired))

            reduced_positive_class_X_train, reduced_positive_class_y_train = get_reduced_data(in_X_train, in_y_train, train_n_desired)
            reduced_positive_class_X_test, reduced_positive_class_y_test = get_reduced_data(in_X_test, in_y_test, test_n_desired)

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
