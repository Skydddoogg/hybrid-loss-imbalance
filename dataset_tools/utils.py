import sys
sys.path.append("../")

import os
import numpy as np
from imblearn.datasets import fetch_datasets, make_imbalance
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from collections import Counter
from config import train_set_path, test_set_path

def split_train_test(dataset_name, iteration):
    
    print("Splitting data into training and test set ...")
    
    data = fetch_datasets()[dataset_name]
    
    X, y = get_Xy(data)
    
    # Get the locations of train and test folder
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

    for iter in range(iteration):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=iter, stratify=y)

        np.save(train_folder_path + '/X/' + str(iter + 1) + '.npy', X_train)
        np.save(test_folder_path + '/X/' + str(iter + 1) + '.npy', X_test)
        np.save(train_folder_path + '/y/' + str(iter + 1) + '.npy', y_train)
        np.save(test_folder_path + '/y/' + str(iter + 1) + '.npy', y_test)

def get_Xy(data):
    X = data.data
    y = data.target
    y[y == -1] = 0
    return X, y

def get_splitted_data(dataset, iteration):

    X_train = np.load(os.path.join(train_set_path, dataset, 'X', iteration + '.npy'))
    X_test = np.load(os.path.join(test_set_path, dataset, 'X', iteration + '.npy'))
    y_train = np.load(os.path.join(train_set_path, dataset, 'y', iteration + '.npy'))
    y_test = np.load(os.path.join(test_set_path, dataset, 'y', iteration + '.npy'))

    return X_train, X_test, y_train, y_test

def get_mocked_imbalanced_data(n_samples, n_features, neg_ratio, seed = 0):
    X, y = make_classification(n_samples = n_samples, n_features = n_features)

    class_count = Counter(y)
    remaining_ratio = 100
    step_down = int(neg_ratio * 100)
    sampling_strategy = {}

    for _class in set(y):
        sampling_strategy[_class] = (class_count[_class] * remaining_ratio) // 100
        remaining_ratio -= step_down

    X_imb, y_imb = make_imbalance(X, y, sampling_strategy = sampling_strategy)

    X_train, X_test, y_train, y_test = train_test_split(X_imb, y_imb, test_size = 0.20, random_state=seed, stratify=y_imb)

    return X_train, X_test, y_train, y_test