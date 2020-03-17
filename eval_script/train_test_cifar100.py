import sys
sys.path.append("../")

import numpy as np
from dataset_tools.utils import get_splitted_data, get_mocked_imbalanced_data
from sklearn import preprocessing
import os
import argparse
import warnings
from config import result_path, BATCH_SIZE, EPOCHS, LOSS, EARLY_STOPPING
from eval_script.utils import save_results
from external_models.DeepLearning import image_network, utils
import tensorflow as tf

warnings.filterwarnings('ignore')

def train_test(args_list):

    reduction_ratio, dataset_name, classification_algorithm, loss = args_list

    X_train, X_test, y_train, y_test = get_splitted_data(dataset_name, reduction_ratio)
    # X_train, X_test, y_train, y_test = get_mocked_imbalanced_data(n_samples = 100, n_features = 5, neg_ratio = 0.9)

    log_dir = "gs://sky-movo/class_imbalance/logs/fit/" + dataset_name + '/' + classification_algorithm
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Model
    model = image_network.make_model(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), loss = LOSS[loss])
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[EARLY_STOPPING, tensorboard_callback],
        verbose=0)
    
    # Get predictions
    y_pred = model.predict_classes(X_test).T[0]

    # Save
    save_results(y_test, y_pred, classification_algorithm, dataset_name, reduction_ratio)
    # utils.save_model(model, classification_algorithm + '_' + dataset_name + '_' + reduction_ratio, dataset_name)
    # utils.save_history(history, classification_algorithm + '_' + dataset_name + '_' + reduction_ratio, dataset_name)


if __name__ == '__main__':

    count = 1

    DATASETS = ['Tree1_cifar100', 'Tree2_cifar100']
    REDUCTION_RATIO = [20, 10, 5]

    for dataset in DATASETS:
        for loss in LOSS:
            args_list = [[str(reduction_ratio), dataset, 'dl-' + loss, loss] for reduction_ratio in REDUCTION_RATIO]
            for args in args_list:
                train_test(args)
        print("Completed evaluating on {0} ({1}/{2})".format(dataset, count, len(DATASETS)))
        count += 1
