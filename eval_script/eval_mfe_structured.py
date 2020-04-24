import sys
sys.path.append("../")

import numpy as np
from dataset_tools.utils import get_splitted_data, get_mocked_imbalanced_data
from sklearn import preprocessing
from sklearn.metrics import f1_score
import os
import argparse
from multiprocessing import Pool
import warnings
from config import DATASETS, ALPHA_RANGE, GAMMA_RANGE
from config_path import result_path
from eval_script.utils import save_results
from external_models.DeepLearning import structured_net, utils
import tensorflow as tf
from tensorflow import keras
from custom_functions import custom_loss

warnings.filterwarnings('ignore')

def train_test(args_list):

    iteration, dataset_name, classification_algorithm, loss, network = args_list

    X_train, X_test, y_train, y_test = get_splitted_data(dataset_name, iteration)

    # Normalize
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # log_dir = "gs://sky-movo/class_imbalance/logs/fit/" + dataset_name + '/' + classification_algorithm
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    loss_function = custom_loss.MeanFalseError().mean_false_error

    model = structured_net.make_model(input_shape = (X_train_scaled.shape[1],), loss = loss_function)
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[EARLY_STOPPING],
        verbose=0)

    # Get predictions
    y_pred = model.predict_classes(X_test_scaled).T[0]

    f1 = f1_score(y_test, y_pred)

    # Save
    save_results(y_test, y_pred, classification_algorithm, dataset_name, iteration, network)

    best_f1 = f1

    return best_f1


if __name__ == '__main__':

    ITERATION = 5
    EPOCHS = 200
    BATCH_SIZE = 64
    EARLY_STOPPING = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=1e-6,
        verbose=0,
        patience=10,
        mode='min',
        restore_best_weights=True)

    loss = 'MFE'
    network = "structured"

    count = 1
    for dataset in DATASETS:
        args_list = [[str(stage + 1), dataset, 'dl-' + loss, loss, network] for stage in range(ITERATION)]
        best_performance_collector = []
        for args in args_list:
            best_performance = train_test(args)
            best_performance_collector.append(best_performance)
        best_performance_collector = np.array(best_performance_collector)
        print("[{0}] Completed evaluating on {1} ({2:.2f}) - ({3}/{4})".format(loss, dataset, np.average(best_performance_collector), count, len(DATASETS)))
        count += 1
