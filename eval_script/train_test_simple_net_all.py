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
import tensorflow as tf
from config import result_path, ITERATION, BATCH_SIZE, EPOCHS, LOSS, DATASETS, EARLY_STOPPING, HP_FACTOR
from eval_script.utils import save_results
from external_models.DeepLearning import simple_net, utils

warnings.filterwarnings('ignore')

def train_test(args_list):

    iteration, dataset_name, classification_algorithm, loss = args_list

    log_dir = "gs://sky-movo/class_imbalance/logs/fit/" + dataset_name + '/' + classification_algorithm
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    X_train, X_test, y_train, y_test = get_splitted_data(dataset_name, iteration)
    # X_train, X_test, y_train, y_test = get_mocked_imbalanced_data(n_samples = 100, n_features = 5, neg_ratio = 0.9)

    # Normalize
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    comparable_f1 = -np.Infinity

    for factor in HP_FACTOR:

        # Model
        model = simple_net.make_model(input_shape = (X_train_scaled.shape[1],), loss = LOSS[loss], factor = factor)
        history = model.fit(
            X_train_scaled,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=[EARLY_STOPPING, tensorboard_callback],
            verbose=0)
        
        # Get predictions
        y_pred = model.predict_classes(X_test_scaled).T[0]

        f1 = f1_score(y_test, y_pred)

        if f1 > comparable_f1:
            # Save
            save_results(y_test, y_pred, classification_algorithm, dataset_name, iteration)
            utils.save_model(model, classification_algorithm + '_' + dataset_name + '_' + iteration, dataset_name)
            utils.save_history(history, classification_algorithm + '_' + dataset_name + '_' + iteration, dataset_name)

            comparable_f1 = f1

if __name__ == '__main__':

    count = 1
    for dataset in DATASETS:
        for loss in LOSS:
            args_list = [[str(stage + 1), dataset, 'dl-' + loss, loss] for stage in range(ITERATION)]
            for args in args_list:
                train_test(args)
        print("Completed evaluating on {0} ({1}/{2})".format(dataset, count, len(DATASETS)))
        count += 1