import sys
sys.path.append("../")

import numpy as np
from dataset_tools.utils import get_splitted_data, get_mocked_imbalanced_data
from sklearn import preprocessing
import os
import argparse
from multiprocessing import Pool
import warnings
from config import result_path, ITERATION, BATCH_SIZE, EPOCHS, LOSS, EARLY_STOPPING
from eval_script.utils import save_results
from external_models.DeepLearning import simple_net, utils

warnings.filterwarnings('ignore')

def train_test(args_list):

    iteration, dataset_name, classification_algorithm, loss = args_list
    print('Evaluating {0} on {1} at round {2} ...'.format(classification_algorithm, dataset_name, iteration))

    X_train, X_test, y_train, y_test = get_splitted_data(dataset_name, iteration)
    # X_train, X_test, y_train, y_test = get_mocked_imbalanced_data(n_samples = 100, n_features = 5, neg_ratio = 0.9)

    # Normalize
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = simple_net.make_model(input_shape = (X_train_scaled.shape[1],), loss = LOSS[loss])
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

    # Save
    save_results(y_test, y_pred, classification_algorithm, dataset_name, iteration)
    utils.save_model(model, classification_algorithm + '_' + dataset_name + '_' + iteration, dataset_name)
    utils.save_history(history, classification_algorithm + '_' + dataset_name + '_' + iteration, dataset_name)

if __name__ == '__main__':

    # Argument management
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("classification_algorithm")
    parser.add_argument("loss")
    args = parser.parse_args()

    args_list = [[str(stage + 1), args.dataset_name, args.classification_algorithm, args.loss] for stage in range(ITERATION)]

    # with Pool(processes=5) as pool:
    #     pool.map(train_test, args_list)
    for args in args_list:
        train_test(args)
