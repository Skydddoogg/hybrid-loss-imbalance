import sys
sys.path.append("../")
import numpy as np
from utils import get_splitted_data
from sklearn import preprocessing
import os
import argparse
from multiprocessing import Pool
import warnings
from config import result_path, ITERATION, BATCH_SIZE, EPOCHS

from external_models.DeepLearning import simple_net, utils

warnings.filterwarnings('ignore')

def train_test(args_list):

    iteration, dataset_name, classification_algorithm = args_list
    print('Evaluating {0} on {1} at round {2} ...'.format(classification_algorithm, dataset_name, iteration))

    X_train, X_test, y_train, y_test = get_splitted_data(dataset_name, iteration)

    # Normalize
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = simple_net.make_model((X_train_scaled.shape[1],))
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=0)
    
    # Get predictions
    y_pred = model.predict_classes(X_test_scaled).T[0]

    # Save
    utils.save_model(model, classification_algorithm + '_' + dataset_name + '_' + iteration)
    utils.save_history(history, classification_algorithm + '_' + dataset_name + '_' + iteration)
    np.save(os.path.join(result_path, 'groundtruth', classification_algorithm + '_' + dataset_name + '_' + iteration + ".npy"), np.array(y_test))
    np.save(os.path.join(result_path, 'prediction', classification_algorithm + '_' + dataset_name + '_' + iteration + ".npy"), np.array(y_pred))

if __name__ == '__main__':

    # Argument management
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("classification_algorithm")
    args = parser.parse_args()

    args_list = [[str(stage + 1), args.dataset_name, args.classification_algorithm] for stage in range(ITERATION)]

    with Pool(processes=5) as pool:
        pool.map(train_test, args_list)
    # for args in args_list:
    #     train_test(args)