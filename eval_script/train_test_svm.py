import sys
sys.path.append("../")

from sklearn.metrics import f1_score
from external_models.CSSVM.python.svmutil import *
import numpy as np
from utils import get_splitted_data
from sklearn import preprocessing
import os
import argparse
from multiprocessing import Pool
import warnings
from config import result_path, ITERATION

warnings.filterwarnings('ignore')

def train_test(args_list):

    iteration, dataset_name, classification_algorithm = args_list
    print('Evaluating {0} on {1} at round {2} ...'.format(classification_algorithm, dataset_name, iteration))

    X_train, X_test, y_train, y_test = get_splitted_data(dataset_name, iteration)

    # Normalize
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = [dict(enumerate(i, 1)) for i in X_train_scaled]
    X_test_scaled = [dict(enumerate(i, 1)) for i in X_test_scaled]
    y_train = [1.0 if _y>0 else -1.0 for _y in y_train]
    y_test = [1.0 if _y>0 else -1.0 for _y in y_test]

    C_range = [10**x for x in range(-6, 7)]
    gamma_range = [10**x for x in range(-6, 7)]

    f1_comparable = -np.Infinity
    best_C = np.Infinity
    best_gamma = np.Infinity

    for C in C_range:
        for gamma in gamma_range:
            params = '-c {0} -t {1} -g {2}'.format(C, 2, gamma) # -t 0 is to linear kernel
            model = svm_train(y_train, X_train_scaled, params)
            y_pred, p_acc, p_val = svm_predict(y_test, X_test_scaled, model)
            y_pred = [1.0 if _y>0 else 0.0 for _y in y_pred]
            old_y_test = [1.0 if _y>0 else 0.0 for _y in y_test]

            f1 = f1_score(old_y_test, y_pred)

            if f1 >= f1_comparable:
                best_C = C
                best_gamma = gamma
                f1_comparable = f1

                # Save
                svm_save_model(os.path.join(result_path, 'model', classification_algorithm + '_' + dataset_name + '_' + iteration + ".model"), model)
                np.save(os.path.join(result_path, 'groundtruth', classification_algorithm + '_' + dataset_name + '_' + iteration + ".npy"), np.array(old_y_test))
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