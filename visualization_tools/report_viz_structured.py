import sys
sys.path.append("../")

import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import argparse
from joblib import dump, load
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from config_path import result_path, viz_path
from config import DATASETS

def create_report(dataset_name, network, loss_list):

    f1_loss = {}
    best_loss = None
    comparison_param = {
        'f1': -np.Infinity,
        'std': -np.Infinity
    }

    result_path_dataset = os.path.join(result_path, dataset_name)

    for loss in loss_list:

        classification_algorithm = 'dl-' + loss
        f1_collector = []

        for _round in range(1, N_ROUND + 1):
            y_test = np.load(os.path.join(result_path_dataset, 'groundtruth', network + '_' + classification_algorithm + '_' + dataset_name + '_' + str(_round) + ".npy"))
            y_pred = np.load(os.path.join(result_path_dataset, 'prediction', network + '_' + classification_algorithm + '_' + dataset_name + '_' + str(_round) + ".npy"))

            cm = confusion_matrix(y_test, y_pred)

            f1 = f1_score(y_test, y_pred)

            f1_collector.append(f1)

        f1_collector = np.array(f1_collector)
        f1_loss[loss] = f1_collector

        if np.average(f1_collector) > comparison_param['f1']:
            comparison_param['f1'] = np.average(f1_collector)
            comparison_param['std'] = np.std(f1_collector)
            best_loss = loss
        elif np.average(f1_collector) == comparison_param['f1']:
            if comparison_param['std'] > np.std(f1_collector):
                comparison_param['f1'] = np.average(f1_collector)
                comparison_param['std'] = np.std(f1_collector)
                best_loss = loss
        else:
            pass

    for loss in LOSS_LIST:
        if loss == best_loss:
            print('%20s: %.2f±%.2f - Best' % (loss, np.average(f1_loss[loss]), np.std(f1_loss[loss])))
        else:
            print('%20s: %.2f±%.2f' % (loss, np.average(f1_loss[loss]), np.std(f1_loss[loss])))


if __name__ == '__main__':

    LOSS_LIST = [
        'MFE',
        'MSFE',
        'Balanced-FL',
        'Balanced-Hybrid'
    ]

    N_ROUND = 5

    model_architecture = 'structured'

    for dataset_name in DATASETS:

        print(dataset_name)

        create_report(dataset_name, model_architecture, LOSS_LIST)
        print(" ")

