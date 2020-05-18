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

def create_report(dataset_name, reduction_ratio, network, loss_list):

    f1_loss = {}
    best_loss = None
    comparison_param = {
        'f1': -np.Infinity,
        'std': -np.Infinity
    }

    for loss in loss_list:

        classification_algorithm = 'dl-' + loss
        f1_collector = []

        y_test = np.load(os.path.join(result_path, dataset_name, 'groundtruth', network + '_' + classification_algorithm + '_' + dataset_name + '_' + str(reduction_ratio) + ".npy"))
        y_pred = np.load(os.path.join(result_path, dataset_name, 'prediction', network + '_' + classification_algorithm + '_' + dataset_name + '_' + str(reduction_ratio) + ".npy"))

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

    dataset_name = 'imdb_reviews'

    LOSS_LIST = [
        'MFE',
        'MSFE',
        'FL',
        'Hybrid',
    ]

    model_architecture = 'lstm'

    REDUCTION_RATIO = [10]

    for ratio in REDUCTION_RATIO:
        print("{0} at {1} reduction ratio".format(dataset_name, ratio))
        create_report(dataset_name, ratio, model_architecture, LOSS_LIST)

