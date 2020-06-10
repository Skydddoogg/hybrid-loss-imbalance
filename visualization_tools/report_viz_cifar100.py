import sys
sys.path.append("../")

import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, auc, roc_curve
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
    auc_loss = {}
    best_loss = None
    comparison_param = {
        'f1': -np.Infinity,
        'std': -np.Infinity
    }

    for loss in loss_list:

        classification_algorithm = 'dl-' + loss
        f1_collector = []
        auc_collector = []

        for _round in range(1, N_ROUND + 1):
            y_test = np.load(os.path.join(result_path, dataset_name, 'groundtruth', network + '_' + 'round_' + str(_round) + '_' + classification_algorithm + '_' + dataset_name + '_' + str(reduction_ratio) + ".npy"))
            y_pred = np.load(os.path.join(result_path, dataset_name, 'prediction', network + '_' + 'round_' + str(_round) + '_' + classification_algorithm + '_' + dataset_name + '_' + str(reduction_ratio) + ".npy"))
            y_prob = np.load(os.path.join(result_path, dataset_name, 'probability', network + '_' + 'round_' + str(_round) + '_' + classification_algorithm + '_' + dataset_name + '_' + str(reduction_ratio) + ".npy"))
    
            cm = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)


            f1 = f1_score(y_test, y_pred)
            _auc = auc(fpr, tpr)

            f1_collector.append(f1)
            auc_collector.append(_auc)

        f1_collector = np.array(f1_collector)
        f1_loss[loss] = f1_collector

        auc_collector = np.array(auc_collector)
        auc_loss[loss] = auc_collector

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
            print('%20s: %.2f±%.2f (AUC = %.2f) - Best' % (loss, np.average(f1_loss[loss]) * 100, np.std(f1_loss[loss]), np.average(auc_loss[loss]) * 100))
        else:
            print('%20s: %.2f±%.2f (AUC = %.2f)' % (loss, np.average(f1_loss[loss]) * 100, np.std(f1_loss[loss]), np.average(auc_loss[loss]) * 100))

    return f1_loss, auc_loss


if __name__ == '__main__':

    # Argument management
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    args = parser.parse_args()

    DATASETS = {
        'Household_cifar100': {
            'network': 'MFE_image_net2'
            }, 
        'Tree1_cifar100': {
            'network': 'MFE_image_net1'
            }, 
        'Tree2_cifar100': {
            'network': 'MFE_image_net1'
            }
        }

    LOSS_LIST = [
        'MFE',
        'MSFE',
        'FL',
        'Hybrid'
    ]

    N_ROUND = 5

    REDUCTION_RATIO = [20, 10, 5]

    for dataset_name in DATASETS:

        if args.network == '-':
            model_architecture = DATASETS[dataset_name]['network']
        else:
            model_architecture = args.network

        df_list = list()
        f1_ratio = {}
        auc_ratio = {}

        for ratio in REDUCTION_RATIO:
            print("{0} at {1} reduction ratio".format(dataset_name, ratio))
            f1, _auc = create_report(dataset_name, ratio, model_architecture, LOSS_LIST)
            f1_ratio[ratio] = f1
            auc_ratio[ratio] = _auc

        for loss in LOSS_LIST:
            print('{0} & {1:.2f} & {2:.2f} & {3:.2f} & {4:.2f} & {5:.2f} & {6:.2f}'.format(loss, np.average(f1_ratio[20][loss]) * 100, np.average(f1_ratio[10][loss]) * 100, np.average(f1_ratio[5][loss]) * 100, np.average(auc_ratio[20][loss]) * 100, np.average(auc_ratio[10][loss]) * 100, np.average(auc_ratio[5][loss]) * 100))

