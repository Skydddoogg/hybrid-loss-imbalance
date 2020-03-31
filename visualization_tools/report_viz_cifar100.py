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
from config import LOSS

def create_report(dataset_name, classification_algorithm, reduction_ratio, network):

    y_test = np.load(os.path.join(result_path, dataset_name, 'groundtruth', network + '_' + classification_algorithm + '_' + dataset_name + '_' + str(reduction_ratio) + ".npy"))
    y_pred = np.load(os.path.join(result_path, dataset_name, 'prediction', network + '_' + classification_algorithm + '_' + dataset_name + '_' + str(reduction_ratio) + ".npy"))

    cm = confusion_matrix(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    print('%20s: %.2f' % (classification_algorithm, f1))


if __name__ == '__main__':

    # Argument management
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    args = parser.parse_args()

    DATASETS = ['Tree1_cifar100', 'Tree2_cifar100', 'Household_cifar100']
    REDUCTION_RATIO = [20, 10, 5]

    for dataset_name in DATASETS:
        df_list = list()
        for ratio in REDUCTION_RATIO:
            print("{0} at {1} reduction ratio".format(dataset_name, ratio))
            for loss in LOSS:
                create_report(dataset_name, 'dl-' + loss, ratio, args.network)

