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

from config import result_path, ITERATION, DATASETS, LOSS, viz_path

def create_report_dataframe(dataset_name, classification_algorithm, report_label):

    average_cm = np.zeros((2, 2))

    f1_collector = list()

    for _iter in range(1, ITERATION + 1):

        y_test = np.load(os.path.join(result_path, dataset_name, 'groundtruth', classification_algorithm + '_' + dataset_name + '_' + str(_iter) + ".npy"))
        y_pred = np.load(os.path.join(result_path, dataset_name, 'prediction', classification_algorithm + '_' + dataset_name + '_' + str(_iter) + ".npy"))       

        cm = confusion_matrix(y_test, y_pred)

        f1 = f1_score(y_test, y_pred)

        f1_collector.append(f1)

        average_cm += cm

    arr_f1_collector = np.array(f1_collector)

    average_cm = average_cm / ITERATION

    # Plot
    df_cm = pd.DataFrame(average_cm, columns=np.unique([0, 1]), index = np.unique([0, 1]))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (8,6))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, fmt='g')# font size

    tmp_label = '{0} ({1:.2f}±{2:.2f})'.format(report_label, np.average(arr_f1_collector), np.std(arr_f1_collector))
    data = {
        'Round': [x for x in range(1, ITERATION + 1)],
        tmp_label: arr_f1_collector
    }
    df = pd.DataFrame (data, columns = data.keys())
    plt.title("{0} - {1}".format(dataset_name, report_label))

    plt.savefig(os.path.join(viz_path, 'performance_report', 'confusion_mat_' + classification_algorithm + '_' + dataset_name + '.png'))

    return df

def plot_performance_lines(df, dataset_name):
        
    plt.figure(figsize = (10,7))
    idx = [x for x in range(1, ITERATION + 1)]
    for model in list(df.columns):
        if model != 'Round':
            plt.plot(range(len(idx)), model, data=df)
    plt.xticks(range(len(idx)), idx)
    plt.xlabel('Round')
    plt.ylabel('F1 score')
    plt.title(dataset_name)
    plt.legend()
    plt.savefig(os.path.join(viz_path, 'performance_report', 'comparison_graph_' + dataset_name + '.png'))

if __name__ == '__main__':

    for dataset_name in DATASETS:
        df_list = list()
        for loss in LOSS:
            df = create_report_dataframe(dataset_name, 'dl-' + loss, loss)
            df_list.append(df)

        concat_df = pd.concat(df_list, axis=1)
        concat_df = concat_df.loc[:,~concat_df.columns.duplicated()]
        plot_performance_lines(concat_df, dataset_name)

