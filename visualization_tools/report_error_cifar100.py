import sys
sys.path.append("../")

import numpy as np
import os
import argparse
from config_path import result_path

def unique_count(a):
    unique, counts = np.unique(a, return_counts=True)
    return dict(zip(unique, counts))

def compute_error(y_true, y_pred, positive_prob):

    pos = positive_prob
    neg = 1 - pos
    pos_error = 0
    neg_error = 0
    
    count = unique_count(y_true)
    
    y_pred_logit = y_pred.copy()

    y_pred_logit[y_pred_logit >= 0.5] = 1
    y_pred_logit[y_pred_logit < 0.5] = 0
    
    pos_acc = 0
    neg_acc = 0

    for idx in range(y_true.shape[0]):
        if (y_true[idx] == 0) and (y_true[idx] == y_pred_logit[idx]):
            neg_acc += 1
        elif (y_true[idx] == 1) and (y_true[idx] == y_pred_logit[idx]):
            pos_acc += 1
            
    pos_error = 1 - (pos_acc / count[1])
    neg_error = 1 - (neg_acc / count[0])

    print("positive: {0:.2f}, negative: {1:.2f}".format((pos_error * 100), (neg_error * 100)))
    print("mean: %.2f" % (((pos_error + neg_error) / 2) * 100))

    return pos_error * 100, neg_error * 100

if __name__ == "__main__":

    # Argument management
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("network")
    args = parser.parse_args()



    dataset_name = args.dataset + '_cifar100'
    network = args.network
    LOSS_LIST = [
        'MFE',
        'MSFE',
        'FL',
        'Hybrid',
        ]
    N_ROUND = 5
    pos_error_ratio = {}
    neg_error_ratio = {}
    for ratio in [20, 10, 5]:
        print(ratio)
        pos_error_loss = {}
        neg_error_loss = {}
        for loss in LOSS_LIST:
            print(loss, '----------------------------------------------------')
            classification_algorithm = 'dl-' + loss
            pos_error_list = []
            neg_error_list = []
            for _round in range(1, N_ROUND + 1):
                y_true = np.load(os.path.join(result_path, dataset_name, 'groundtruth', network + '_' + 'round_' + str(_round) + '_' + classification_algorithm + '_' + dataset_name + '_' + str(ratio) + ".npy"))
                y_pred_prob = np.load(os.path.join(result_path, dataset_name, 'probability', network + '_' + 'round_' + str(_round) + '_' + classification_algorithm + '_' + dataset_name + '_' + str(ratio) + ".npy"))
                y_pred = np.load(os.path.join(result_path, dataset_name, 'prediction', network + '_' + 'round_' + str(_round) + '_' + classification_algorithm + '_' + dataset_name + '_' + str(ratio) + ".npy"))
                _pos, _neg = compute_error(y_true, y_pred, y_pred_prob)

                pos_error_list.append(_pos)
                neg_error_list.append(_neg)

            pos_error_loss[loss] = np.array(pos_error_list)
            neg_error_loss[loss] = np.array(neg_error_list)

        pos_error_ratio[ratio] = pos_error_loss
        neg_error_ratio[ratio] = neg_error_loss

    for loss in LOSS_LIST:
        mean_error_20 = np.array([neg_error_ratio[20][loss], pos_error_ratio[20][loss]])
        mean_error_10 = np.array([neg_error_ratio[10][loss], pos_error_ratio[10][loss]])
        mean_error_5 = np.array([neg_error_ratio[5][loss], pos_error_ratio[5][loss]])

        print('{0} & {1:.2f}\pm{2:.2f} & {3:.2f}\pm{4:.2f} & {5:.2f}\pm{6:.2f} & {7:.2f}\pm{8:.2f} & {9:.2f}\pm{10:.2f} & {11:.2f}\pm{12:.2f} & {13:.2f}\pm{14:.2f} & {15:.2f}\pm{16:.2f} & {17:.2f}\pm{18:.2f}'.format(loss, np.average(pos_error_ratio[20][loss]), np.std(pos_error_ratio[20][loss]), np.average(pos_error_ratio[10][loss]), np.std(pos_error_ratio[10][loss]), np.average(pos_error_ratio[5][loss]), np.std(pos_error_ratio[5][loss]), np.average(neg_error_ratio[20][loss]), np.std(neg_error_ratio[20][loss]), np.average(neg_error_ratio[10][loss]), np.std(neg_error_ratio[10][loss]), np.average(neg_error_ratio[5][loss]), np.std(neg_error_ratio[5][loss]), np.average(mean_error_20), np.std(mean_error_20), np.average(mean_error_10), np.std(mean_error_10), np.average(mean_error_5), np.std(mean_error_5)))
