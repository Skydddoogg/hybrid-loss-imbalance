import numpy as np
import os
from config_path import result_path
from external_models.DeepLearning import resnetV2, MFE_image_net1, MFE_image_net2, densenet121

def save_results(y_true, y_pred, classification_algorithm, dataset_name, iteration, network):
    result_path_dataset = os.path.join(result_path, dataset_name)

    if not os.path.isdir(result_path_dataset):
        os.mkdir(result_path_dataset)
        os.mkdir(result_path_dataset + '/groundtruth')
        os.mkdir(result_path_dataset + '/prediction')
        os.mkdir(result_path_dataset + '/model')

    np.save(os.path.join(result_path_dataset, 'groundtruth', network + '_' + classification_algorithm + '_' + dataset_name + '_' + iteration + ".npy"), np.array(y_true))
    np.save(os.path.join(result_path_dataset, 'prediction', network + '_' + classification_algorithm + '_' + dataset_name + '_' + iteration + ".npy"), np.array(y_pred))

def choose_network(network_name):
    # Model
    print(network_name)
    if network_name == 'resnetV2':
        return resnetV2
    elif network_name == 'MFE_image_net1':
        return MFE_image_net1
    elif network_name == 'MFE_image_net2':
        return MFE_image_net2
    elif network_name == 'densenet121':
        return densenet121
