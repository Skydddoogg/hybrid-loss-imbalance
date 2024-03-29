import sys
sys.path.append("../")

import numpy as np
from dataset_tools.utils import get_splitted_data, get_mocked_imbalanced_data
from sklearn import preprocessing
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import warnings
from eval_script.config_cifar100 import BATCH_SIZE, EPOCHS, EARLY_STOPPING, SEED, METRICS, N_ROUND
from config_path import result_path
from eval_script.utils import save_results, choose_network
from external_models.DeepLearning import resnetV2, utils, MFE_image_net1, MFE_image_net2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from custom_functions import custom_loss

warnings.filterwarnings('ignore')

# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# tf.random.set_seed(SEED)

def train_test(args_list):

    reduction_ratio, dataset_name, classification_algorithm, loss, network, _round = args_list

    # X_train, X_test, y_train, y_test = get_splitted_data(dataset_name, reduction_ratio)
    X_train, X_test, X_valid, y_train, y_test, y_valid = get_splitted_data(dataset_name, reduction_ratio, validation = True, seed = _round)

    # Prepare callbacks
    # log_dir = "gs://sky-movo/class_imbalance/cifar100_logs/fit/" + network + '/' + dataset_name + '/' + 'reduction_ratio_' + reduction_ratio + '/' + classification_algorithm
    # log_dir = "cifar100_logs/fit/" + dataset_name + '/' + 'reduction_ratio_' + reduction_ratio + '/' + classification_algorithm
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    # Model
    model, get_prediction = choose_network(network).make_model(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    loss_function = custom_loss.MeanFalseError().mean_false_error

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_function,
        metrics=METRICS)

    initial_weight_path = os.path.join('model_initial_weights')

    if not os.path.isdir(initial_weight_path):
        os.mkdir(initial_weight_path)
        os.mkdir(os.path.join(initial_weight_path, network))
        utils.make_initial_weights(model, os.path.join(initial_weight_path, network, 'initial_weights'))
    else:
        if not os.path.isdir(os.path.join(initial_weight_path, network)):
            os.mkdir(os.path.join(initial_weight_path, network))
            utils.make_initial_weights(model, os.path.join(initial_weight_path, network, 'initial_weights'))

    model.load_weights(os.path.join(initial_weight_path, network, 'initial_weights', 'initial_weights'))
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_valid, y_valid),
        verbose=0)

    # Get predictions
    y_pred = get_prediction(model, X_test)
    y_pred_prob = model.predict(X_test)
    y_pred_prob = np.reshape(y_pred_prob, (y_pred_prob.shape[0],))

    # Save
    save_results(y_test, y_pred, y_pred_prob, 'round_' + str(_round) + '_' + classification_algorithm, dataset_name, reduction_ratio, network)
    # utils.save_model(model, classification_algorithm + '_' + dataset_name + '_' + reduction_ratio, dataset_name)
    utils.save_history(history, 'round_' + str(_round) + '_' + classification_algorithm + '_' + dataset_name + '_' + reduction_ratio, dataset_name)


if __name__ == '__main__':

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
    REDUCTION_RATIO = [20, 10, 5]

    # Argument management
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    args = parser.parse_args()

    count = 1

    loss = 'MFE'

    for dataset in DATASETS:

        if args.network == '-':
            model_architecture = DATASETS[dataset]['network']
        else:
            model_architecture = args.network
        for _round in range(1, N_ROUND + 1):

            args_list = [[str(reduction_ratio), dataset, 'dl-' + loss, loss, model_architecture, _round] for reduction_ratio in REDUCTION_RATIO]
            for arg_set in args_list:
                train_test(arg_set)

            print("(round {0}) Completed evaluating on {1} ({2}/{3})".format(_round, dataset, count, len(DATASETS) * N_ROUND))
            count += 1
