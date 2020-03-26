import sys
sys.path.append("../")

import numpy as np
from dataset_tools.utils import get_splitted_data, get_mocked_imbalanced_data
from sklearn import preprocessing
import os
import argparse
import warnings
from config import BATCH_SIZE, EPOCHS, LOSS, EARLY_STOPPING, SEED
from config_path import result_path
from eval_script.utils import save_results
from external_models.DeepLearning import resnetV2, utils
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

warnings.filterwarnings('ignore')

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)

def train_test(args_list):

    reduction_ratio, dataset_name, classification_algorithm, loss = args_list

    X_train, X_test, X_valid, y_train, y_test, y_valid = get_splitted_data(dataset_name, reduction_ratio, validation = True)
    # X_train, X_test, y_train, y_test = get_mocked_imbalanced_data(n_samples = 100, n_features = 5, neg_ratio = 0.9)

    # Prepare callbacks
    log_dir = "gs://sky-movo/class_imbalance/cifar100_logs/fit/" + dataset_name + '/' + 'reduction_ratio_' + reduction_ratio + '/' + classification_algorithm
    # log_dir = "cifar100_logs/fit/" + dataset_name + '/' + 'reduction_ratio_' + reduction_ratio + '/' + classification_algorithm
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    lr_scheduler = LearningRateScheduler(utils.lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)

    # Model
    model = resnetV2.make_model(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), loss = LOSS[loss])

    initial_weight_path = os.path.join('model_initial_weights')

    if not os.path.isdir(initial_weight_path):
        os.mkdir(initial_weight_path)
        os.mkdir(os.path.join(initial_weight_path, 'resnetV2'))

        utils.make_initial_weights(model, os.path.join(initial_weight_path, 'resnetV2', 'initial_weights'))

    model.load_weights(os.path.join(initial_weight_path, 'resnetV2', 'initial_weights', 'initial_weights'))
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_valid, y_valid),
        callbacks=[EARLY_STOPPING, lr_scheduler, lr_reducer, tensorboard_callback],
        verbose=1)
    
    # Get predictions
    # y_pred = model.predict_classes(X_test).T[0]
    y_pred = model.predict(X_test)
    y_pred = np.reshape(y_pred, (y_pred.shape[0],))

    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    print(y_pred)

    # if (np.all(np.array(y_pred) == 0)) or (np.all(np.array(y_pred) == 1)):
    #     print("Got the fucking result...")
    #     train_test(args_list)

    # Save
    save_results(y_test, y_pred, classification_algorithm, dataset_name, reduction_ratio)
    # utils.save_model(model, classification_algorithm + '_' + dataset_name + '_' + reduction_ratio, dataset_name)
    # utils.save_history(history, classification_algorithm + '_' + dataset_name + '_' + reduction_ratio, dataset_name)


if __name__ == '__main__':

    DATASETS = ['Tree1_cifar100', 'Tree2_cifar100']
    REDUCTION_RATIO = [20, 10, 5]

    count = 1

    for dataset in DATASETS:

        for loss in LOSS:

            args_list = [[str(reduction_ratio), dataset, 'dl-' + loss, loss] for reduction_ratio in REDUCTION_RATIO]
            for args in args_list:
                print('.', end='')
                train_test(args)

        print('')
        print("Completed evaluating on {0} ({1}/{2})".format(dataset, count, len(DATASETS)))
        count += 1
