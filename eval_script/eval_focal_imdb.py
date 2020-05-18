import sys
sys.path.append("../")

import numpy as np
from dataset_tools.utils import get_splitted_data, get_mocked_imbalanced_data
from sklearn import preprocessing
import os
import argparse
import warnings
from eval_script.config_imdb import BATCH_SIZE, EPOCHS, EARLY_STOPPING, METRICS, ALPHA_RANGE, GAMMA_RANGE, BUFFER_SIZE, max_features
from config_path import result_path
from eval_script.utils import save_results, choose_network
from external_models.DeepLearning import lstm, utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from custom_functions import custom_loss
from sklearn.metrics import f1_score
import tensorflow_datasets as tfds

warnings.filterwarnings('ignore')

def train_test(args_list):

    dataset_name, classification_algorithm, loss, network, imb_ratio, encoder = args_list

    X_train, X_test, y_train, y_test = get_splitted_data(dataset_name, imb_ratio)

#     # Prepare callbacks
#     # log_dir = "gs://sky-movo/class_imbalance/cifar100_logs/fit/" + network + '/' + dataset_name + '/' + 'imb_ratio_' + imb_ratio + '/' + classification_algorithm
#     # log_dir = "cifar100_logs/fit/" + dataset_name + '/' + 'imb_ratio_' + imb_ratio + '/' + classification_algorithm
#     # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    initial_learning_rate = 0.1
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    # Model
    model = lstm.make_model(max_features)

    comparable_f1 = -np.Infinity
    count = 1

    for alpha in ALPHA_RANGE:
        for gamma in GAMMA_RANGE:

            loss_function = custom_loss.Focal(gamma=gamma, alpha=alpha).balanced_focal

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
                validation_data=(X_test, y_test),
                callbacks=[EARLY_STOPPING],
                verbose=1)

            # Get predictions
            y_pred = model.predict(X_test)
            y_pred = np.reshape(y_pred, (y_pred.shape[0],))

            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0

            y_pred_prob = model.predict(X_test)
            y_pred_prob = np.reshape(y_pred_prob, (y_pred_prob.shape[0],))

            f1 = f1_score(y_test, y_pred)
            if f1 >= comparable_f1:

                # Save
                save_results(y_test, y_pred, y_pred_prob, classification_algorithm, dataset_name, imb_ratio, network)
                # utils.save_model(model, classification_algorithm + '_' + dataset_name + '_' + imb_ratio, dataset_name)
                utils.save_history(history, classification_algorithm + '_' + dataset_name + '_' + imb_ratio, dataset_name)

                comparable_f1 = f1
            print("{0}/{1}".format(count, len(ALPHA_RANGE)*len(GAMMA_RANGE)))
            count += 1


if __name__ == '__main__':

    dataset_name = 'imdb_reviews'
    network = 'lstm'
    imb_ratio = '10'
    loss = 'FL'
    classification_algorithm = 'dl-' + loss

    dataset, info = tfds.load(dataset_name + '/subwords8k', with_info=True, as_supervised=True)

    encoder = info.features['text'].encoder

    train_test([dataset_name, classification_algorithm, loss, network, imb_ratio, encoder])
