import sys
sys.path.append("../")

import numpy as np
from dataset_tools.utils import get_splitted_data, get_mocked_imbalanced_data
from sklearn import preprocessing
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import warnings
from eval_script.config_imdb import BATCH_SIZE, EPOCHS, EARLY_STOPPING, METRICS, ALPHA_RANGE, GAMMA_RANGE, BUFFER_SIZE, max_features, IMB_LV, maxlen, N_ROUND
from config_path import result_path
from eval_script.utils import save_results, choose_network
from external_models.DeepLearning import utils, transformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from custom_functions import custom_loss
from sklearn.metrics import f1_score
import tensorflow_datasets as tfds

warnings.filterwarnings('ignore')

def train_test(args_list):

    dataset_name, classification_algorithm, loss, network, imb_ratio, encoder, _round = args_list

    X_train, X_test, X_valid, y_train, y_test, y_valid = get_splitted_data(dataset_name, imb_ratio, validation = True, seed = _round)

#     # Prepare callbacks
#     # log_dir = "gs://sky-movo/class_imbalance/cifar100_logs/fit/" + network + '/' + dataset_name + '/' + 'imb_ratio_' + imb_ratio + '/' + classification_algorithm
#     # log_dir = "cifar100_logs/fit/" + dataset_name + '/' + 'imb_ratio_' + imb_ratio + '/' + classification_algorithm
#     # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    # Model
    model = transformer.make_model(maxlen=maxlen, vocab_size=max_features)

    loss_function = custom_loss.MeanFalseError().mean_squared_false_error

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_function,
        metrics=METRICS)

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_valid, y_valid),
        callbacks=[EARLY_STOPPING],
        verbose=1)

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred = np.reshape(y_pred, (y_pred.shape[0],))

    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    y_pred_prob = model.predict(X_test)
    y_pred_prob = np.reshape(y_pred_prob, (y_pred_prob.shape[0],))

    # Save
    save_results(y_test, y_pred, y_pred_prob, 'round_' + str(_round) + '_' + classification_algorithm, dataset_name, imb_ratio, network)
    # utils.save_model(model, classification_algorithm + '_' + dataset_name + '_' + imb_ratio, dataset_name)
    utils.save_history(history, 'round_' + str(_round) + '_' + classification_algorithm + '_' + dataset_name + '_' + imb_ratio, dataset_name)



if __name__ == '__main__':

    dataset_name = 'imdb_reviews'
    network = 'transformer'
    loss = 'MSFE'
    classification_algorithm = 'dl-' + loss

    dataset, info = tfds.load(dataset_name + '/subwords8k', with_info=True, as_supervised=True)

    encoder = info.features['text'].encoder

    for _round in range(1, N_ROUND + 1):
        for imb_ratio in IMB_LV:
            train_test([dataset_name, classification_algorithm, loss, network, str(imb_ratio), encoder, _round])
