import sys
sys.path.append("../")

import numpy as np
from dataset_tools.utils import get_splitted_data, get_mocked_imbalanced_data
from sklearn import preprocessing
import os
import argparse
import warnings
from config import result_path, BATCH_SIZE, EPOCHS, LOSS, EARLY_STOPPING, SEED
from eval_script.utils import save_results
from external_models.DeepLearning import image_network, utils
import tensorflow as tf

warnings.filterwarnings('ignore')

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)

def train_test(args_list):

    reduction_ratio, dataset_name, classification_algorithm, loss = args_list

    X_train, X_test, y_train, y_test = get_splitted_data(dataset_name, reduction_ratio)
    # X_train, X_test, y_train, y_test = get_mocked_imbalanced_data(n_samples = 100, n_features = 5, neg_ratio = 0.9)

    # log_dir = "gs://sky-movo/class_imbalance/cifar100_logs/fit/" + dataset_name + '/' + 'reduction_ratio_' + reduction_ratio + '/' + classification_algorithm
    # log_dir = "cifar100_logs/fit/" + dataset_name + '/' + 'reduction_ratio_' + reduction_ratio + '/' + classification_algorithm
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Model
    model = image_network.make_model(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), loss = LOSS[loss])

    initial_weight_path = os.path.join('model_initial_weights')

    if not os.path.isdir(initial_weight_path):
        os.mkdir(initial_weight_path)
        os.mkdir(os.path.join(initial_weight_path, 'image_network'))

        utils.make_initial_weights(model, os.path.join(initial_weight_path, 'image_network', 'initial_weights'))

    model.load_weights(os.path.join(initial_weight_path, 'image_network', 'initial_weights', 'initial_weights'))
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[EARLY_STOPPING],
        verbose=0)
    
    # Get predictions
    y_pred = model.predict_classes(X_test).T[0]

    check = np.all(np.array(y_pred) == 0)

    print(y_pred)
    print(check)

    if (np.all(np.array(y_pred) == 0)) or (np.all(np.array(y_pred) == 1)):
        train_test(args_list)

    # Save
    save_results(y_test, y_pred, classification_algorithm, dataset_name, reduction_ratio)
    # utils.save_model(model, classification_algorithm + '_' + dataset_name + '_' + reduction_ratio, dataset_name)
    # utils.save_history(history, classification_algorithm + '_' + dataset_name + '_' + reduction_ratio, dataset_name)


if __name__ == '__main__':

    # Argument management
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("reduction_ratio")
    parser.add_argument("loss")
    args = parser.parse_args()

    args_list = [str(args.reduction_ratio), args.dataset, 'dl-' + args.loss, args.loss]
    train_test(args_list)

    print("Completed evaluating on {0} at {1} ratio with {2} loss".format(args.dataset, args.reduction_ratio, args.loss))
