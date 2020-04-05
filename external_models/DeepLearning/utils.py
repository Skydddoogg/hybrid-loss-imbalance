import os
import pickle
import math
import sys
sys.path.append("../")

from config_path import result_path

def save_history(history, model_name, dataset_name):
    history.history['epoch'] = history.epoch
    pickle_history = open(os.path.join(result_path, dataset_name, 'model', "history_" + model_name + ".pickle"), "wb")
    pickle.dump(history.history, pickle_history)
    pickle_history.close()

def save_model(model, model_name, dataset_name):
    model.save(os.path.join(result_path, dataset_name, 'model', model_name + '.h5'))

def make_initial_weights(model, path):
    initial_weights = os.path.join(path, 'initial_weights')
    model.save_weights(initial_weights)

def lr_schedule(epoch):

    initial_lrate = 1e-3
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    print('Learning rate:', lrate)

    return lrate
