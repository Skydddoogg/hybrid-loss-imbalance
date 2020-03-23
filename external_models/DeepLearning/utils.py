import os
import pickle
import sys
sys.path.append("../")

from config import result_path

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