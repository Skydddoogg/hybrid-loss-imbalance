import os
import pickle
import sys
sys.path.append("../")

from config import result_path

def save_history(history, model_name):
    history.history['epoch'] = history.epoch
    pickle_history = open(os.path.join(result_path, 'model', "history_" + model_name + ".pickle"), "wb")
    pickle.dump(history.history, pickle_history)
    pickle_history.close()

def save_model(model, model_name):
    model.save(os.path.join(result_path, 'model', model_name + '.h5'))