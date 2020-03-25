import os
import pickle
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
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr