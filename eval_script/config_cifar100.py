from tensorflow import keras

BATCH_SIZE = 128
EPOCHS = 500
EARLY_STOPPING = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=0,
    patience=10,
    mode='min',
    restore_best_weights=True)
SEED = 10
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.Accuracy('accuracy')
]
N_ROUND = 5
ALPHA_RANGE = [0.25]
GAMMA_RANGE = [2.0]