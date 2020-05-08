from tensorflow import keras

N_ROUND = 5
EPOCHS = 500
BATCH_SIZE = 128
SEED = 10
ALPHA_RANGE = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
GAMMA_RANGE = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
EARLY_STOPPING = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=0,
    patience=10,
    mode='min',
    restore_best_weights=True)

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

DATASETS = [
    'optical_digits',
    'satimage',
    'pen_digits',
    'scene'
]