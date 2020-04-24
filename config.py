from tensorflow import keras
from custom_functions import custom_loss
from external_models.DeepLearning.utils import lr_schedule

# Constant
N_ROUND = 1
BATCH_SIZE = 16
EPOCHS = 500
SEED = 10
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]
EARLY_STOPPING = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    min_delta=1e-6,
    verbose=0,
    patience=10,
    mode='min',
    restore_best_weights=True)
LOSS = {
      # 'BCE': custom_loss.CrossEntropy().binary_crossentropy,
      'Balanced-BCE': custom_loss.CrossEntropy().balanced_binary_crossentropy,
      # 'MSE': custom_loss.MeanSquareError().mean_square_error,
      'MFE': custom_loss.MeanFalseError().mean_false_error,
      'MSFE': custom_loss.MeanFalseError().mean_squared_false_error,
      'FL': custom_loss.Focal().focal,
      'Balanced-FL': custom_loss.Focal().balanced_focal,
      'Hybrid': custom_loss.Hybrid().hybrid,
      'Balanced-Hybrid': custom_loss.Hybrid().balanced_hybrid
}

# ALPHA_RANGE = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
# GAMMA_RANGE = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
ALPHA_RANGE = [0.25]
GAMMA_RANGE = [2.0]

# OPTIMIZER = keras.optimizers.Adam(lr=lr_schedule(0))
OPTIMIZER = keras.optimizers.Adam(lr=1e-6)

# Source
DATASETS = {
      'ecoli': {
            'n_features': 7,
            'n_samples': 336
      },
      'optical_digits': {
            'n_features': 64,
            'n_samples': 5620
      },
      'satimage': {
            'n_features': 36,
            'n_samples': 6435
      },
      # 'pen_digits': {
      #       'n_features': 16,
      #       'n_samples': 10992
      # },
      'abalone': {
            'n_features': 10,
            'n_samples': 4177
      },
      'sick_euthyroid': {
            'n_features': 42,
            'n_samples': 3163
      },
      'spectrometer': {
            'n_features': 93,
            'n_samples': 531
      },
      'car_eval_34': {
            'n_features': 21,
            'n_samples': 1728
      },
      'isolet': {
            'n_features': 617,
            'n_samples': 7797
      },
      'us_crime': {
            'n_features': 100,
            'n_samples': 1994
      },
      'yeast_ml8': {
            'n_features': 103,
            'n_samples': 2417
      },
      'scene': {
            'n_features': 294,
            'n_samples': 2407
      },
      'libras_move': {
            'n_features': 90,
            'n_samples': 360
      },
      'thyroid_sick': {
            'n_features': 52,
            'n_samples': 3772
      },
      'coil_2000': {
            'n_features': 85,
            'n_samples': 9822
      },
      'arrhythmia': {
            'n_features': 278,
            'n_samples': 452
      },
      'solar_flare_m0': {
            'n_features': 32,
            'n_samples': 1389
      },
      'oil': {
            'n_features': 49,
            'n_samples': 937
      },
      'car_eval_4': {
            'n_features': 21,
            'n_samples': 1728
      },
      'wine_quality': {
            'n_features': 11,
            'n_samples': 4898
      },
      # 'letter_img': {
      #       'n_features': 16,
      #       'n_samples': 20000
      # },
      'yeast_me2': {
            'n_features': 8,
            'n_samples': 1484
      },
      # 'webpage': {
      #       'n_features': 300,
      #       'n_samples': 34780
      # },
      'ozone_level': {
            'n_features': 72,
            'n_samples': 2536
      },
      # 'mammography': {
      #       'n_features': 6,
      #       'n_samples': 11183
      # },
      # 'protein_homo': {
      #       'n_features': 74,
      #       'n_samples': 145751
      # },
      'abalone_19': {
            'n_features': 10,
            'n_samples': 4177
      },
}
