from tensorflow import keras
from custom_functions import custom_loss
import os

# Constant
ITERATION = 1
BATCH_SIZE = 16
EPOCHS = 500
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
    monitor='loss', 
    min_delta=1e-5,
    verbose=0,
    patience=5,
    mode='min',
    restore_best_weights=True)
LR = 1e-3
LOSS = {
      # 'BCE': custom_loss.CrossEntropy().binary_crossentropy,
      # 'Balanced-BCE': custom_loss.CrossEntropy().balanced_binary_crossentropy,
      # 'MSE': custom_loss.MeanSquareError().mean_square_error,
      'MFE': custom_loss.MeanFalseError().mean_false_error,
      'Balanced-MFE': custom_loss.MeanFalseError().mean_squared_false_error,
      'FL': custom_loss.Focal().focal,
      # 'Balanced-FL': custom_loss.Focal().balanced_focal,
      'Hybrid-MFE-FL': custom_loss.Hybrid().hybrid_mfe_fl,
      # 'Balanced-Hybrid-MFE-FL': custom_loss.Hybrid().balanced_hybrid_mfe_fl
}
OPTIMIZER = keras.optimizers.Adam(lr=LR)

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

# Path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = os.path.join(ROOT_DIR, 'dataset', 'train')
test_set_path = os.path.join(ROOT_DIR, 'dataset', 'test')
result_path = os.path.join(ROOT_DIR, 'results')
viz_path = os.path.join(ROOT_DIR, 'visualizations')
