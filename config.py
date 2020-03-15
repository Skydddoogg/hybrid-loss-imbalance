from tensorflow import keras
from custom_functions import custom_loss
import os

# Constant
ITERATION = 5
BATCH_SIZE = 16
EPOCHS = 100
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
    monitor='val_auc', 
    verbose=0,
    patience=10,
    mode='max',
    restore_best_weights=True)
LR = 1e-3
LOSS = {
      'BCE': custom_loss.binary_crossentropy,
      # 'Balanced-BCE': custom_loss.balanced_binary_crossentropy,
      'MSE': custom_loss.mean_square_error,
      'MFE': custom_loss.mean_false_error,
      # 'Balanced-MFE': custom_loss.mean_squared_false_error,
      'FL': custom_loss.focal,
      # 'Balanced-FL': custom_loss.balanced_focal,
      'Hybrid-MFE-FL': custom_loss.hybrid_mfe_fl,
      # 'Balanced-Hybrid-MFE-FL': custom_loss.balanced_hybrid_mfe_fl
}
OPTIMIZER = keras.optimizers.Adam(lr=LR)

# Source
DATASETS = [
      'ecoli',
      'optical_digits',
      'satimage',
      'pen_digits',
      'abalone',
      'sick_euthyroid',
      'spectrometer',
      'car_eval_34',
      'isolet',
      'us_crime',
      'yeast_ml8',
      'scene',
      'libras_move',
      'thyroid_sick',
      # 'coil_2000',
      # 'arrhythmia',
      # 'solar_flare_m0',
      # 'oil',
      # 'car_eval_4',
      # 'wine_quality',
      # 'letter_img',
      # 'yeast_me2',
      # 'webpage',
      # 'ozone_level',
      # 'mammography',
      # 'protein_homo',
      # 'abalone_19',
]

# Path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = os.path.join(ROOT_DIR, 'dataset', 'train')
test_set_path = os.path.join(ROOT_DIR, 'dataset', 'test')
result_path = os.path.join(ROOT_DIR, 'results')
viz_path = os.path.join(ROOT_DIR, 'visualizations')
