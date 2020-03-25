import os

# Path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_set_path = os.path.join(ROOT_DIR, 'dataset', 'train')
test_set_path = os.path.join(ROOT_DIR, 'dataset', 'test')
result_path = os.path.join(ROOT_DIR, 'results')
viz_path = os.path.join(ROOT_DIR, 'visualizations')