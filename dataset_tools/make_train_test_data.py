from utils import split_train_test
import argparse
from config import ITERATION

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    args = parser.parse_args()

    split_train_test(args.dataset_name, ITERATION)