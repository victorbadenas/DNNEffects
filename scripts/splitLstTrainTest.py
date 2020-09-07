import pandas as pd
import argparse
from pathlib import Path
import random
random.seed(1)

def split_file(arguments):
    full_lst = load_csv(arguments.file_path)
    train_file_path = arguments.file_path.parent / arguments.file_path.stem / "train.lst"
    test_file_path = arguments.file_path.parent / arguments.file_path.stem / "test.lst"
    train_file_path.parent.mkdir(exist_ok=True, parents=True)
    num_test_files = int(arguments.test_percentage * len(full_lst))
    all_idx = set(range(len(full_lst)))
    test_rows_indexes = random.sample(all_idx, num_test_files)
    train_rows_idx = list(all_idx - set(test_rows_indexes))
    train_df = full_lst.iloc[train_rows_idx].sort_index()
    test_df = full_lst.iloc[test_rows_indexes].sort_index()
    train_df.to_csv(train_file_path, sep='\t')
    test_df.to_csv(test_file_path, sep='\t')

def load_csv(file):
    return pd.read_csv(file, sep='\t', index_col=0)

def parse_arguments_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--file_path", type=Path, required=True)
    parser.add_argument("-test", "--test_percentage", type=float, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments_from_command_line()
    split_file(args)