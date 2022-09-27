import argparse
import os
import tqdm
import shutil

from collections import Counter

parser = argparse.ArgumentParser("mobilenet")
parser.add_argument("labels_file", help="labels_file", type=str)
parser.add_argument("output_labels_file_test", type=str)
parser.add_argument("output_labels_file_val", type=str)
args = parser.parse_args()

def run():
    label_name = args.labels_file

    with open(label_name) as label_f:
        labels_dict = {s.split()[0]: s.split()[1] for s in label_f.readlines()}

    # get a subset so there are N of each class label
    N = 45
    # validation set
    N_VAL = 5
    subset_test = []
    subset_val = []
    label_counter = Counter()
    for file_name, label in labels_dict.items():
        if label_counter[label] < N:
            label_counter[label] += 1
            subset_test.append((file_name, label))
        elif label_counter[label] < N + N_VAL:
            label_counter[label] += 1
            subset_val.append((file_name, label))

    # sanity check
    for label, ctr in label_counter.items():
        expected = N + N_VAL
        assert expected == ctr, f"{expected}, {ctr}"

    with open(args.output_labels_file_test, "w") as f:
        for file_name, label in subset_test:
            f.write(f"{file_name} {label}\n")

    with open(args.output_labels_file_val, "w") as f:
        for file_name, label in subset_val:
            f.write(f"{file_name} {label}\n")

run()
