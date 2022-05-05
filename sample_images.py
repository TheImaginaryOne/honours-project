import argparse
import os
import tqdm
import shutil

from collections import Counter

parser = argparse.ArgumentParser("mobilenet")
parser.add_argument("labels_file", help="labels_file", type=str)
parser.add_argument("output_labels_file", type=str)
args = parser.parse_args()

def run():
    label_name = args.labels_file

    with open(label_name) as label_f:
        labels_dict = {s.split()[0]: s.split()[1] for s in label_f.readlines()}

    # get a subset so there are N of each class label
    N = 5
    subset = []
    label_counter = Counter()
    for file_name, label in labels_dict.items():
        if label_counter[label] < N:
            label_counter[label] += 1
            subset.append((file_name, label))

    # sanity check
    for label, ctr in label_counter.items():
        assert N == ctr, f"{N}, {ctr}"

    with open(args.output_labels_file, "w") as f:
        for file_name, label in subset:
            f.write(f"{file_name} {label}\n")

run()
