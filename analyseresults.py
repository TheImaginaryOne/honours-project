import numpy as np
import pandas as pd
import argparse
from tabulate import tabulate
from lib.utils import CONFIG_SETS

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser("mobilenet")
parser.add_argument("labels_file_name", help="labels file", type=str)

subp = parser.add_subparsers(help="analyse all or one of the results", dest="command")

set_parser = subp.add_parser("set")
set_parser.add_argument("dir", help="predictions directory", type=str)
set_parser.add_argument("net_name", help="name of chosen neural net", type=str)
set_parser.add_argument("subset", help="subset of configs to show", type=str)

one_parser = subp.add_parser("one")
one_parser.add_argument("file_name", help="predictions file", type=str)

args = parser.parse_args()

def value_counts(x: np.ndarray):
    unique, counts = np.unique(x, return_counts=True)
    return np.asarray((unique, counts)).T

def compare():
    label_name = args.labels_file_name

    with open(label_name) as label_f:
        labels = np.array([int(s.split()[1]) for s in label_f.readlines()])

    # load prediction weights
    fname = args.file_name

    with open(fname, 'rb') as f:
        quantpreds = np.load(f)

    #print(value_counts(quantpreds))

    quant_pred_labels = quantpreds.argmax(axis=1)

    print("len labels:", len(labels), ", len predicted labels:", len(quant_pred_labels))
    print(labels, quant_pred_labels)
    print("top 1 error:", np.mean(labels != quant_pred_labels))

# 
def print_set():
    labels_file_name = args.labels_file_name
    # The labels of the images
    with open(labels_file_name) as label_f:
        labels = np.array([int(s.split()[1]) for s in label_f.readlines()])

    data = []

    # For each configuration, we get the results
    for (quant_config, bounds) in CONFIG_SETS[args.net_name][args.subset]:
        import os
        # load prediction weights
        # Note that the outputs are a numpy table
        fname = os.path.join(args.dir, f"quantpreds_{args.net_name}_{quant_config}_{bounds}.npy")

        with open(fname, 'rb') as f:
            quantpreds = np.load(f)

        quant_pred_labels = quantpreds.argmax(axis=1)

        # compute the accuracy
        data.append({"quant_config": quant_config, "bounds": bounds, "acc": np.mean(labels == quant_pred_labels)})

    dataframe = pd.DataFrame(data).pivot(columns="quant_config", index="bounds", values="acc")
    print(tabulate(dataframe, dataframe.columns.values, tablefmt="latex"))

if args.command=="one":
    compare()
elif args.command == "set":
    print_set()
else:
    print("The command is not valid!")
