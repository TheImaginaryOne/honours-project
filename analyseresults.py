import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from lib.models import CONFIG_SETS

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser("mobilenet")
parser.add_argument("labels_file_name", help="labels file", type=str)

subp = parser.add_subparsers(help="analyse all or one of the results", dest="command")

set_parser = subp.add_parser("all")
set_parser.add_argument("net_name", help="name of chosen neural net", type=str)

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

    print(value_counts(quantpreds))

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

    dir = "output"

    # For each configuration, we get the results
    for quant_config in CONFIG_SETS[args.net_name]:
        import os
        # load prediction weights
        # Note that the outputs are a numpy table
        fname = os.path.join(dir, f"quantpreds_{args.net_name}_{quant_config}.npy")

        with open(fname, 'rb') as f:
            quantpreds = np.load(f)
            # in case there is a problem with the shape
            quantpreds = np.reshape(quantpreds, (-1, 1000))
        

        quant_pred_labels = quantpreds.argmax(axis=1)

        # compute the accuracy
        data.append({"quant_config": quant_config, "acc": np.mean(labels == quant_pred_labels)})

    dataframe = pd.DataFrame(data)


    # plot dataframe as a heatmap.

    # new_index = pd.DataFrame(new_index_list, index=dataframe.index)
    # numeric_cols = dataframe.columns
    # dataframe = pd.concat((new_index, dataframe), axis=1)
    # dataframe = pd.melt(dataframe, id_vars=['activations', 'weights'], var_name='bit_config')

    print(dataframe)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.barplot(data=dataframe, y='acc', x='quant_config', ax=ax)
#    for row in g.axes:
#        for ax in row:
#            for container in ax.containers:
#                ax.bar_label(container)
#                ax.set_xlim(0,1)
            #ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
            #     ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
            #     textcoords='offset points')

    plt.savefig(f'output/results_{args.net_name}_all_with_sizes.pdf')
    #print(tabulate(dataframe, dataframe.columns.values, tablefmt="latex", showindex=False))

if args.command=="one":
    compare()
elif args.command == "all":
    print_set()
else:
    print("The command is not valid!")
