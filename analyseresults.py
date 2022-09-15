from typing import Optional
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import torch
from tabulate import tabulate
from lib.models import CONFIG_SETS, get_net
from lib.net_ops import profile_net_bit_ops
from lib.net_stats import quant_model_size

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

def scatter_plot(dataframe: pd.DataFrame, x: str, y: str, label: str, ax, hue: Optional[str] = None):
    if hue != None:
        p = sns.scatterplot(data=dataframe, x=x, y=y, ax=ax, hue=hue)
    else:
        p = sns.scatterplot(data=dataframe, x=x, y=y, ax=ax)

    texts = []

    for line in range(0, dataframe.shape[0]):
        t = p.text(dataframe[x][line], dataframe[y][line], 
        dataframe[label][line], horizontalalignment='left', 
        size='small', color='black')
        texts.append(t)
    
    # https://stackoverflow.com/questions/19073683/how-to-fix-overlapping-annotations-text/34762716#34762716
    adjust_text(texts, force_points=0.3, arrowprops=dict(arrowstyle='->', color='b', lw=0.5))

def print_set():
    labels_file_name = args.labels_file_name
    # The labels of the images
    with open(labels_file_name) as label_f:
        labels = np.array([int(s.split()[1]) for s in label_f.readlines()])

    data = []

    dir = "output"

    net = get_net(args.net_name)

    # For each configuration, we get the results
    for quant_config_name, quant_config in CONFIG_SETS[args.net_name].items():
        import os
        # load prediction weights
        # Note that the outputs are a numpy table
        fname = os.path.join(dir, f"quantpreds_{args.net_name}_{quant_config_name}.npy")

        with open(fname, 'rb') as f:
            quantpreds = np.load(f)
            # in case there is a problem with the shape
            quantpreds = np.reshape(quantpreds, (-1, 1000))

        quant_pred_labels = quantpreds.argmax(axis=1)

        model_size = quant_model_size(net, quant_config)
        
        model_bit_ops = profile_net_bit_ops(net, quant_config, (1, 3, 224, 224))

        # compute the accuracy
        data.append({"quant_config": quant_config_name, "acc": np.mean(labels == quant_pred_labels), "model_size": model_size, "model_bit_ops": model_bit_ops})

    dataframe = pd.DataFrame(data)


    # plot dataframe as a heatmap.

    # new_index = pd.DataFrame(new_index_list, index=dataframe.index)
    # numeric_cols = dataframe.columns
    # dataframe = pd.concat((new_index, dataframe), axis=1)
    # dataframe = pd.melt(dataframe, id_vars=['activations', 'weights'], var_name='bit_config')

    dataframe['label'] = dataframe['quant_config'] + dataframe['acc'].map(lambda acc: f"; {acc}")

    print(dataframe)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    scatter_plot(dataframe, 'model_size', 'acc', 'quant_config', ax[0])
    scatter_plot(dataframe, 'model_bit_ops', 'acc', 'quant_config', ax[1])
    fig.tight_layout()
    fig.savefig(f'output/results_{args.net_name}_all_with_size_and_ops.pdf')

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    scatter_plot(dataframe, 'model_bit_ops', 'model_size', 'label', ax, 'acc')
    fig.tight_layout()
    fig.savefig(f'output/{args.net_name}_accuracy_and_size_and_ops.pdf')

plt.style.use('ggplot')

if args.command=="one":
    compare()
elif args.command == "all":
    print_set()
else:
    print("The command is not valid!")
