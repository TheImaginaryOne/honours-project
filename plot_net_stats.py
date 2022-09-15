import argparse
from typing import Collection, List, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torchsummary
from lib.quantnet import get_net_activation_bounds, get_net_weight_bounds

from lib.models import get_net
from lib.utils import iter_quantisable_modules_with_names

from collections import OrderedDict

parser = argparse.ArgumentParser("quant-net")
parser.add_argument("net_name", help="the net to test.", type=str)

args = parser.parse_args()

def plot_bounds_chart(bounds: pd.DataFrame, ax):
    from matplotlib import cm
    # === plot
    cmap = cm.coolwarm

    x_axis = list(OrderedDict.fromkeys(bounds['layer_name']))

    cats = list(OrderedDict.fromkeys(bounds['bounds']))
    cats_indices = {k: i for i, k in enumerate(x_axis)}

    ax.set_yticks(list(2. ** np.arange(-2, 8)) + list(-(2. ** np.arange(-2, 8))))
    ax.set_ylim(np.min(bounds['lower']) * 1.2, np.max(bounds['upper']) * 1.2)

    ax.set_xlim(-1, len(x_axis))
    ax.set_xticks(np.arange(0, len(x_axis)))
    ax.set_xticklabels(x_axis, rotation=90)

    line_segs = {k: [] for k in cats}

    for _, row in bounds.iterrows():
        x_pos = cats_indices[row['layer_name']]
        line_segs[row['bounds']].append([(x_pos, row['lower']), (x_pos, row['upper'])])
    
    line_collections = []
    # Reversed, because we want to see all values
    for i, (_, line_seg) in enumerate(reversed(list(line_segs.items()))):
        ln = LineCollection(line_seg, colors=cmap(i / (len(cats)-1)), lw=6)
        line_collections.append(ln)
        ax.add_collection(ln)
    
    ax.legend(line_collections, reversed(cats), bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

def plot_weight_bounds(net):
    bounds = get_net_weight_bounds(net)

    bounds = pd.DataFrame(bounds, columns=['layer_name', 'bounds', 'type', 'lower', 'upper'])
    print(bounds)

    #g = sns.catplot(data=weight_bounds, kind="bar", row="type", x="layer_name", y="upper", hue="bounds", height=5, aspect=2)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    plot_bounds_chart(bounds[bounds['type'] == 'weight'], ax[0])
    plot_bounds_chart(bounds[bounds['type'] == 'bias'], ax[1])

    fig.tight_layout()
    fig.savefig(f"output/{args.net_name}_bounds.png")

def plot_act_bounds(net_name, net):
    bounds = get_net_activation_bounds(args.net_name, net)

    bounds = pd.DataFrame(bounds, columns=['layer_name', 'bounds', 'lower', 'upper'])
    print(bounds)

    fig, ax = plt.subplots(figsize=(10, 7))
    #sns.lineplot(data=bounds, x="layer_name", x="upper", hue="bounds", ax=ax)#, log=True)
    plot_bounds_chart(bounds, ax)
    fig.tight_layout()

    fig.savefig(f"output/{args.net_name}_act_bounds.png")

import functools
def product_of_tuple(l: Tuple[int]):
    return functools.reduce(lambda a, b: a * b, l)

def main(args):
    plt.style.use('ggplot')

    net = get_net(args.net_name)

    quantisable_layers = iter_quantisable_modules_with_names(net.get_net())

    weight_counts = []

    for layer_name, layer in quantisable_layers:
        weight_bounds = layer.weight.detach()
        bias_bounds = layer.bias.detach()
        weight_counts.append({'layer_name': layer_name, 'weight_size': product_of_tuple(weight_bounds.shape), 'bias_size': product_of_tuple(bias_bounds.shape)})

    weight_counts = pd.DataFrame(weight_counts)
    weight_counts.to_csv(f"output/{args.net_name}_sizes.csv")

    # == Plot weights
    plot_weight_bounds(net)

    # == Plot activation histograms
    plot_act_bounds(args.net_name, net)


main(args)
