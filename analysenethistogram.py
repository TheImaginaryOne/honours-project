import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

import argparse

from lib.utils import get_net, get_module

parser = argparse.ArgumentParser("analyse-output-hist")
parser.add_argument('net_name')

subparsers = parser.add_subparsers(dest='choice')

activations_parser = subparsers.add_parser('activations')
weight_parser = subparsers.add_parser('weights')
activations_parser.add_argument('filename_prefix')
args = parser.parse_args()

plt.style.use('ggplot')
plt.rcParams["font.family"] = "Liberation Sans"

class GridPlot:
    """ Basically a wrapper for subplots """
    def __init__(self, count, width, **kwargs):
        self.width = width
        self.count = count
        self.height = (count - 1) // width + 1
        self.fig, self.ax = plt.subplots(self.height, self.width, **kwargs)
        # clear some axes
        for i in range(count, self.width * self.height):
            y, x = i % width, i // width
            self.ax[(x, y)].axis('off')

    def get_ax(self, i):
        y, x = i % self.width, i // self.width
        return self.ax[(x, y)]

    def get_fig(self):
        return self.fig

def plot_activations(filename):
    """Plot network activations"""

    print("Reading histograms")
    with open(f"output/{filename}.pkl", "rb") as f:
        histograms = pickle.load(f)

    grid_plot = GridPlot(len(histograms), 5, figsize=(18, 15), sharex=True, sharey=True)

    names = ["Input"] + [f"Conv Unit {i}" for i in range(8)] + [f"Fully Connected {i}" for i in range(3)]


    for i, (range_pow_2, hist) in enumerate(histograms):
        ax = grid_plot.get_ax(i)
        print(f"Plotting histogram {i}")

        partition = np.linspace(-2**range_pow_2, 2**range_pow_2, len(hist) + 1, endpoint=True)
        ax.bar(partition[:-1], hist, (partition[1:] - partition[:-1]), align="edge")
        ax.set_title(f'{names[i]} Output')
        ax.set_yscale('symlog')


    grid_plot.get_fig().savefig(f"output/{filename}.png")

def plot_weights(net_name):
    """Plot network weights"""
    net = get_net(net_name)
    net.get_net().eval()
    layer_names = net.get_layers_to_quantise()
    grid_plot = GridPlot(len(layer_names), 4, figsize=(15, 15), sharex=True, sharey=True)

    for i, layer_name in enumerate(layer_names):
        layer = get_module(net.get_net(), layer_name)
        ax = grid_plot.get_ax(i)
        ax.set_title(f'Layer {i}')
        ax.hist(layer.bias.detach().numpy().flatten(), bins=32, alpha=0.5, label='bias')
        ax.hist(layer.weight.detach().numpy().flatten(), bins=32, alpha=0.5, label='weight')
        ax.set_yscale('symlog')
        ax.legend()

    grid_plot.get_fig().savefig(f"output/{net_name}_weightshistogram.png")

def main():
    if args.choice == 'activations':
        plot_activations(args.filename_prefix), # args.net_name
    else:
        plot_weights(args.net_name)


main()
