import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

import argparse

from lib.utils import get_net

parse = argparse.ArgumentParser("analyse-output-hist")
parse.add_argument("choice", choices=['activations', 'weights'])
args = parse.parse_args()

plt.style.use('ggplot')
plt.rcParams["font.family"] = "Liberation Sans"

class GridPlot:
    """ Basically a wrapper for subplots """
    def __init__(self, count, width, **kwargs):
        self.width = width
        self.count = count
        self.height = (count - 1) // width + 1
        self.fig, self.ax = plt.subplots(self.width, self.height, **kwargs)
        # clear some axes
        for i in range(count, self.width * self.height):
            x, y = i % width, i // width
            self.ax[(x, y)].axis('off')

    def get_ax(self, i):
        x, y = i % self.width, i // self.width
        return self.ax[(x, y)]

    def get_fig(self):
        return self.fig

def plot_activations():
    """Plot network activations"""

    print("Reading histograms")
    with open("output/outputhistogram.pkl", "rb") as f:
        histograms = pickle.load(f)

    grid_plot = GridPlot(len(histograms), 5, figsize=(15, 15), sharex=True, sharey=True)

    names = ["Input"] + [f"Conv Unit {i}" for i in range(8)] + ['Average Pool'] + [f"Fully Connected {i}" for i in range(3)]


    for i, (range_pow_2, hist) in enumerate(histograms):
        ax = grid_plot.get_ax(i)
        print(f"Plotting histogram {i}")

        partition = np.linspace(-2**range_pow_2, 2**range_pow_2, len(hist) + 1, endpoint=True)
        ax.bar(partition[:-1], hist, partition[1:] - partition[:-1], align="edge")
        ax.set_title(f'{names[i]} Output')
        ax.set_yscale('symlog')


    grid_plot.get_fig().savefig("output/outputhistogram.png")

def plot_weights():
    """Plot network weights"""
    net = get_net()

    grid_plot = GridPlot(11, 4, figsize=(15, 15), sharex=True, sharey=True)

    cnt = 0
    i = 0
    j = 0
    with torch.no_grad():
        for layer in net.features:
            if isinstance(layer, torch.nn.Conv2d):
                ax = grid_plot.get_ax(cnt)
                ax.set_title(f'Conv {i}')
                ax.hist(layer.bias.numpy().flatten(), bins=32, alpha=0.5, label='bias')
                ax.hist(layer.weight.numpy().flatten(), bins=32, alpha=0.5, label='weight')
                ax.set_yscale('symlog')
                ax.legend()
                cnt += 1

                i += 1

        for layer in net.classifier:
            if isinstance(layer, torch.nn.Linear):
                ax = grid_plot.get_ax(cnt)
                ax.set_title(f'Fully Connected {j}')
                ax.hist(layer.bias.numpy().flatten(), bins=32, alpha=0.5, label='bias')
                ax.hist(layer.weight.numpy().flatten(), bins=32, alpha=0.5, label='weight')
                ax.set_yscale('symlog')
                ax.legend()
                cnt += 1
                j += 1

    grid_plot.get_fig().savefig("output/weightshistogram.png")

def main():
    if args.choice == 'activations':
        plot_activations()
    else:
        plot_weights()


main()
