import argparse
from typing import Collection, List, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lib.models import get_net

from lib.net_ops import profile_net_mac_ops
from lib.net_stats import get_model_layer_sizes
from lib.utils import get_readable_layer_names

parser = argparse.ArgumentParser("quant-net")
parser.add_argument("net_name", help="the net to test.", type=str)

args = parser.parse_args()

def main(args):
    plt.style.use('ggplot')

    net = get_net(args.net_name)

    mac_ops = profile_net_mac_ops(net, (1, 3, 224, 224))
    mac_ops = pd.DataFrame(mac_ops)

    sizes = get_model_layer_sizes(net)
    sizes = pd.DataFrame(sizes)

    print(mac_ops, sizes)

    df = mac_ops.merge(sizes, on='layer_name', how='outer')
    df['layer_name'] = get_readable_layer_names(args.net_name, df['layer_name'])
    print(df)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.barplot(data=df, x="layer_name", y="mac_ops", ax=ax[0])
    sns.barplot(data=df, x="layer_name", y="weight_size", ax=ax[1])

    for i in [0,1]:
        ax[i].tick_params(axis='x', rotation=90)

    fig.tight_layout()

    fig.savefig(f'output/{args.net_name}_size_and_ops.pdf')



main(args)
