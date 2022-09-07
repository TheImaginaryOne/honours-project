import torch, torchvision
import numpy as np
import pickle
import os
from torch import nn
from typing import Any, List, cast, Callable
from lib.math_utils import decide_bounds_min_max, decide_bounds_percentile, min_pow_2_scale, quantize_tensor_percentile, get_tensor_percentile

from lib.layer_tracker import HistogramTracker, Histogram
from lib.utils import get_module, iter_quantisable_modules_with_names, iter_trackable_modules, iter_trackable_modules_with_names, set_module
from lib.models import QuantisableModule

class FakeQuantize(nn.Module):
    def __init__(self, bit_width: int = 8, scale: int = 1):
        super().__init__()
        self.bit_width = bit_width
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        min_val, max_val = -2**(self.bit_width-1), 2**(self.bit_width - 1) - 1
        torch.round(input / self.scale, out=input)
        torch.clamp(input, min_val, max_val, out=input)
        input *= self.scale
        #print(self.scale, self.bit_width)
        return input


class QuantConfig:
    def __init__(self, activation_bit_widths: List[int], weight_bit_widths: List[tuple[int, int]]):
        # A list of integers, where the nth value denotes the number of bits on the nth quantisable layer
        # We have two configs for the activation and weight bit widths.
        self.activation_bit_widths = activation_bit_widths
        self.weight_bit_widths = weight_bit_widths


def assert_equal(a, b):
    assert a == b, f"{a} != {b}"

def setup_quant_net(net: QuantisableModule, activation_histograms: List[Histogram], quant_config: QuantConfig, \
        bounds_alg: Callable[[np.ndarray], tuple[int, int]], \
        weights_bounds_alg: Callable[[np.ndarray, int], int] \
        ) -> QuantisableModule:
    import copy
    quant_net = copy.deepcopy(net)

    trackable_modules = list(iter_trackable_modules_with_names(net.get_net()))
    # Add start module because we must insert a fakequantise before the first layer too!
    start = trackable_modules[0]
    trackable_modules = [start] + trackable_modules

    assert_equal(len(trackable_modules), len(activation_histograms))#, "Unexpected number of histograms found"
    # Check length of config bit widths are as expected
    assert_equal(len(trackable_modules), len(quant_config.activation_bit_widths))

    quantisable_layers = list(iter_quantisable_modules_with_names(net.get_net()))
    assert_equal(len(quantisable_layers), len(quant_config.weight_bit_widths))

    w = weights_bounds_alg

    # quantise the weights of the layers.
    for i, (layer_name, _) in enumerate(quantisable_layers):
        layer = get_module(quant_net.get_net(), layer_name)
        set_module(quant_net.get_net(), layer_name + ".weight", nn.Parameter(w(layer.weight, quant_config.weight_bit_widths[i][0])))
        set_module(quant_net.get_net(), layer_name + ".bias", nn.Parameter(w(layer.bias, quant_config.weight_bit_widths[i][1])))
    

    # MUST EXEC THIS BELOW CODE AFTER THE ABOVE LOOP
    # insert fake_quant layers (these layers are the activations)
    for i, (histogram, (layer_name, _)) in enumerate(zip(activation_histograms, trackable_modules)):
        min_bin, max_bin = bounds_alg(histogram.values)

        min_val = (min_bin - len(histogram.values) // 2) / len(histogram.values) * 2**(histogram.range_pow_2 + 1)
        max_val = (max_bin - len(histogram.values) // 2 + 1) / len(histogram.values) * 2**(histogram.range_pow_2 + 1)

        fake_quant = FakeQuantize()
        fake_quant.bit_width = quant_config.activation_bit_widths[i]
        scale = 2**min_pow_2_scale(min_val, max_val, quant_config.activation_bit_widths[i])
        fake_quant.scale = scale

        #print("quant acts", scale)

        layer = get_module(quant_net.get_net(), layer_name)
        # insert the quant layer
        if i > 0:
            set_module(quant_net.get_net(), layer_name, torch.nn.Sequential(layer, fake_quant))
        else:
            # the start layer
            set_module(quant_net.get_net(), layer_name, torch.nn.Sequential(fake_quant, layer))


    return quant_net

def merge_dicts(dict_list):
    result = {}
    for d in dict_list:
        result.update(d)
    return result

def test_quant(net: QuantisableModule, net_name: str, images: torch.utils.data.Dataset, quant_config_name: str, bounds_config_name: str, ignore_existing_file: bool):
    output_file_name = f'output/quantpreds_{net_name}_{quant_config_name}_{bounds_config_name}.npy'
    # Skip if the result file exists
    if ignore_existing_file and os.path.exists(output_file_name):
        print(f"{output_file_name} exists; skipping")
        return

    with open(f"output/outputhistogram_{net_name}.pkl", "rb") as f:
        activation_histograms = pickle.load(f)

    import tqdm
    configs = {'vgg11': {'8b': QuantConfig([8] * 12, [(8,8)] * 11), 
            '8b7b_fc': QuantConfig([8] * 9 + [7] * 3, [(8,8)] * 8 + [(7,7)] * 3),
            '8b6b_fc': QuantConfig([8] * 9 + [6] * 3, [(8,8)] * 8 + [(6,6)] * 3),
            '8b5b_fc': QuantConfig([8] * 9 + [5] * 3, [(8,8)] * 8 + [(5,5)] * 3),
            '8b4b_fc': QuantConfig([8] * 9 + [4] * 3, [(8,8)] * 8 + [(4,4)] * 3),
            #'6b4b_1': QuantConfig([6] * 3 + [4] * 9, [(6,6)] * 2 + [(4,4)] * 9),
            '7b': QuantConfig([7] * 12, [(7,7)] * 11),
            '6b': QuantConfig([6] * 12, [(6,6)] * 11),
            '4b': QuantConfig([4] * 12, [(4,4)] * 11),
            '5b': QuantConfig([5] * 12, [(5,5)] * 11)},
            'resnet18': {'8b': QuantConfig([8] * 41, [(8,8)] * 21),
            '6b': QuantConfig([6] * 41, [(6,6)] * 21),
            '4b': QuantConfig([4] * 41, [(4,4)] * 21),
            '8b6b': QuantConfig([8] * 21 + [6] * 21, [(8,8)] * 10 + [(6,6)] * 11), # 8 bits for first two blocks; 6 bits for next blocks.
            '8b4b': QuantConfig([8] * 21 + [4] * 21, [(8,8)] * 10 + [(4,4)] * 11) # 8 bits for first two blocks; 6 bits for next blocks.
            },
            'resnet34': {'8b': QuantConfig([8] * 42, [(8,8)] * 37),
            '6b': QuantConfig([6] * 42, [(6,6)] * 37),
            '4b': QuantConfig([4] * 42, [(4,4)] * 37),
            '8b6b': QuantConfig([8] * 19 + [6] * 23, [(8,8)] * 16 + [(6,6)] * 21), # 8 bits for first two blocks; 6 bits for next blocks.
            '8b4b': QuantConfig([8] * 19 + [4] * 23, [(8,8)] * 16 + [(4,4)] * 21) # 8 bits for first two blocks; 6 bits for next blocks.
            },
            }
    if quant_config_name not in configs[net_name]:
        print("Invalid configuration, try one of", configs.keys())
        return

    def qtp(tail):
        return lambda input, bit_width: quantize_tensor_percentile(input, bit_width, tail, 1 - tail)
    def dbp(tail):
        return lambda cdf: decide_bounds_percentile(cdf, tail)

    import itertools
    # The "bounds" algorithms for activations.
    # Basically, this is for deciding how to set the range for quantisation.
    # For example '3' means that the 99.99% percentile and 00.001% percentile of values are the minimum and maximum
    # (the values outside these ranges are ignored, and can be "clipped")
    A_ = {'m': decide_bounds_min_max, '3': dbp(1 / 1000), '4': dbp(1 / 10000), '5': dbp(1 / 100000), '6': dbp(1 / 1000000)}
    # The "bounds" algorithms for weights
    B_ = {'m': qtp(0.), '3': qtp(1 / 1000), '4': qtp(1 / 10000), '5': qtp(1 / 100000), '6': qtp(1 / 1000000)}
    bounds_configs = merge_dicts([{k1 + '_' + k2: (v1, v2) for (k2, v2) in B_.items()} for (k1, v1) in A_.items()])

    if bounds_config_name not in bounds_configs:
        print("Invalid bounds configuration, try one of", bounds_configs.keys())
        return

    config = configs[net_name][quant_config_name]

    # RUN THE INFERENCE!
    with torch.no_grad():

        bound_algs = bounds_configs[bounds_config_name]

        quant_net = setup_quant_net(net, activation_histograms, config, bound_algs[0], bound_algs[1])
        #print(quant_net.get_net())
        all_preds = []

        loader = torch.utils.data.DataLoader(images, batch_size=10)
        # evaluate network
        with torch.no_grad():
            for X in tqdm.tqdm(loader):
                preds = quant_net.get_net()(X)
                # convert output to numpy
                preds_np = preds.cpu().detach().numpy()
                all_preds.append(preds_np)


        with open(output_file_name, 'wb') as f:
            concated = np.concatenate(all_preds)
            #print(concated.shape)
            np.save(f, concated)


percentiles = {'1e-3': 1 / 1000, '1e-4': 1 / 10000, '1e-5': 1 / 100000, 'm': 0.}
def get_net_weight_bounds(net: QuantisableModule) -> List[Any]:
    """ 
    Calculate the bounds of the weights for different percentiles.
    Only for debug / visualisation purposes
    """
    quantisable_layers = list(iter_quantisable_modules_with_names(net.get_net()))

    bounds = []

    for i, (layer_name, layer) in enumerate(quantisable_layers):
        for name, perc in percentiles.items():
            weight_bounds = get_tensor_percentile(layer.weight.detach(), perc, 1 - perc)
            bounds.append((layer_name, name, 'weight', weight_bounds[0], weight_bounds[1]))

            bias_bounds = get_tensor_percentile(layer.bias.detach(), perc, 1 - perc)
            bounds.append((layer_name, name, 'bias', bias_bounds[0], bias_bounds[1]))
    
    return bounds

def get_net_activation_bounds(net_name: str, net: QuantisableModule) -> List[Any]:
    with open(f"output/outputhistogram_{net_name}.pkl", "rb") as f:
        activation_histograms = pickle.load(f)
    
    bounds = []

    trackable_modules = [('start', None)] + list(iter_trackable_modules_with_names(net.get_net()))

    for i, (histogram, (layer_name, _)) in enumerate(zip(activation_histograms, trackable_modules)):
        for name, perc in percentiles.items():
            if perc == 0.:
                min_bin, max_bin = decide_bounds_min_max(histogram.values)
            else:
                min_bin, max_bin = decide_bounds_percentile(histogram.values, perc)
    
            # TODO move
            min_val = (min_bin - len(histogram.values) // 2) / len(histogram.values) * 2**(histogram.range_pow_2 + 1)
            max_val = (max_bin - len(histogram.values) // 2 + 1) / len(histogram.values) * 2**(histogram.range_pow_2 + 1)

            bounds.append((layer_name, name, min_val, max_val))
    
    return bounds