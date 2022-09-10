import torch, torchvision
import numpy as np
import pickle
import os
from torch import nn
from typing import Any, List, cast, Callable
from lib.math_utils import min_pow_2_scale, quantize_tensor_percentile, get_tensor_percentile

from lib.layer_tracker import HistogramTracker, Histogram, decide_bounds_min_max, decide_bounds_percentile
from lib.utils import get_module, iter_quantisable_modules_with_names, iter_trackable_modules, iter_trackable_modules_with_names, set_module
from lib.models import CONFIG_SETS, QuantConfig, QuantisableModule

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

def eval_results(preds: np.ndarray, labels: np.ndarray):
    """ Evaluate prediction score, given some predictions. """

    pred_labels = preds.argmax(axis=1)

    return np.mean(labels == pred_labels)

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
        min_val, max_val = bounds_alg(histogram)

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

def run_net(net: QuantisableModule, loader: torch.utils.data.DataLoader):
    """ Run net, get predictions. """
    all_preds = []
    labels = []

    import tqdm
    # evaluate network
    with torch.no_grad():
        for X, label in tqdm.tqdm(loader):
            preds = net.get_net()(X)
            # convert output to numpy
            preds_np = preds.cpu().detach().numpy()
            all_preds.append(preds_np)
            labels.append(label)
    
    return np.concatenate(all_preds), np.concatenate(labels)

def test_quant(net: QuantisableModule, net_name: str, images: torch.utils.data.Dataset, val_images: torch.utils.data.Dataset, quant_config_name: str, ignore_existing_file: bool):
    output_file_name = f'output/quantpreds_{net_name}_{quant_config_name}.npy'
    # Skip if the result file exists
    if ignore_existing_file and os.path.exists(output_file_name):
        print(f"{output_file_name} exists; skipping")
        return

    with open(f"output/outputhistogram_{net_name}.pkl", "rb") as f:
        activation_histograms = pickle.load(f)

    configs = CONFIG_SETS

    import tqdm
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
    AB = {'3': dbp(1 / 1000), '4': dbp(1 / 10000), '5': dbp(1 / 100000)}
    # The "bounds" algorithms for weights
    WB = {'3': qtp(1 / 1000), '4': qtp(1 / 10000), '5': qtp(1 / 100000)}

    quant_config = configs[net_name][quant_config_name]

    # RUN THE INFERENCE!
    with torch.no_grad():

        val_scores = {}

        print("Running validation / calibration loop")
        # validation loop.
        for k1, ab in AB.items():
            for k2, wb in WB.items():
                loader = torch.utils.data.DataLoader(val_images, batch_size=10)
                candidate_net = setup_quant_net(net, activation_histograms, quant_config, ab, wb)
                preds, labels = run_net(candidate_net, loader)

                score = eval_results(preds, labels)
                key = (k1, k2)
                print(f"Score for {key}: {score}")
                val_scores[key] = score

                # save for debugging
                val_output_file_name = f'output/quantpreds_{net_name}_{quant_config_name}_val_{k1}_{k2}.npy'
                with open(val_output_file_name, 'wb') as f:
                    np.save(f, preds)
        
        # basically argmax of validation scores.
        best_bounds_alg = max(val_scores, key=val_scores.get)
        print(f"Using: {best_bounds_alg}")
        print("Testing...")
        ab, wb = AB[best_bounds_alg[0]], WB[best_bounds_alg[1]]

        # TEST!!!
        loader = torch.utils.data.DataLoader(images, batch_size=10)
        quant_net = setup_quant_net(net, activation_histograms, quant_config, ab, wb)
        preds, labels = run_net(quant_net, loader)

        score = eval_results(preds, labels)
        print(f"-- Final score: {score}")

        with open(output_file_name, 'wb') as f:
            #print(concated.shape)
            np.save(f, preds)


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
                min_val, max_val = decide_bounds_min_max(histogram)
            else:
                min_val, max_val = decide_bounds_percentile(histogram, perc)

            bounds.append((layer_name, name, min_val, max_val))
    
    return bounds