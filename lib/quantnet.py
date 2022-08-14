import torch, torchvision
import numpy as np
import pickle
from torch import nn
from typing import List, cast, Callable
from lib.math import min_pow_2

from lib.layer_tracker import HistogramTracker, Histogram
from lib.utils import QuantisableModule, get_module, set_module

def quantize_tensor(input: torch.Tensor, bit_width: int, scale: int) -> torch.Tensor:
    """ Fake quantize utility """
    return torch.fake_quantize_per_tensor_affine(input, scale, 0, -2**(bit_width-1), 2**(bit_width - 1) - 1)

def min_pow_2_scale(min_val: float, max_val: float, bit_width: int):
    """ Min power of 2 to represent """
    # this only works if this is true
    assert min_val <= 0
    assert max_val >= 0
    x = min_pow_2(min_val / (-2**(bit_width-1) - 1./2))
    y = min_pow_2(max_val / (2**(bit_width-1) - 1./2))
    return max(x, y)


def quantize_tensor_min_max(input: torch.Tensor, bit_width: int) -> torch.Tensor:
    return quantize_tensor_percentile(input, bit_width, 0., 1.)

def quantize_tensor_percentile(input: torch.Tensor, bit_width: int, lq: float, rq: float) -> torch.Tensor:
    #print(input.size())
    p = np.quantile(input.numpy(), [lq, rq])
    min_val, max_val = p[0], p[1]
    # print("quantile", min_val, max_val)
    # Minimum power of 2 required to represent all tensor values
    scale = 2**min_pow_2_scale(min_val.item(), max_val.item(), bit_width)
    #print("quant tensor:", scale)
    tensor = quantize_tensor(input, bit_width, scale)

    # Sanity check that quantization works properly
    m = torch.max(torch.abs(tensor - input))
    #print(tensor - input)
    #assert m < scale, f"{m}, {bit_width}"
    return tensor

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
        self.activation_bit_widths = activation_bit_widths
        self.weight_bit_widths = weight_bit_widths

def get_bin_for_percentile(cdf: np.ndarray, percentile: float, is_top: bool):
    bin_1 = len(cdf[cdf <= percentile]) - 1
    #print(bin_1)
    bin_2 = bin_1 + 1
    # is it closer to bin 1?
    bin = 0
    if bin_1 < 0:
        bin = bin_2
    else:
        assert cdf[bin_1] <= percentile <= cdf[bin_2]
        if percentile - cdf[bin_1] < cdf[bin_2] - percentile:
            bin = bin_1
        else:
            bin = bin_2

    if not is_top:
        bin += 1
    return bin


def decide_bounds_percentile(histogram: np.ndarray, tail: float):
    cdf = np.cumsum(histogram) / np.sum(histogram)
    #print(cdf)
    min_bin = get_bin_for_percentile(cdf, tail, False)
    max_bin = get_bin_for_percentile(cdf, 1. - tail, True)

    return (min_bin, max_bin)

def decide_bounds_min_max(histogram: np.ndarray):
    min_bin = np.nonzero(histogram)[0][0]
    max_bin = np.nonzero(histogram)[0][-1]

    return (min_bin, max_bin)

def get_intermediate_tracker(quant_net) -> list[HistogramTracker]:
    # Log all activations (outputs) of relevant layers
    output_layers = [quant_net.quantize] + [cast(VggUnit, unit).relu for unit in quant_net.features] \
            + [cast(FakeQuantize, quant_net.classifier[i]) for i in [2, 5, 7]]

    hist_tracker = [HistogramTracker() for i in range(len(output_layers))]

    def hist_tracker_hook(hist_tracker):
        def f(module, input, output):
            hist_tracker.update(output)
        return f

    for i, layer in enumerate(output_layers):
        layer.register_forward_hook(hist_tracker_hook(hist_tracker[i]))
    
    return hist_tracker

def assert_equal(a, b):
    assert a == b, f"{a} != {b}"

def setup_quant_net(net: QuantisableModule, activation_histograms: List[Histogram], quant_config: QuantConfig, \
        bounds_alg: Callable[[np.ndarray], tuple[int, int]], \
        weights_bounds_alg: Callable[[np.ndarray, int], int] \
        ) -> QuantisableModule:
    import copy
    quant_net = copy.deepcopy(net)

    start_layer_name, output_layers_names = quant_net.get_layers_to_track()
    layers_to_mutate_names = [start_layer_name] + output_layers_names
    assert_equal(len(layers_to_mutate_names), len(activation_histograms))#, "Unexpected number of histograms found"
    assert_equal(len(layers_to_mutate_names), len(quant_config.activation_bit_widths))

    w = weights_bounds_alg

    # quantise the weights of the layers.
    for i, layer_name in enumerate(quant_net.get_layers_to_quantise()):
        layer = get_module(quant_net.get_net(), layer_name)
        set_module(quant_net.get_net(), layer_name + ".weight", nn.Parameter(w(layer.weight, quant_config.weight_bit_widths[i][0])))
        set_module(quant_net.get_net(), layer_name + ".bias", nn.Parameter(w(layer.bias, quant_config.weight_bit_widths[i][1])))

    # MUST EXEC THIS BELOW CODE AFTER THE ABOVE LOOP
    # insert fake_quant layers (these layers are the activations)
    for i, (histogram, layer_name) in enumerate(zip(activation_histograms, layers_to_mutate_names)):
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

def test_quant(net: QuantisableModule, net_name: str, images: torch.utils.data.Dataset, quant_config_name: str, bounds_config_name: str):
    with open(f"output/outputhistogram_{net_name}.pkl", "rb") as f:
        activation_histograms = pickle.load(f)

    import tqdm
    configs = {'vgg11': {'8b': QuantConfig([8] * 12, [(8,8)] * 11), 
            '8b7b_fc_1': QuantConfig([8] * 9 + [7] * 3, [(8,8)] * 8 + [(7,7)] * 3),
            '8b6b_fc_1': QuantConfig([8] * 9 + [6] * 3, [(8,8)] * 8 + [(6,6)] * 3),
            '8b5b_fc_1': QuantConfig([8] * 9 + [5] * 3, [(8,8)] * 8 + [(5,5)] * 3),
            '8b4b_fc_1': QuantConfig([8] * 9 + [4] * 3, [(8,8)] * 8 + [(4,4)] * 3),
            #'6b4b_1': QuantConfig([6] * 3 + [4] * 9, [(6,6)] * 2 + [(4,4)] * 9),
            '7b': QuantConfig([7] * 12, [(7,7)] * 11),
            '6b': QuantConfig([6] * 12, [(6,6)] * 11),
            '4b': QuantConfig([4] * 12, [(4,4)] * 11),
            '5b': QuantConfig([5] * 12, [(5,5)] * 11)},
            'resnet18': {'8b': QuantConfig([16] + [8] * 25, [(8,8)] * 21),
            '6b': QuantConfig([6] * 26, [(6,6)] * 21),
            '4b': QuantConfig([4] * 26, [(4,4)] * 21)}}
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
    B_ = {'m': quantize_tensor_min_max, '3': qtp(1 / 1000), '4': qtp(1 / 10000), '5': qtp(1 / 100000), '6': qtp(1 / 1000000)}
    bounds_configs = merge_dicts([{k1 + '_' + k2: (v1, v2) for (k2, v2) in B_.items()} for (k1, v1) in A_.items()])

    if bounds_config_name not in bounds_configs:
        print("Invalid bounds configuration, try one of", bounds_configs.keys())
        return

    config = configs[net_name][quant_config_name]

    # RUN THE INFERENCE!
    with torch.no_grad():
        #config = QuantConfig([4] * 13, [(4,4)] * 11)
        #config = QuantConfig([6] * 13, [(6,6)] * 11)

        bound_algs = bounds_configs[bounds_config_name]

        quant_net = setup_quant_net(net, activation_histograms, config, bound_algs[0], bound_algs[1])
        #print(quant_net.get_net())
        all_preds = []

        # Need to track intermediate values during inference
        #trackers = get_intermediate_tracker(quant_net)

        loader = torch.utils.data.DataLoader(images, batch_size=10)
        # evaluate network
        with torch.no_grad():
            for X in tqdm.tqdm(loader):
                preds = quant_net.get_net()(X)
                # convert output to numpy
                preds_np = preds.cpu().detach().numpy()
                all_preds.append(preds_np)


        with open(f'output/quantpreds_{net_name}_{quant_config_name}_{bounds_config_name}.npy', 'wb') as f:
            concated = np.concatenate(all_preds)
            #print(concated.shape)
            np.save(f, concated)

        #histograms = [(tracker.range_pow_2, tracker.histogram.numpy()) for tracker in trackers]

        #with open(f"output/outputhistograms_{net_name}_{quant_config_name}_{bounds_config_name}.pkl", "wb") as output_file:
        #    pickle.dump(histograms, output_file, protocol=pickle.HIGHEST_PROTOCOL)
