import torch, torchvision
import numpy as np
from torch import nn
from typing import List, cast, Callable
from lib.math import min_pow_2

from lib.layer_tracker import HistogramTracker

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
    # Minimum power of 2 required to represent all tensor values
    scale = 2**min_pow_2_scale(min_val.item(), max_val.item(), bit_width)
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

class VggUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, max_pool: bool):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.quantize = FakeQuantize()
        self.relu = nn.ReLU(inplace=True)
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.max_pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        x = self.quantize(x)
        x = self.relu(x)
        if self.max_pool is not None:
            x = self.max_pool(x)
        return x


class QuantisedVgg(nn.Module):
    def __init__(self, features: nn.Sequential):
        super().__init__()
        num_classes = 1000
        self.quantize = FakeQuantize()
        self.features = features
        #self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)), FakeQuantize())
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            FakeQuantize(),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            FakeQuantize(),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
            FakeQuantize(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.quantize(input)
        x = self.features(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Based on pytorch code
def make_layers(cfg: List[tuple[int, bool]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for out_channels, max_pool in cfg:
        layers.append(VggUnit(in_channels, out_channels, max_pool))
        in_channels = out_channels
    return nn.Sequential(*layers)

def make_vgg11() -> QuantisedVgg:
    layers: List[tuple[int, bool]] = [
            (64, True), 
            (128, True), 
            (256, False),
            (256, True),
            (512, False),
            (512, True),
            (512, False),
            (512, True)]
    return QuantisedVgg(make_layers(layers))

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

def setup_quant_net(net: torchvision.models.vgg.VGG, activation_histograms: List[tuple[int, np.ndarray]], quant_config: QuantConfig, \
        bounds_alg: Callable[[np.ndarray], tuple[int, int]], \
        weights_bounds_alg: Callable[[np.ndarray], int] \
        ) -> QuantisedVgg:

    quant_net = make_vgg11()
    from torchsummary import summary
    #summary(quant_net, input_size=(3, 225, 225))

    fake_quants = [quant_net.quantize] + [cast(VggUnit, unit).quantize for unit in quant_net.features] \
            + [cast(FakeQuantize, quant_net.classifier[i]) for i in [1, 4, 7]]
            #+ [cast(FakeQuantize, quant_net.avgpool[1])] \

    i = 0
    # set fake_quant layers
    for (pow_2, histogram), fake_quant in zip(activation_histograms, fake_quants):

        min_bin, max_bin = bounds_alg(histogram)

        #max_bin_100 = np.nonzero(histogram)[0][-1]
        #print(pow_2)
        #print(np.nonzero(histogram)[0][-1])
        min_val = (min_bin - len(histogram) // 2) / len(histogram) * 2**(pow_2 + 1)
        max_val = (max_bin - len(histogram) // 2 + 1) / len(histogram) * 2**(pow_2 + 1)
        #print("--", min_val, max_val)

        fake_quant.bit_width = quant_config.activation_bit_widths[i]
        scale = 2**min_pow_2_scale(min_val, max_val, quant_config.activation_bit_widths[i])
        fake_quant.scale = scale
        #print("-", pow_2)
        i += 1

    output_feature_layers = [net.features[i] for i in [0, 3, 6, 8, 11, 13, 16, 18]]
    classifier_layers = [net.classifier[i] for i in [0, 3, 6]]

    w = weights_bounds_alg
    # copy weights
    for i, layer in enumerate(output_feature_layers):
        quant_net.features[i].conv2d.weight = nn.Parameter(w(layer.weight, quant_config.weight_bit_widths[i][0]))
        quant_net.features[i].conv2d.bias = nn.Parameter(w(layer.bias, quant_config.weight_bit_widths[i][1]))
    for i, layer, j in zip([0, 3, 6], classifier_layers, range(len(output_feature_layers), len(output_feature_layers) + 3)):
        # just an offset
        quant_net.classifier[i].weight = nn.Parameter(w(layer.weight, quant_config.weight_bit_widths[j][0]))
        quant_net.classifier[i].bias = nn.Parameter(w(layer.bias, quant_config.weight_bit_widths[j][1]))

    return quant_net

def merge_dicts(dict_list):
    result = {}
    for d in dict_list:
        result.update(d)
    return result

def test_quant(net: torchvision.models.vgg.VGG, activation_histograms: List[tuple[int, np.ndarray]], images: torch.utils.data.Dataset, quant_config_name: str, bounds_config_name: str):
    import tqdm
    configs = {'8b': QuantConfig([8] * 12, [(8,8)] * 11), 
            #'8b6b_0': QuantConfig([8] * 1 + [6] * 11, [(8,8)] * 0 + [(6,6)] * 11),
            '8b6b_1': QuantConfig([8] * 3 + [6] * 9, [(8,8)] * 2 + [(6,6)] * 9),
            '8b6b_2': QuantConfig([8] * 5 + [6] * 7, [(8,8)] * 4 + [(6,6)] * 7),
            '8b6b_3': QuantConfig([8] * 7 + [6] * 5, [(8,8)] * 6 + [(6,6)] * 5),
            '8b7b_fc_1': QuantConfig([8] * 9 + [7] * 3, [(8,8)] * 8 + [(7,7)] * 3),
            '8b6b_fc_1': QuantConfig([8] * 9 + [6] * 3, [(8,8)] * 8 + [(6,6)] * 3),
            '8b5b_fc_1': QuantConfig([8] * 9 + [5] * 3, [(8,8)] * 8 + [(5,5)] * 3),
            '8b4b_fc_1': QuantConfig([8] * 9 + [4] * 3, [(8,8)] * 8 + [(4,4)] * 3),
            #'6b4b_1': QuantConfig([6] * 3 + [4] * 9, [(6,6)] * 2 + [(4,4)] * 9),
            '7b': QuantConfig([7] * 12, [(7,7)] * 11),
            '6b': QuantConfig([6] * 12, [(6,6)] * 11),
            '4b': QuantConfig([4] * 12, [(4,4)] * 11),
            '5b': QuantConfig([5] * 12, [(5,5)] * 11)}
    if quant_config_name not in configs:
        print("Invalid configuration, try one of", configs.keys())
        return

    def qtp(tail):
        return lambda input, bit_width: quantize_tensor_percentile(input, bit_width, tail, 1 - tail)
    def dbp(tail):
        return lambda cdf: decide_bounds_percentile(cdf, tail)

    # The first one configures the bounds of the activations;
    # The second one configures the bounds of the weights.
    #bounds_configs = {'minmax': (decide_bounds_min_max, quantize_tensor_min_max), 
    #        'p_m_99.9': (decide_bounds_min_max, qtp(1 / 1000)),
    #        'p_m_99.99': (decide_bounds_min_max, qtp(1 / 10000)),
    #        'p_m_99.999': (decide_bounds_min_max, qtp(1 / 100000)),
    #        'p_99.9_m': 
            #'percent_1_2^17': (dbp(1 / 131072), qtp(1 / 131072)),
            #'percent_1_2^16': (dbp(1 / 65536), qtp(1 / 65536)),
            #'percent_1_2^15': (dbp(1 / 32768), qtp(1 / 32768)),
            #'percent_1_2^14': (dbp(1 / 16384), qtp(1 / 16384)),
            #'percent_1_2^13': (dbp(1 / 8192), qtp(1 / 8192)),
            #'percent_1_2^12': (dbp(1 / 4096), qtp(1 / 4096)),
            #'percent_1_2^17_fw': (dbp(1 / 131072), quantize_tensor_min_max),
            #'percent_1_2^16_fw': (dbp(1 / 65536), quantize_tensor_min_max),
            #'percent_1_2^15_fw': (dbp(1 / 32768), quantize_tensor_min_max),
            #'percent_1_2^14_fw': (dbp(1 / 16384), quantize_tensor_min_max),
            #'percent_1_2^13_fw': (dbp(1 / 8192), quantize_tensor_min_max),
            #'percent_1_2^12_fw': (dbp(1 / 4096), quantize_tensor_min_max),
            #'percent_1_2^17_fa': (decide_bounds_min_max, qtp(1 / 131072)),
            #'percent_1_2^16_fa': (decide_bounds_min_max, qtp(1 / 65536)),
            #'percent_1_2^15_fa': (decide_bounds_min_max, qtp(1 / 32768)),
            #'percent_1_2^14_fa': (decide_bounds_min_max, qtp(1 / 16384)),
            #'percent_1_2^13_fa': (decide_bounds_min_max, qtp(1 / 8192)),
            #'percent_1_2^12_fa': (decide_bounds_min_max, qtp(1 / 4096)),
    #        }

    import itertools
    # The "bounds" algorithms for activations.
    # Basically, this is for deciding how to set the range for quantisation.
    # For example '3' means that the 99.99% percentile and 00.001% percentile of values are the minimum and maximum
    # (the values outside these ranges are ignored, and can be "clipped")
    A_ = {'m': decide_bounds_min_max, '3': dbp(1 / 1000), '4': dbp(1 / 10000), '5': dbp(1 / 100000)}
    # The "bounds" algorithms for weights
    B_ = {'m': quantize_tensor_min_max, '3': qtp(1 / 1000), '4': qtp(1 / 10000), '5': qtp(1 / 100000)}
    bounds_configs = merge_dicts([{k1 + '_' + k2: (v1, v2) for (k2, v2) in B_.items()} for (k1, v1) in A_.items()])

    if bounds_config_name not in bounds_configs:
        print("Invalid bounds configuration, try one of", bounds_configs.keys())
        return

    config = configs[quant_config_name]
    with torch.no_grad():
        #config = QuantConfig([4] * 13, [(4,4)] * 11)
        #config = QuantConfig([6] * 13, [(6,6)] * 11)

        bound_algs = bounds_configs[bounds_config_name]

        quant_net = setup_quant_net(net, activation_histograms, config, bound_algs[0], bound_algs[1])
        all_preds = []

        # Need to track intermediate values during inference
        trackers = get_intermediate_tracker(quant_net)

        loader = torch.utils.data.DataLoader(images, batch_size=10)
        # evaluate network
        with torch.no_grad():
            for X in tqdm.tqdm(loader):
                preds = quant_net(X)
                # convert output to numpy
                preds_np = preds.cpu().detach().numpy()
                all_preds.append(preds_np)


        with open(f'output/quantpreds_{quant_config_name}_{bounds_config_name}.npy', 'wb') as f:
            concated = np.concatenate(all_preds)
            #print(concated.shape)
            np.save(f, concated)

        histograms = [(tracker.range_pow_2, tracker.histogram.numpy()) for tracker in trackers]

        import pickle
        with open(f"output/outputhistograms_{quant_config_name}_{bounds_config_name}.pkl", "wb") as output_file:
            pickle.dump(histograms, output_file, protocol=pickle.HIGHEST_PROTOCOL)
