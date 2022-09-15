
import functools
from math import prod
from typing import List, Tuple

from lib.models import QuantConfig, QuantisableModule
from lib.quantnet import assert_equal
from lib.utils import iter_quantisable_modules_with_names

def product_of_tuple(l: Tuple[int]) -> int:
    return functools.reduce(lambda a, b: a * b, l)

def quant_model_size(net: QuantisableModule, quant_config: QuantConfig) -> int:
    """ Estimated size (bytes) when quantised using a particular config """
    import copy
    quant_net = copy.deepcopy(net)

    quantisable_layers = list(iter_quantisable_modules_with_names(net.get_net()))
    assert_equal(len(quantisable_layers), len(quant_config.weight_bit_widths))

    total_size = 0

    # quantise the weights of the layers.
    for i, (layer_name, layer) in enumerate(quantisable_layers):
        weight_size = product_of_tuple(layer.weight.detach().size())
        bias_size = product_of_tuple(layer.bias.detach().size())

        weight_bits, bias_bits = quant_config.weight_bit_widths[i]

        total_size += weight_bits * weight_size + bias_bits * bias_size
    
    return total_size / 8

def get_model_layer_sizes(net: QuantisableModule) -> List:
    quantisable_layers = list(iter_quantisable_modules_with_names(net.get_net()))

    total_size = 0

    sizes = []

    # quantise the weights of the layers.
    for i, (layer_name, layer) in enumerate(quantisable_layers):
        weight_size = product_of_tuple(layer.weight.detach().size())
        bias_size = product_of_tuple(layer.bias.detach().size())

        sizes.append({'layer_name': layer_name, 'weight_size': weight_size, 'bias_size': bias_size})
    
    return sizes