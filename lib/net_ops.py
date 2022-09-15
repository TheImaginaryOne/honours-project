from copy import deepcopy
from operator import mod
from typing import Tuple
import torch
import numpy as np
from lib.models import QuantConfig, QuantisableModule
from lib.quantnet import assert_equal

from lib.utils import get_module, iter_quantisable_modules_with_names, iter_trackable_modules_with_names, set_module

class DummyQuantTensor:
    """ A tensor object that also stores its bitwidth (just for tracking) """
    def __init__(self, data, bit_width=None, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self.bit_width = bit_width

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # get all inner data out first
        args_inner = [a._t if hasattr(a, '_t') else a for a in args]
        bit_widths = tuple(a.bit_width for a in args if hasattr(a, 'bit_width'))
        ret = func(*args_inner, **kwargs)
        return DummyQuantTensor(ret, bit_widths[0])
    
    def size(self) -> Tuple:
        return self._t.size()

class DummyQuantise(torch.nn.Module):
    def __init__(self, bit_width: int = 8):
        super().__init__()
        self.bit_width = bit_width

    def forward(self, x: DummyQuantTensor) -> torch.Tensor:
        """ A mock quantisation routine """
        x.bit_width = self.bit_width
        return x

def count_macs_conv(module: torch.nn.Conv2d, inputs: Tuple, y: torch.Tensor):
    hk, wk = module.kernel_size
    _, c_out, h_out, w_out = y.size()

    x = inputs[0]
    c_in = x.size()[1]

    module.__ops += wk * hk * c_in * c_out * h_out * w_out

def count_conv(module: torch.nn.Conv2d, inputs: Tuple, y: torch.Tensor):
    hk, wk = module.kernel_size
    _, c_out, h_out, w_out = y.size()

    x = inputs[0]
    c_in = x.size()[1]

    weight_bit_width = module.__weight_bit_width
    input_bit_width = x.bit_width

    module.__ops += wk * hk * c_in * c_out * h_out * w_out * (weight_bit_width * input_bit_width)

def count_macs_linear(module: torch.nn.Linear, inputs: Tuple, y: torch.Tensor):
    x = inputs[0]

    module.__ops += module.in_features * module.out_features

def count_linear(module: torch.nn.Linear, inputs: Tuple, y: torch.Tensor):
    weight_bit_width = module.__weight_bit_width

    x = inputs[0]
    input_bit_width = x.bit_width

    module.__ops += module.in_features * module.out_features * (weight_bit_width * input_bit_width)


PROFILERS = {
    torch.nn.Conv2d: count_conv,
    torch.nn.Linear: count_linear,
}

MAC_PROFILERS = {
    torch.nn.Conv2d: count_macs_conv,
    torch.nn.Linear: count_macs_linear,
}

def setup_net_for_profile(net: QuantisableModule, quant_config: QuantConfig) -> QuantisableModule:
    """ Inject dummy quantisation layers. """
    import copy
    quant_net = copy.deepcopy(net)

    trackable_modules = list(iter_trackable_modules_with_names(net.get_net()))
    # Add start module because we must insert a fakequantise before the first layer too!
    start = trackable_modules[0]
    trackable_modules = [start] + trackable_modules

    # Check length of config bit widths are as expected
    assert_equal(len(trackable_modules), len(quant_config.activation_bit_widths))

    quantisable_layers = list(iter_quantisable_modules_with_names(net.get_net()))
    assert_equal(len(quantisable_layers), len(quant_config.weight_bit_widths))

    # quantise the weights of the layers.
    for i, (layer_name, _) in enumerate(quantisable_layers):
        layer = get_module(quant_net.get_net(), layer_name)
        setattr(layer, "__weight_bit_width", quant_config.weight_bit_widths[i][0])
        setattr(layer, "__bias_bit_width", quant_config.weight_bit_widths[i][1])
    

    for i, (layer_name, _) in enumerate(trackable_modules):

        fake_quant = DummyQuantise(quant_config.activation_bit_widths[i])

        layer = get_module(quant_net.get_net(), layer_name)
        # insert the quant layer
        if i > 0:
            set_module(quant_net.get_net(), layer_name, torch.nn.Sequential(layer, fake_quant))
        else:
            # the start layer
            set_module(quant_net.get_net(), layer_name, torch.nn.Sequential(fake_quant, layer))

    return quant_net

def profile_net_bit_ops(net: QuantisableModule, quant_config: QuantConfig, input_shape: Tuple):
    quant_net = setup_net_for_profile(net, quant_config).get_net()

    dummy_input = DummyQuantTensor(torch.zeros(input_shape), None)
    
    def register_profiler(module: torch.nn.Module):
        def _test(module, x, y):
            print(module)

        # A variable that contains the number of ops.
        setattr(module, "__ops", 0)
        #module.register_forward_hook(test)

        module_type = type(module)
        if module_type in PROFILERS:
            module.register_forward_hook(PROFILERS[module_type])

    
    quant_net.apply(register_profiler)

    _ = quant_net(dummy_input)

    # sum all the ops
    total_bin_ops = 0

    for module in quant_net.modules():
        if hasattr(module, "__ops"):
            total_bin_ops += module.__ops

    return total_bin_ops
    
def profile_net_mac_ops(net: QuantisableModule, input_shape: Tuple):
    net = deepcopy(net.get_net())

    dummy_input = torch.zeros(input_shape)
    
    def register_profiler(module: torch.nn.Module):
        def _test(module, x, y):
            print(module)


        module_type = type(module)
        if module_type in MAC_PROFILERS:
            setattr(module, "__ops", 0)
            module.register_forward_hook(MAC_PROFILERS[module_type])

    
    net.apply(register_profiler)

    _ = net(dummy_input)

    # sum all the ops
    macs_by_layer = []

    for name, module in net.named_modules():
        if hasattr(module, "__ops"):
            macs_by_layer.append({'layer_name': name, 'mac_ops': module.__ops})

    return macs_by_layer