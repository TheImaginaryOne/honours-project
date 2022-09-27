import itertools
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List
import torch, torchvision

def dict_subset(d: dict, keys: List[str]):
    return {k: d[k] for k in keys}
class QuantConfig:
    def __init__(self, activation_bit_widths: List[int], weight_bit_widths: List[tuple[int, int]]):
        # A list of integers, where the nth value denotes the number of bits on the nth quantisable layer
        # We have two configs for the activation and weight bit widths.
        self.activation_bit_widths = activation_bit_widths
        self.weight_bit_widths = weight_bit_widths

def vggnet_configs():
    c = {'8b': QuantConfig([8] * 12, [(8, 8)] * 11),\
        '6b': QuantConfig([6] * 12, [(6, 6)] * 11),
        '4b': QuantConfig([4] * 12, [(4, 4)] * 11),
        }
    
    N_LAYERS = 11
    for i in [2, 4, 6, 8]:
        for b1, b2 in [(8,6),(6,4),(8,4)]:
            # Note: the (i + 1) is because there are N_LAYER + 1 activations, if we count the input image.
            c[f"{b1}b_{b2}b_{i}"] = \
                QuantConfig([b1] * i + [b2] * (N_LAYERS - i + 1), [(b1, b1)] * i + [(b2, b2)] * (N_LAYERS - i))
    
    return c

def resnet18_configs():
    c = {'8b': QuantConfig([8] * 30, [(8, 8)] * 21),\
        '6b': QuantConfig([6] * 30, [(6, 6)] * 21),
        '4b': QuantConfig([4] * 30, [(4, 4)] * 21),
        }
    
    keys = ["l1", "l2", "l3"]
    
    N_TRACKED_LAYERS = 30
    N_QUANT_LAYERS = 21
    for key, (i, j) in zip(keys, [(7, 5), (14, 10), (21, 15)]):
        for b1, b2 in [(8,6),(6,4),(8,4)]:
            # Note: the (i + 1) is because there are N_LAYER + 1 activations, if we count the input image.
            c[f"{b1}b_{b2}b_{key}"] = \
                QuantConfig([b1] * i + [b2] * (N_TRACKED_LAYERS - i),\
                     [(b1, b1)] * j + [(b2, b2)] * (N_QUANT_LAYERS - j))
    
    return c

ALL_VGGNET_CONFIGS = vggnet_configs()

# the product of tabulated sets
CONFIG_SETS = {'vgg11': vggnet_configs(), \
    'resnet18': resnet18_configs(),
    #'resnet34': ALL_CONFIGS,
    } #, 'fa': ALL_NET_CONFIGS_FA, 'fw': ALL_NET_CONFIGS_FW}

PERCENTILE_TEST_CONFIG_SETS = {'vgg11': dict_subset(vggnet_configs(), ["8b", "6b", "8b_6b_8", "8b_4b_8", "6b_4b_8"]),\
    'resnet18': dict_subset(resnet18_configs(), ["8b", "6b", "8b_6b_l2", "8b_4b_l2"])}

class QuantisableModule(ABC):
    @abstractmethod
    def get_net(self) -> torch.nn.Module:
        pass

def fuse_vgg11(net):
    features_fuse = [["0", "1"], ["3", "4"], ["6", "7"], ["8", "9"], ["11", "12"], ["13", "14"], ["16", "17"], ["18", "19"]]
    clf_fuse = [["0", "1"], ["3", "4"]]
    fused_model = deepcopy(net)
    fused_model.features = torch.quantization.fuse_modules(net.features, features_fuse)
    fused_model.classifier = torch.quantization.fuse_modules(net.classifier, clf_fuse)
    return fused_model
class QuantisableVgg11(QuantisableModule):
    def __init__(self):
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        self.net = fuse_vgg11(self.net)
        self.net.eval()
    def get_net(self) -> torch.nn.Module:
        return self.net

# === ResNet stuff
def fuse_resnet(net):
    fused_model = torch.quantization.fuse_modules(net, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for block_name, basic_block in module.named_children():
                # Don't fuse the ReLU!! It appears after the residual connections inside a basic block
                basic_block = torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"]])
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
                basic_block = QuantizableBasicBlock(basic_block)

                setattr(module, block_name, basic_block)
    return fused_model
class QuantizableConvRelu(torch.nn.Module):
    def __init__(self, conv: torch.nn.Module, relu: torch.nn.Module) -> None:
        super().__init__()

        self.conv = conv
        self.relu = relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        return out

class QuantizableBasicBlock(torch.nn.Module):
    """
    Based on torchvision code.
    Because the self.relu is reused two times in the original BasicBlock,
    this is a problem for quantisation. So we modify it a little bit
    """
    def __init__(self, block: torchvision.models.resnet.BasicBlock) -> None:
        super().__init__()
        # copy the old basic block.

        # For inference purposes, we don't need the batchnorm layer
        # (as we fused it before).
        self.conv_relu1 = QuantizableConvRelu(deepcopy(block.conv1), torch.nn.ReLU())

        self.conv2 = deepcopy(block.conv2)
        self.downsample = deepcopy(block.downsample)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv_relu1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu2(out)

        return out

class QuantisableResnet(QuantisableModule):
    layer_names = ["layer1", "layer2", "layer3", "layer4"]

    def __init__(self, net: torch.nn.Module, layer_sizes: list[int]):
        self.net = net
        self.net.eval()
        self.net = fuse_resnet(self.net)
        self.layer_sizes = layer_sizes
    def get_net(self) -> torch.nn.Module:
        return self.net

def get_resnet(name):
    if name == "resnet18":
        return QuantisableResnet(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True), [2,2,2,2])
    elif name == "resnet34":
        return QuantisableResnet(torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True), [3,4,6,3])

NETS = {'vgg11': QuantisableVgg11(),
        'resnet18': get_resnet("resnet18"),
        }

def get_net(name: str) -> QuantisableModule:
    #print(NETS[name].get_net())
    return NETS[name]
