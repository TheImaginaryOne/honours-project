import itertools
from abc import ABC, abstractmethod
import torch

def self_product(l):
    return list(itertools.chain(*[[n1 + '_' + n2 for n2 in l] for n1 in l]))

NAMES = ['m', '3', '4', '5']
A = ['3', '4', '5', 'm']

ALL_VGGNET_CONFIGS = itertools.product(["8b", 
                "6b",
                "4b",
                "8b6b_fc",
                "8b4b_fc",
                ], self_product(A))

ALL_CONFIGS = itertools.product(["8b", 
                "6b",
                "4b",
                #"8b6b",
                #"8b4b",
                ], self_product(A))

# the product of tabulated sets
CONFIG_SETS = {'vgg11': {'all': ALL_VGGNET_CONFIGS}, \
    'resnet18': {'all': ALL_CONFIGS},
    'resnet34': {'all': ALL_CONFIGS},
    } #, 'fa': ALL_NET_CONFIGS_FA, 'fw': ALL_NET_CONFIGS_FW}

class QuantisableModule(ABC):
    @abstractmethod
    def get_net(self) -> torch.nn.Module:
        pass

class QuantisableVgg11(QuantisableModule):
    def __init__(self):
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        self.net.eval()
    def get_net(self) -> torch.nn.Module:
        return self.net

def fuse_resnet(net):
    fused_model = torch.quantization.fuse_modules(net, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    return fused_model

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
        'resnet34': get_resnet("resnet34"),
        }

def get_net(name: str) -> QuantisableModule:
    #print(NETS[name].get_net())
    return NETS[name]