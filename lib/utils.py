import os
import argparse
from abc import ABC, abstractmethod
import numpy as np
import torch
import PIL
from typing import Optional
import torchvision
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

process_img = T.Compose([T.Resize(256, interpolation=InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), normalize])

import itertools

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

# Utility to set network to eval mode.
def set_eval(net):
    net.eval()
    return net

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

# ====
def iter_trackable_modules(module: torch.nn.Module):
    for _, child in iter_trackable_modules_helper_(module, None):
        yield child

def iter_trackable_modules_with_names(module: torch.nn.Module):
    yield from iter_trackable_modules_helper_(module, None)

def iter_trackable_modules_helper_(module: torch.nn.Module, parent_name: Optional[str]):
    """ Iterate moduls recursively """
    # These are leaves that are sequential modules but we want to ignore
    # the "content" inside, because they are assumed to be one operation in the quantised version.
    seq_leaf_types = [torchvision.ops.misc.ConvNormActivation, torch.nn.intrinsic.modules._FusedModule]

    ignore = [torch.nn.Identity, torch.nn.Dropout]
    for name, child in module.named_children():
        full_name = name if parent_name is None else f"{parent_name}.{name}"
        # not a leaf; ignore it
        if not type(child) in seq_leaf_types:
            yield from iter_trackable_modules_helper_(child, full_name)
            
        if not (type(child) in ignore or isinstance(child, torch.nn.Sequential)) \
            or type(child) in seq_leaf_types:
            yield (full_name, child)

def iter_quantisable_modules_with_names(module: torch.nn.Module):
    yield from iter_quantisable_modules_helper_(module, None)

def iter_quantisable_modules_helper_(module: torch.nn.Module, parent_name: Optional[str]):
    """ Iterate moduls recursively """
    leaves = [torch.nn.Conv2d, torch.nn.Linear]
    for name, child in module.named_children():
        full_name = name if parent_name is None else f"{parent_name}.{name}"
        yield from iter_quantisable_modules_helper_(child, full_name)
            
        if type(child) in leaves:
            yield (full_name, child)

class QuantisableModule(ABC):
    @abstractmethod
    def get_net(self) -> torch.nn.Module:
        pass

# Access a nested object by dot notation (for example, relu.0)
# Based on code in pytorch/pytorch
def get_module(model, submodule_key):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod

def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)

    #print("mod:", cur_mod)
    if not hasattr(cur_mod, tokens[-1]):
        raise RuntimeError(f"attr does not exist: {cur_mod}, {tokens[-1]}")
    setattr(cur_mod, tokens[-1], module)

class QuantisableVgg11(QuantisableModule):
    def __init__(self):
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        self.net.eval()
    def get_net(self) -> torch.nn.Module:
        return self.net

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

# ========

# Get image path to tune the classifier quantiser params.
def get_images(images_dir, labels_file: Optional[str] = None) -> list[str]:
    if labels_file is not None:
        with open(labels_file, "r") as f:
            lines = f.readlines()
            image_files = [os.path.join(images_dir, l.split()[0]) for l in lines]
    else:
        image_files = [os.path.join(images_dir, d) for d in os.listdir(images_dir)]
    # sort by filenmae only. Important for labels compatibility
    image_files.sort(key=lambda f: f.split("/")[-1])

    # first is validation; second is testing
    return image_files

class CustomImageData(torch.utils.data.Dataset):
    
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.len = len(file_paths)
    
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        img = PIL.Image.open(file_path).convert('RGB')
        img = process_img(img)
        return img
    
    def __len__(self):
        return self.len
