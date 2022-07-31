import os
import argparse
import numpy as np
import torch
import PIL
from typing import Optional
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

process_img = T.Compose([T.Resize(256, interpolation=InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), normalize])

import itertools

NAMES = ['m', '3', '4', '5']
ALL_BOUNDS = list(itertools.chain(*[[n1 + '_' + n2 for n2 in NAMES] for n1 in NAMES]))

ALL_NET_CONFIGS = itertools.product(["8b", 
                "6b",
                "4b",
                "8b6b_fc_1",
                "8b4b_fc_1",
                ], ALL_BOUNDS)

# the product of tabulated sets
CONFIG_SETS = {'all': ALL_NET_CONFIGS} #, 'fa': ALL_NET_CONFIGS_FA, 'fw': ALL_NET_CONFIGS_FW}

# Utility to set network to eval mode.
def set_eval(net):
    net.eval()
    return net

def fuse_resnet(net):
    fused_model = torch.quantization.fuse_modules(net, [["conv1", "bn1"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    return fused_model

class QuantisableModule:
    def get_layers_to_track(self):
        pass
    def get_net(self) -> torch.nn.Module:
        pass


class QuantisableVgg11(QuantisableModule):
    def __init__(self):
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        self.net.eval()
    def get_layers_to_track(self):
        net = self.net
        start_layer = net.features[0]
        # Track the activations of the ReLU layers
        output_layers = [net.features[i] for i in [1, 4, 7, 9, 12, 14, 17, 19]] \
            + [net.classifier[i] for i in [1, 4, 6]]
        return start_layer, output_layers
    def get_net(self) -> torch.nn.Module:
        return self.net

class QuantisableResnet18(QuantisableModule):
    def __init__(self):
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.net.eval()
        self.net = fuse_resnet(self.net)
    def get_layers_to_track(self):
        net = self.net
        start_layer = net.conv1
        output_layers = [net.relu]
        for i, l in enumerate([net.layer1, net.layer2, net.layer3, net.layer4]):
            output_layers.append(l[0].relu)
            output_layers.append(l[0].bn2)
            if i > 0:
                output_layers.append(l[0].downsample[1])
            output_layers.append(l[1].relu)
            output_layers.append(l[1].bn2)
        output_layers.append(net.avgpool)
        output_layers.append(net.fc)
        return start_layer, output_layers
    def get_net(self) -> torch.nn.Module:
        return self.net

NETS = {'vgg11': QuantisableVgg11(),
        'resnet18': QuantisableResnet18()}

def get_net(name: str) -> QuantisableModule:
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
