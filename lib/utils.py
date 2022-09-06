import os
import argparse
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


# Utility to set network to eval mode.
def set_eval(net):
    net.eval()
    return net

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
