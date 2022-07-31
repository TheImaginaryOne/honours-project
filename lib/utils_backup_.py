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
ALL_NET_CONFIGS = itertools.product(["8b", 
                "7b",
                "6b",
                "5b",
                "4b",
                "8b7b_fc_1",
                "8b6b_fc_1",
                "8b5b_fc_1",
                "8b4b_fc_1",
                ], ["minmax", "percent_1_2^17", "percent_1_2^16", "percent_1_2^15", "percent_1_2^14"])

ALL_NET_CONFIGS_FW = itertools.product(["8b", 
                "7b",
                "6b",
                "5b",
                "4b",
                "8b7b_fc_1",
                "8b6b_fc_1",
                "8b5b_fc_1",
                "8b4b_fc_1",
                ],['minmax', 'percent_1_2^17_fw',
                 'percent_1_2^16_fw',
                 'percent_1_2^15_fw',
                 'percent_1_2^14_fw',
                 'percent_1_2^13_fw',
                 'percent_1_2^12_fw',
                ])
ALL_NET_CONFIGS_FA = itertools.product(["8b", 
                "7b",
                "6b",
                "5b",
                "4b",
                "8b7b_fc_1",
                "8b6b_fc_1",
                "8b5b_fc_1",
                "8b4b_fc_1",
                ],['percent_1_2^17_fa',
                 'percent_1_2^16_fa',
                 'percent_1_2^15_fa',
                 'percent_1_2^14_fa',
                 'percent_1_2^13_fa',
                 'percent_1_2^12_fa',
                ])

# the product of tabulated sets
CONFIG_SETS = {'all': ALL} #, 'fa': ALL_NET_CONFIGS_FA, 'fw': ALL_NET_CONFIGS_FW}

def get_net():
    net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
    return net

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
