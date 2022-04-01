import os
import argparse
import numpy as np
import torch
import PIL
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

process_img = T.Compose([T.Resize(256, interpolation=InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), normalize])

# Get image path to tune the classifier quantiser params.
def get_images(images_dir):
    image_files = [os.path.join(images_dir, d) for d in os.listdir(images_dir)]
    # sort by filenmae only. Important for labels compatibility
    image_files.sort(key=lambda f: f.split("/")[-1])

    split_point = len(image_files) // 5
    # first is validation; second is testing
    return (image_files[:split_point], image_files[split_point:])

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
