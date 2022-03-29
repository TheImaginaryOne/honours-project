import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import argparse
import numpy as np

def crop_center(image, final_width, final_height):
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        cropped_image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)
    return tf.image.resize(cropped_image, (final_width, final_height), method=tf.image.ResizeMethod.BICUBIC)

def process_img(x):
    x = tf.cast(x, tf.float32)
    x = crop_center(x, 256, 256)
    x = tf.image.central_crop(x, central_fraction=0.875)
    #x = tf.expand_dims(x, 0)
    #x = tf.image.resize(x, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    #x = tf.squeeze(x, [0])
    
    # x = tf.keras.applications.mobilenet.preprocess_input(x)
    
    # scale colours
    x = tf.multiply(x, 1./127.5)
    x = tf.subtract(x, 1.0)
    
    return x

# Get image path to tune the classifier quantiser params.
def get_images(images_dir):
    image_files = [os.path.join(images_dir, d) for d in os.listdir(images_dir)]
    # sort by filenmae only. Important for labels compatibility
    image_files.sort(key=lambda f: f.split("/")[-1])

    split_point = len(image_files) // 5
    # first is validation; second is testing
    return (image_files[:split_point], image_files[split_point:])

class CustomImageGen(tf.keras.utils.Sequence):
    
    def __init__(self, file_paths, batch_size):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.len = len(file_paths)
    
    def on_epoch_end(self):
        pass
    
    def __getitem__(self, index):
        b = self.batch_size
        images = []
        for file_path in self.file_paths[b * index:b * (index + 1)]:
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = process_img(img)
            images.append(img)
        return np.stack(images)
    
    def __len__(self):
        return (self.len - 1) // self.batch_size + 1

def get_image_gen(image_files):
    return CustomImageGen(image_files, 20)
