import tensorflow as tf
import os
import argparse
from tensorflow.keras.applications.mobilenet import MobileNet
from utils import process_img, get_images, get_image_gen

from tensorflow.python.client import timeline

parser = argparse.ArgumentParser("mobilenet")
parser.add_argument("images_dir", help="images directory", type=str)
#parser.add_argument("type", help="which type", type=str, choices=["quant", "normal"])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main(args):
    """ Test the network! """
    images_dir = args.images_dir
#    quant = args.type == "quant"
#    print(f"Type: {args.type}")

    img_height = 224
    img_width = 224
    batch_size = 50


    net = MobileNet(weights='mobilenet/mobilenet_1_0_224_tf.h5', input_shape=(224,224,3))

    _, image_files = get_images(images_dir)
    image_gen = get_image_gen(image_files)

    # train loop
    preds = net.predict(image_gen, verbose=1)
    with open('output/mobilenetpreds.npy', 'wb') as f:
        np.save(f, preds, axis=0)


main(args)
