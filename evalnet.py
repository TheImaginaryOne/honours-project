import tensorflow as tf
import os
import datetime
import argparse
import numpy as np
import tqdm
from tensorflow.keras.applications.mobilenet import MobileNet
from utils import process_img, get_images, get_image_gen

from tensorflow.python.client import timeline

parser = argparse.ArgumentParser("mobilenet")

subparsers = parser.add_subparsers(dest='which') # store subcommand name in "which" field
parser_test = subparsers.add_parser('test-float')
parser_test.add_argument("images_dir", help="images directory", type=str)

parser_log = subparsers.add_parser('log-fixed')
parser_log.add_argument("images_dir", help="images directory", type=str)
#parser.add_argument("type", help="which type", type=str, choices=["quant", "normal"])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def test_accuracy(net, image_gen):

    preds = net.predict(image_gen, verbose=1)

    with open('output/mobilenetpreds.npy', 'wb') as f:
        np.save(f, preds)

def get_intermediate(net, image_gen):
    features_list = [layer.output for layer in net.layers]

    extract_net = tf.keras.Model(inputs = net.input, outputs = features_list)

    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    w = tf.summary.create_file_writer(args.log_dir)
    features_list = extract_net.predict(image_gen)

    features = batch.predict(image_gen, verbose=1)
    for batch in tqdm.tqdm(image_gen):
        run(batch)

def main(args):
    """ Test the network! """
    images_dir = args.images_dir
#    quant = args.type == "quant"
#    print(f"Type: {args.type}")

    img_height = 224
    img_width = 224
    batch_size = 50

    validation_files, testing_files = get_images(images_dir)

    # train loop
    #net = MobileNet(weights='mobilenet/mobilenet_1_0_224_tf.h5', input_shape=(224,224,3))
    net = tf.keras.applications.VGG16(weights = 'imagenet')
    print(net.layers)

    image_gen = get_image_gen(testing_files)
    if args.which == 'test-float':
        test_accuracy(net, image_gen)
    elif args.which == 'log-fixed':
        get_intermediate(net, image_gen)
    else:
        print("No task selected.")

main(args)
