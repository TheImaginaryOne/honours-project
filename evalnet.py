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
parser.add_argument("images_dir", help="images directory", type=str)
parser.add_argument("--log-dir", help="log directory", type=str)
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

    @tf.function
    def run(x):
        return extract_net(x)

    if args.log_dir != None:
        log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        
        w = tf.summary.create_file_writer(args.log_dir)
        with w.as_default():
            tf.summary.trace_on(graph=True)
            features_list = extract_net.predict(image_gen)

            for batch in tqdm.tqdm(image_gen):
                run(batch)
            # Forward pass
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)
            for features in features_list:
                for layer in features:
                    tf.summary.histogram(f"activations/{layer.name}", layer, step=0)

    #print(features)

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
    test_accuracy(net, image_gen)
    #get_intermediate(net, image_gen)

main(args)
