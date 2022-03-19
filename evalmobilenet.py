import tensorflow as tf
import os
import argparse
from tensorflow.keras.applications.mobilenet import MobileNet
from tqdm import tqdm

from tensorflow.python.client import timeline

parser = argparse.ArgumentParser("mobilenet")
parser.add_argument("images_dir", help="images directory", type=str)
#parser.add_argument("type", help="which type", type=str, choices=["quant", "normal"])
args = parser.parse_args()

# tf.compat.v1.enable_eager_execution()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np

def crop_center(image, final_width, final_height):
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        cropped_image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)
    return tf.image.resize(cropped_image, (final_width, final_height), method=tf.image.ResizeMethod.BICUBIC)

def process_img(file_path):
    img = tf.io.read_file(file_path)
    x = tf.image.decode_jpeg(img, channels=3)
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

def load_graph(path):
    graph = tf.Graph()
    # self.sess = tf.InteractiveSession(graph = self.graph)

    with open(path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def
    
# @tf.function
def load_images(img_list, batch_size):
    for i in range(0, len(img_list), batch_size):
        yield img_list[i:i+batch_size]


def main(args):
    """ Test the network! """
    images_dir = args.images_dir
#    quant = args.type == "quant"
#    print(f"Type: {args.type}")

    img_height = 224
    img_width = 224
    batch_size = 50

    image_files = [os.path.join(images_dir, d) for d in os.listdir(images_dir)]
    # sort by filenmae only. Important for labels compatibility
    image_files.sort(key=lambda f: f.split("/")[-1])

    net = MobileNet(weights='mobilenet/mobilenet_1_0_224_tf.h5', input_shape=(224,224,3))

    preds = []
    it = load_images(image_files, 20)
    # train loop
    for _ in tqdm(range(len(image_files) // 20)):
        fils = next(it)
        value = [process_img(x) for x in fils]
        pred = net(np.stack(value))   
        preds.append(pred)
#    fname = "mobilenet/mobilenet_v1_1.0_224_quant_frozen.pb" if quant else "mobilenet/mobilenet_v1_1.0_224_frozen.pb"
#
#    print(f"Loading {fname}")
#
#    gd = load_graph(fname)
#    inp, predictions = tf.compat.v1.import_graph_def(gd, return_elements = ['input:0', 'MobilenetV1/Predictions/Reshape_1:0'])
#
#    preds = []
#    with tf.compat.v1.Session(graph=inp.graph) as sess:
#        it = load_images(image_files, 20)
#        # cahced processing function
#        blank = tf.compat.v1.placeholder(tf.string, None)
#        cached = process_img(blank)
#        # train loop
#        for _ in tqdm(range(len(image_files) // 20)):
#            fils = next(it)
#            value = [sess.run(cached, feed_dict={blank: x}) for x in fils]
#            pred = sess.run(predictions, feed_dict={inp: value})   
#            preds.append(pred)
    #with open('output/mobilenetquantpreds.npy' if quant else 'output/mobilenetpreds.npy', 'wb') as f:
    with open('output/mobilenetpreds.npy', 'wb') as f:
        np.save(f, np.concatenate(preds, axis=0))


main(args)
