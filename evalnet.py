import os
import datetime
import argparse
import numpy as np
import tqdm
import pickle

import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from lib.utils import process_img, get_images, CustomImageData, get_net
from lib.layer_tracker import HistogramTracker

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
    # turn off some features for inference time (IMPORTANT)
    net.eval()

    loader = DataLoader(image_gen, batch_size=20)

    all_preds = []
    with torch.no_grad():
        for X in tqdm.tqdm(loader):
            preds = net(X)
            # convert output to numpy
            preds_np = preds.cpu().detach().numpy()
            all_preds.append(preds_np)


    with open('output/floatpreds.npy', 'wb') as f:
        concated = np.concatenate(all_preds)
        print(concated.shape)
        np.save(f, concated)

def get_intermediate(net, image_gen):
    net.eval()

    print(net)

    # Log all activations (outputs) of relevant layers
    output_layers = [net.features[i] for i in [0, 2, 5, 7, 10, 12, 15, 17, 20]] + [net.avgpool] + [net.classifier[i] for i in [1, 4, 6]]
    hist_tracker = [HistogramTracker() for i in range(len(output_layers))]

    def hist_tracker_hook(hist_tracker):
        def f(module, input, output):
            hist_tracker.update(output)
        return f
    def hist_tracker_input_hook(hist_tracker):
        def f(module, input, output):
            hist_tracker.update(input[0])
        return f

    for i, layer in enumerate(output_layers):
        if i == 0:
            layer.register_forward_hook(hist_tracker_input_hook(hist_tracker[i]))
        else:
            layer.register_forward_hook(hist_tracker_hook(hist_tracker[i]))
    
    loader = DataLoader(image_gen, batch_size=20)
    with torch.no_grad():
        for X in tqdm.tqdm(loader):
            preds = net(X)

    histograms = [(tracker.range_pow_2, tracker.histogram.numpy()) for tracker in hist_tracker]

    with open(r"output/outputhistogram.pkl", "wb") as output_file:
        pickle.dump(histograms, output_file, protocol=pickle.HIGHEST_PROTOCOL)

def main(args):
    """ Test the network! """
    images_dir = args.images_dir
#    quant = args.type == "quant"
#    print(f"Type: {args.type}")

    validation_files, testing_files = get_images(images_dir)

    # train loop
    #net = MobileNet(weights='mobilenet/mobilenet_1_0_224_tf.h5', input_shape=(224,224,3))
    net = get_net()
    #net = models.vgg16(pretrained=True)

    if args.which == 'test-float':
        image_gen = CustomImageData(testing_files)
        test_accuracy(net, image_gen)
    elif args.which == 'log-fixed':
        image_gen = CustomImageData(validation_files)
        get_intermediate(net, image_gen)
    else:
        print("No task selected.")

main(args)
