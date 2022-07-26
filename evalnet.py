import os
import datetime
import argparse
import numpy as np
import tqdm
import pickle

import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from lib.utils import process_img, get_images, CustomImageData, get_net, CONFIG_SETS
from lib.layer_tracker import HistogramTracker
from lib.quantnet import test_quant

parser = argparse.ArgumentParser("quant-net")
parser.add_argument("images_dir", help="images directory", type=str)
parser.add_argument("-l", "--labels-file", help="optional file with labels (use for image list)", type=str)

subparsers = parser.add_subparsers(dest='which') # store subcommand name in "which" field

parser_test = subparsers.add_parser('test-float')
parser_log = subparsers.add_parser('log-fixed')
parser_subset_fixed = subparsers.add_parser('test-subset-fixed')
parser_subset_fixed.add_argument('subset', type=str) #= subparsers.add_parser('test-subset-fixed')

parser_test_fixed = subparsers.add_parser('test-fixed')
parser_test_fixed.add_argument('quant_config', help='the quant config to use', type=str)
parser_test_fixed.add_argument('bounds', help='the bounds to use', type=str)
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
    output_layers = [net.features[i] for i in [0, 1, 4, 7, 9, 12, 14, 17, 19]] \
        + [net.classifier[i] for i in [1, 4, 6]]
        # + [net.avgpool]  is a no-op
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

def get_accuracy(label_file_name, file_name):
    with open(label_file_name) as label_f:
        labels = np.array([int(s.split()[1]) for s in label_f.readlines()])

    with open(file_name, 'rb') as f:
        quantpreds = np.load(f)

    #print(value_counts(quantpreds))

    quant_pred_labels = quantpreds.argmax(axis=1)

    #print(labels, quant_pred_labels)
    print("top 1 error:", np.mean(labels != quant_pred_labels))

def main(args):
    """ Test the network! """
    images_dir = args.images_dir
#    quant = args.type == "quant"
#    print(f"Type: {args.type}")

    testing_files = get_images(images_dir, args.labels_file)

    # train loop
    #net = MobileNet(weights='mobilenet/mobilenet_1_0_224_tf.h5', input_shape=(224,224,3))
    net = get_net()
    #net = models.vgg16(pretrained=True)

    if args.which == 'test-float':
        image_gen = CustomImageData(testing_files)
        test_accuracy(net, image_gen)
    elif args.which == 'log-fixed':
        image_gen = CustomImageData(testing_files)
        get_intermediate(net, image_gen)
    elif args.which == 'test-fixed':
        image_gen = CustomImageData(testing_files)
        with open("output/outputhistogram.pkl", "rb") as f:
            histograms = pickle.load(f)
        test_quant(net, histograms, image_gen, args.quant_config, args.bounds) # in other module
    elif args.which == "test-subset-fixed":
        image_gen = CustomImageData(testing_files)
        with open("output/outputhistogram.pkl", "rb") as f:
            histograms = pickle.load(f)

        import itertools
        # test the neural net for all configurations
        for (quant_config, bounds) in CONFIG_SETS[args.subset]:
            print("Testing:", quant_config, bounds)
            test_quant(net, histograms, image_gen, quant_config, bounds) # in other module
    else:
        print("No task selected.")

main(args)
