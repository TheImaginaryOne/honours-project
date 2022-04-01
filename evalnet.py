import os
import datetime
import argparse
import numpy as np
import tqdm

import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from utils import process_img, get_images, CustomImageData

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
    #features_list = [layer.output for layer in net.layers]

    #log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    print(net)
    for name, layer in net.named_parameters():
        print(name)
    
    #w = tf.summary.create_file_writer(args.log_dir)
    #features_list = extract_net.predict(image_gen)

def main(args):
    """ Test the network! """
    images_dir = args.images_dir
#    quant = args.type == "quant"
#    print(f"Type: {args.type}")

    validation_files, testing_files = get_images(images_dir)

    # train loop
    #net = MobileNet(weights='mobilenet/mobilenet_1_0_224_tf.h5', input_shape=(224,224,3))
    net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
    #net = models.vgg16(pretrained=True)

    # turn off some features for inference time
    net.eval()

    image_gen = CustomImageData(testing_files)
    if args.which == 'test-float':
        test_accuracy(net, image_gen)
    elif args.which == 'log-fixed':
        get_intermediate(net, image_gen)
    else:
        print("No task selected.")

main(args)
