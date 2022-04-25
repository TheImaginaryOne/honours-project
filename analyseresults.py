import numpy as np
import argparse

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser("mobilenet")
parser.add_argument("labels_file_name", help="labels_file", type=str)
parser.add_argument("file_name", help="predictions_file", type=str)
args = parser.parse_args()

def value_counts(x: np.ndarray):
    unique, counts = np.unique(x, return_counts=True)
    return np.asarray((unique, counts)).T

def compare():
    label_name = args.labels_file_name

    with open(label_name) as label_f:
        labels = np.array([int(s.split()[1]) for s in label_f.readlines()])

    # load prediction weights
    fname = args.file_name

    with open(fname, 'rb') as f:
        quantpreds = np.load(f)

    #print(value_counts(quantpreds))

    quant_pred_labels = quantpreds.argmax(axis=1)

    print("len labels:", len(labels), ", len predicted labels:", len(quant_pred_labels))
    print(labels, quant_pred_labels)
    print("top 1 error:", np.mean(labels != quant_pred_labels))

compare()
