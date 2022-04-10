import numpy as np
import argparse

parser = argparse.ArgumentParser("mobilenet")
parser.add_argument("which", help="which type", type=str, choices=["quant", "float"])
args = parser.parse_args()

label_name = 'input/truth_correct.txt'

with open(label_name) as label_f:
    labels = np.array([int(s.split()[1]) for s in label_f.readlines()])
labels = labels[5000:]

fname = 'output/quantpreds.npy' if args.which == "quant" else 'output/floatpreds.npy'

with open(fname, 'rb') as f:
    quantpreds = np.load(f)

quant_pred_labels = quantpreds.argmax(axis=1)

print("len labels:", len(labels), ", len predicted labels:", len(quant_pred_labels))
print(labels, quant_pred_labels)
print("top 1 error:", np.mean(labels != quant_pred_labels))
