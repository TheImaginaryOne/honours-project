import numpy as np
import argparse

#parser = argparse.ArgumentParser("mobilenet")
#parser.add_argument("type", help="which type", type=str, choices=["quant", "normal"])
#args = parser.parse_args()

label_name = 'input/truth_correct.txt'

with open(label_name) as label_f:
    labels = np.array([int(s.split()[1]) for s in label_f.readlines()])
labels = labels[10000:]

fname = 'output/floatpreds.npy'

with open(fname, 'rb') as f:
    quantpreds = np.load(f)

quant_pred_labels = quantpreds.argmax(axis=1)

print(labels, quant_pred_labels)
print("top 1 error:", np.mean(labels != quant_pred_labels))
