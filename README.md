## Setup

Download the 50,000 validation images from the Imagenet website

Create a folder called `output`; create another one called `input`.
[TODO: I hardcoded the output folder. It's not configurable at the moment! Sorry for the jank code!]

Place the input labels `truth_correct.txt` anywhere (I placed it in the input folder);

run the script `sample_images.py`:

`python3 sample_images.py input/truth_correct.txt input/truth_test.txt input/truth_val.txt`


Create an `.env` file with configuration variables:

```
TEST_SUBSET_LIST=input/truth_test.txt # The testing set for evaluation
VAL_SUBSET_LIST=input/truth_val.txt # the validation set for hyperparameters
IMAGES_DIR=../val_unblurred # Place the 50,000 image set here
```

## Commands.

The entry point is `evalnet.py`.

Test floating point: `python3 evalnet.py resnet18 test-float`. Possible options: vgg11, resnet18.

Log activation and weight histograms (this is required before the fixed-point tests): `python3 evalnet.py resnet18 log-fixed`.

Run a set of experiments for the quantisation of the neural net, including the hyperparameter tuning of the percentiles: `python3 evalnet.py resnet18 test-fixed-all`

Run the experiments for all percentiles (and output the results for each percentile): `python3 evalnet.py resnet18 test-fixed-debug-percentiles`

Print out the network architecture and some basic info: `python3 evalnet.py resnet18 print-net`

Analyse results (this plots all results):

`python3 analyseresults.py input/truth_subset.txt set ./output resnet18 all`

## Structure

`lib/` has the important code for the quantisation of the neural nets, as well as printing the neural network operations

## Libraries

pytorch (and torchvision), tqdm, python-dotenv, numpy (and pandas), matplotlib, seaborn