import torch
import numpy as np
from quantnet import quantize_tensor, quantize_tensor_min_max, decide_bounds_min_max, decide_bounds_percentile

def test_quant():
    tensor = torch.Tensor([-1.7, 5.6, 7.1, -7.8, -9.1])

    # 3 bit quant; scale factor 2^1. Range from -8 to 6
    quantized = quantize_tensor(tensor, 3, 2)
    assert torch.equal(torch.Tensor([-2, 6, 6, -8, -8]), quantized)

def test_quant_min_max():
    tensor = torch.Tensor([-1.7, 5.6, 7.1, -7.8, -9.1])

    quantized = quantize_tensor_min_max(tensor, 3)
    assert torch.equal(torch.Tensor([0, 4, 8, -8, -8]), quantized)

def test_decide_histogram_min_max():
    hist = np.array([0, 0, 5, 5, 90080, 9900, 8, 2, 0])

    min_bin, max_bin = decide_bounds_min_max(hist)

    assert min_bin == 2
    assert max_bin == 7

def test_decide_histogram_percentile():
    hist = np.array([0, 0, 5, 5, 90080, 9900, 8, 2, 0])

    min_bin, max_bin = decide_bounds_percentile(hist, 0.0001)

    assert min_bin == 4
    assert max_bin == 5

def test_decide_histogram_percentile_two():
    hist = np.array([0, 0, 5, 7, 90083, 9906, 2, 2, 0])

    min_bin, max_bin = decide_bounds_percentile(hist, 0.0001)

    assert min_bin == 4
    assert max_bin == 5

def test_decide_histogram_percentile_three():
    hist = np.array([0, 6, 88, 6, 0])

    min_bin, max_bin = decide_bounds_percentile(hist, 0.05)

    assert min_bin == 2
    assert max_bin == 2
