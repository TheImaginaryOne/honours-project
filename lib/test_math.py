import torch
import numpy as np
from numpy.testing import assert_almost_equal
from lib.math_utils import quantize_tensor, quantize_tensor_percentile, min_pow_2_scale
from lib.layer_tracker import Histogram, decide_bounds_min_max, decide_bounds_percentile

def test_quant():
    tensor = torch.Tensor([-1.7, 5.6, 7.1, -7.8, -9.1])

    # 3 bit quant; scale factor 2^1. Range from -8 to 6
    quantized = quantize_tensor(tensor, 3, 2)
    assert torch.equal(torch.Tensor([-2, 6, 6, -8, -8]), quantized)

def test_quant_min_max():
    tensor = torch.Tensor([-1.7, 5.6, 7.1, -7.8, -9.1])

    quantized = quantize_tensor_percentile(tensor, 3, 0., 1.)
    assert torch.equal(torch.Tensor([0, 4, 8, -8, -8]), quantized)

def test_decide_histogram_min_max():
    # from -2 ** 4 to 2 ** 4
    hist = Histogram(4, np.array([0, 0, 5, 5, 90080, 9900, 10, 0]))

    min_val, max_val = decide_bounds_min_max(hist)

    assert_almost_equal(min_val, -8)
    assert_almost_equal(max_val, 12)

def test_decide_histogram_percentile():
    hist = Histogram(2, np.array([0, 0, 5, 5, 90080, 9900, 8, 2]))

    min_val, max_val = decide_bounds_percentile(hist, 0.0001)
    # 0.0001 percentile is 10; 0.9999 percentile is 99990

    assert_almost_equal(min_val, 0)
    assert_almost_equal(max_val, 2)

def test_decide_histogram_percentile_two():
    hist = Histogram(2, np.array([0, 5, 10, 50, 20, 10, 5, 0]))

    min_val, max_val = decide_bounds_percentile(hist, 0.1)

    assert_almost_equal(min_val, -1.5)
    assert_almost_equal(max_val, 1.5)

def test_min_pow_2():
    # scale factor = 2**3
    assert 3 == min_pow_2_scale(-15, 28, 3)