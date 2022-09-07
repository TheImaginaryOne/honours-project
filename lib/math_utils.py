import math
from typing import Tuple
import torch
import numpy as np

def min_pow_2(x):
    if x < 0:
        x = -x
    if x == 0:
        return -126 # TODO? The minumum floating point value
    return math.ceil(math.log(x, 2))

def quantize_tensor(input: torch.Tensor, bit_width: int, scale: int) -> torch.Tensor:
    """ Fake quantize utility """
    return torch.fake_quantize_per_tensor_affine(input, scale, 0, -2**(bit_width-1), 2**(bit_width - 1) - 1)

def min_pow_2_scale(min_val: float, max_val: float, bit_width: int):
    """ Min power of 2 to represent """
    # this only works if this is true
    assert min_val <= 0
    assert max_val >= 0
    x = min_pow_2(min_val / (-2**(bit_width-1) - 1./2))
    y = min_pow_2(max_val / (2**(bit_width-1) - 1./2))
    return max(x, y)

# ====
def get_tensor_percentile(input: torch.Tensor, lq: float, uq: float) -> Tuple[float, float]:
    p = np.quantile(input.numpy(), [lq, uq])
    return p[0], p[1]

def quantize_tensor_percentile(input: torch.Tensor, bit_width: int, lq: float, uq: float) -> torch.Tensor:
    """ Quantise tensor, bounded according to lower quartile and upper quartile """
    #print(input.size())
    p = np.quantile(input.numpy(), [lq, uq])
    min_val, max_val = p[0], p[1]
    # print("quantile", min_val, max_val)
    # Minimum power of 2 required to represent all tensor values
    scale = 2**min_pow_2_scale(min_val.item(), max_val.item(), bit_width)
    #print("quant tensor:", scale)
    tensor = quantize_tensor(input, bit_width, scale)

    # Sanity check that quantization works properly
    m = torch.max(torch.abs(tensor - input))
    #print(tensor - input)
    #assert m < scale, f"{m}, {bit_width}"
    return tensor


# Sometimes numpy's comparisons are different to python's comparisons,
# which triggers the assertion error
# see https://stackoverflow.com/questions/55944977/less-than-or-equal-to-operator-for-numpy-floating-points
def approx_le(x, y):
    return x <= y or np.isclose(x, y)

def get_bin_for_percentile(cdf: np.ndarray, percentile: float, is_top: bool):
    bin_1 = len(cdf[cdf < percentile]) - 1
    #print(bin_1)
    bin_2 = bin_1 + 1
    # is it closer to bin 1?
    bin = 0
    if bin_1 < 0:
        bin = bin_2
    else:
        u_val = cdf[bin_2] if bin_2 < len(cdf) else 1.
        l_val = cdf[bin_1]
        assert approx_le(l_val, percentile) and approx_le(percentile, u_val),\
             f"{bin_1} => {l_val}, {percentile}, {bin_2} => {u_val}"

        if percentile - l_val < u_val - percentile:
            bin = bin_1
        else:
            bin = bin_2

    if not is_top:
        bin += 1
    return bin


def decide_bounds_percentile(histogram: np.ndarray, tail: float):
    cdf = np.cumsum(histogram) / np.sum(histogram)
    #print(cdf)
    min_bin = get_bin_for_percentile(cdf, tail, False)
    max_bin = get_bin_for_percentile(cdf, 1. - tail, True)

    return (min_bin, max_bin)

def decide_bounds_min_max(histogram: np.ndarray):
    min_bin = np.nonzero(histogram)[0][0]
    max_bin = np.nonzero(histogram)[0][-1]

    return (min_bin, max_bin)