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
    if min_val <= 0:
        x = min_pow_2(min_val / (-2**(bit_width-1) - 1./2))
    else:
        x = -126
    
    if max_val >= 0:
        y = min_pow_2(max_val / (2**(bit_width-1) - 1./2))
    else:
        y = -126
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
