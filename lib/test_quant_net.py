import torch
from quantnet import quantize_tensor, quantize_tensor_min_max

def test_quant():
    tensor = torch.Tensor([-1.7, 5.6, 7.1, -7.8, -9.1])

    # 3 bit quant; scale factor 2^1. Range from -8 to 6
    quantized = quantize_tensor(tensor, 3, 2)
    assert torch.equal(torch.Tensor([-2, 6, 6, -8, -8]), quantized)

def test_quant_min_max():
    tensor = torch.Tensor([-1.7, 5.6, 7.1, -7.8, -9.1])

    quantized = quantize_tensor_min_max(tensor, 3)
    assert torch.equal(torch.Tensor([-2, 6, 6, -8, -8]), quantized)
