import torch, torchvision
import numpy as np
from torch import nn
from typing import List, cast
from lib.math import min_pow_2

def quantize_tensor(input: torch.Tensor, bit_width: int, scale: int) -> torch.Tensor:
    """ Fake quantize utility """
    return torch.fake_quantize_per_tensor_affine(input, scale, 0, -2**(bit_width-1), 2**(bit_width - 1) - 1)

def quantize_tensor_min_max(input: torch.Tensor, bit_width: int) -> torch.Tensor:
    min_val, max_val = torch.aminmax(input)
    # Minimum power of 2 required to represent all tensor values
    pow_2 = max(min_pow_2(max_val / (1 - 1 / 2**(bit_width + 1))),
            min_pow_2(min_val / (-1 - 1 / 2**(bit_width + 1))))

    scale = 2**(pow_2 - bit_width + 1)
    tensor = quantize_tensor(input, bit_width, scale)

    # Sanity check that quantization works properly
    m = torch.max(torch.abs(tensor - input))
    #print(tensor - input)
    assert m < scale, f"{m}, {pow_2}, {bit_width}"
    return tensor

class FakeQuantize(nn.Module):
    def __init__(self, bit_width: int = 8, scale: int = 1):
        super().__init__()
        self.bit_width = bit_width
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quantize_tensor(input, self.bit_width, self.scale)

class VggUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, max_pool: bool):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.quantize = FakeQuantize()
        self.relu = nn.ReLU(inplace=True)
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.max_pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        x = self.quantize(x)
        x = self.relu(x)
        if self.max_pool is not None:
            x = self.max_pool(x)
        return x


class QuantisedVgg(nn.Module):
    def __init__(self, features: nn.Sequential):
        super().__init__()
        num_classes = 1000
        self.quantize = FakeQuantize()
        self.features = features
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)), FakeQuantize())
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            FakeQuantize(),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            FakeQuantize(),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
            FakeQuantize(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.quantize(input)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Based on pytorch code
def make_layers(cfg: List[tuple[int, bool]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for out_channels, max_pool in cfg:
        layers.append(VggUnit(in_channels, out_channels, max_pool))
        in_channels = out_channels
    return nn.Sequential(*layers)

def make_vgg11() -> QuantisedVgg:
    layers: List[tuple[int, bool]] = [
            (64, True), 
            (128, True), 
            (256, False),
            (256, True),
            (512, False),
            (512, True),
            (512, False),
            (512, True)]
    return QuantisedVgg(make_layers(layers))

def setup_quant_net(net: torchvision.models.vgg.VGG, activation_histograms: List[tuple[int, np.ndarray]]) -> QuantisedVgg:
    quant_net = make_vgg11()

    bit_width = 8
    fake_quants = [quant_net.quantize] + [cast(VggUnit, unit).quantize for unit in quant_net.features] \
            + [cast(FakeQuantize, quant_net.avgpool[1])] \
            + [cast(FakeQuantize, quant_net.classifier[i]) for i in [1, 4, 7]]

    # set fake_quant layers
    for (pow_2, histogram), fake_quant in zip(activation_histograms, fake_quants):
        #print(pow_2)
        #print(np.nonzero(histogram)[0][-1])
        max_val = (np.nonzero(histogram)[0][-1] - len(histogram) // 2 + 1) / len(histogram) * 2**(pow_2 + 1)
        #print(max_val)
        min_val = (np.nonzero(histogram)[0][0] - len(histogram) // 2) / len(histogram) * 2**(pow_2 + 1)
        pow_2 = max(min_pow_2(max_val / (1 - 1 / 2**(bit_width + 1))),
                min_pow_2(min_val / (-1 - 1 / 2**(bit_width + 1))))

        fake_quant.bit_width = bit_width
        scale = 2**(pow_2 - bit_width + 1)
        fake_quant.scale = scale
        #print("-", pow_2)

    output_feature_layers = [net.features[i] for i in [0, 3, 6, 8, 11, 13, 16, 18]]
    classifier_layers = [net.classifier[i] for i in [0, 3, 6]]

    # copy weights
    for i, layer in enumerate(output_feature_layers):
        quant_net.features[i].conv2d.weight = nn.Parameter(quantize_tensor_min_max(layer.weight, bit_width))
        quant_net.features[i].conv2d.bias = nn.Parameter(quantize_tensor_min_max(layer.bias, bit_width))
    for i, layer in zip([0, 3, 6], classifier_layers):
        quant_net.classifier[i].weight = nn.Parameter(quantize_tensor_min_max(layer.weight, bit_width))
        quant_net.classifier[i].bias = nn.Parameter(quantize_tensor_min_max(layer.bias, bit_width))

    return quant_net

def test_quant(net: torchvision.models.vgg.VGG, activation_histograms: List[tuple[int, np.ndarray]], images: torch.utils.data.Dataset):
    import tqdm
    with torch.no_grad():
        quant_net = setup_quant_net(net, activation_histograms)
        all_preds = []


        loader = torch.utils.data.DataLoader(images, batch_size=20)
        # evaluate networ
        with torch.no_grad():
            for X in tqdm.tqdm(loader):
                preds = quant_net(X)
                # convert output to numpy
                preds_np = preds.cpu().detach().numpy()
                all_preds.append(preds_np)

        with open('output/quantpreds.npy', 'wb') as f:
            concated = np.concatenate(all_preds)
            print(concated.shape)
            np.save(f, concated)
