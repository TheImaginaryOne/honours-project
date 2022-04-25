import torch, torchvision
import numpy as np
from torch import nn
from typing import List, cast
from lib.math import min_pow_2

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


def quantize_tensor_min_max(input: torch.Tensor, bit_width: int) -> torch.Tensor:
    min_val, max_val = torch.aminmax(input)
    # Minimum power of 2 required to represent all tensor values
    scale = 2**min_pow_2_scale(min_val.item(), max_val.item(), bit_width)
    tensor = quantize_tensor(input, bit_width, scale)

    # Sanity check that quantization works properly
    m = torch.max(torch.abs(tensor - input))
    #print(tensor - input)
    assert m < scale, f"{m}, {bit_width}"
    return tensor

class FakeQuantize(nn.Module):
    def __init__(self, bit_width: int = 8, scale: int = 1):
        super().__init__()
        self.bit_width = bit_width
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        min_val, max_val = -2**(self.bit_width-1), 2**(self.bit_width - 1) - 1
        # Use "floor" function to simulate bit truncate
        torch.floor(input / self.scale, out=input)
        torch.clamp(input, min_val, max_val, out=input)
        input *= self.scale
        #print(self.scale, self.bit_width)
        return input

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

class QuantConfig:
    def __init__(self, activation_bit_widths: List[int], weight_bit_widths: List[tuple[int, int]]):
        self.activation_bit_widths = activation_bit_widths
        self.weight_bit_widths = weight_bit_widths

def setup_quant_net(net: torchvision.models.vgg.VGG, activation_histograms: List[tuple[int, np.ndarray]], quant_config: QuantConfig) -> QuantisedVgg:
    quant_net = make_vgg11()

    fake_quants = [quant_net.quantize] + [cast(VggUnit, unit).quantize for unit in quant_net.features] \
            + [cast(FakeQuantize, quant_net.avgpool[1])] \
            + [cast(FakeQuantize, quant_net.classifier[i]) for i in [1, 4, 7]]

    i = 0
    # set fake_quant layers
    for (pow_2, histogram), fake_quant in zip(activation_histograms, fake_quants):
        #print(pow_2)
        #print(np.nonzero(histogram)[0][-1])
        max_val = (np.nonzero(histogram)[0][-1] - len(histogram) // 2 + 1) / len(histogram) * 2**(pow_2 + 1)
        #print(max_val)
        min_val = (np.nonzero(histogram)[0][0] - len(histogram) // 2) / len(histogram) * 2**(pow_2 + 1)

        fake_quant.bit_width = quant_config.activation_bit_widths[i]
        scale = 2**min_pow_2_scale(min_val, max_val, quant_config.activation_bit_widths[i])
        fake_quant.scale = scale
        #print("-", pow_2)
        i += 1

    output_feature_layers = [net.features[i] for i in [0, 3, 6, 8, 11, 13, 16, 18]]
    classifier_layers = [net.classifier[i] for i in [0, 3, 6]]

    # copy weights
    for i, layer in enumerate(output_feature_layers):
        quant_net.features[i].conv2d.weight = nn.Parameter(quantize_tensor_min_max(layer.weight, quant_config.weight_bit_widths[i][0]))
        quant_net.features[i].conv2d.bias = nn.Parameter(quantize_tensor_min_max(layer.bias, quant_config.weight_bit_widths[i][1]))
    for i, layer, j in zip([0, 3, 6], classifier_layers, range(len(output_feature_layers), len(output_feature_layers) + 3)):
        # just an offset
        quant_net.classifier[i].weight = nn.Parameter(quantize_tensor_min_max(layer.weight, quant_config.weight_bit_widths[j][0]))
        quant_net.classifier[i].bias = nn.Parameter(quantize_tensor_min_max(layer.bias, quant_config.weight_bit_widths[j][1]))

    return quant_net

def test_quant(net: torchvision.models.vgg.VGG, activation_histograms: List[tuple[int, np.ndarray]], images: torch.utils.data.Dataset, quant_config_name: str):
    import tqdm
    configs = {'8b': QuantConfig([8] * 13, [(8,8)] * 11), '7b': QuantConfig([7] * 13, [(7,7)] * 11)}
    if quant_config_name not in configs:
        print("Invalid configuration, try one of", configs.keys())
        return
    config = configs[quant_config_name]
    with torch.no_grad():
        #config = QuantConfig([4] * 13, [(4,4)] * 11)
        #config = QuantConfig([6] * 13, [(6,6)] * 11)

        quant_net = setup_quant_net(net, activation_histograms, config)
        all_preds = []


        loader = torch.utils.data.DataLoader(images, batch_size=10)
        # evaluate networ
        with torch.no_grad():
            for X in tqdm.tqdm(loader):
                preds = quant_net(X)
                # convert output to numpy
                preds_np = preds.cpu().detach().numpy()
                all_preds.append(preds_np)

        with open(f'output/quantpreds_{quant_config_name}.npy', 'wb') as f:
            concated = np.concatenate(all_preds)
            print(concated.shape)
            np.save(f, concated)
