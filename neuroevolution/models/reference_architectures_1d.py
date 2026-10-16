"""Temporal copies of the reference graphs, without image reshaping or 2D ops.

Both axes of a factorized image kernel map to time (1x7 and 7x1 become
independent length-7 convolutions). Adaptive grids become temporal bins, so
only the first dense layer's input size changes; hidden widths are preserved.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.densenet import DenseNet
from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.models.googlenet import InceptionAux as GoogLeNetAux
from torchvision.models.inception import InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux

from .reference_architectures import LeNet5, LeNetSparseC3, LeNetSubsampling


def _temporal(value):
    return max(value) if isinstance(value, (tuple, list)) else value


class _Graph1D(nn.Module):
    def __init__(self, source):
        super().__init__()
        for name, child in source.named_children():
            self.add_module(name, child)


class LeNetSubsampling1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        from .reference_architectures import ScaledTanh
        self.activation = ScaledTanh()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.activation(F.avg_pool1d(x, 2) * self.weight + self.bias)


class LeNetSparseC3_1D(_Graph1D):
    connections = LeNetSparseC3.connections

    def forward(self, x):
        return self.activation(torch.cat([
            unit(x[:, indices]) for unit, indices in zip(self.units, self.connections)
        ], dim=1))


class LeNet5_1D(_Graph1D):
    def forward(self, x):
        x = self.s2(self.activation(self.conv1(x)))
        x = self.s4(self.c3(x))
        x = F.adaptive_avg_pool1d(self.activation(self.c5(x)), 1).flatten(1)
        return self.classifier(self.activation(self.f6(x)))


class DenseNet1D(_Graph1D):
    def forward(self, x):
        x = F.relu(self.features(x), inplace=True)
        return self.classifier(F.adaptive_avg_pool1d(x, 1).flatten(1))


class MobileNetV2_1D(_Graph1D):
    def forward(self, x):
        return self.classifier(F.adaptive_avg_pool1d(self.features(x), 1).flatten(1))


class GoogLeNetAux1D(_Graph1D):
    def __init__(self, source):
        super().__init__(source)
        self.fc1 = nn.Linear(source.conv.conv.out_channels * 4, source.fc1.out_features)

    def forward(self, x):
        x = self.conv(F.adaptive_avg_pool1d(x, 4)).flatten(1)
        return self.fc2(self.dropout(F.relu(self.fc1(x), inplace=True)))


class InceptionAux1D(_Graph1D):
    def forward(self, x):
        x = self.conv1(self.conv0(F.avg_pool1d(x, kernel_size=5, stride=3)))
        return self.fc(F.adaptive_avg_pool1d(x, 1).flatten(1))


class InceptionBranches1D(_Graph1D):
    """Keep every native branch and concatenation, using temporal pooling."""

    def __init__(self, source):
        super().__init__(source)
        self.kind = type(source).__name__

    def forward(self, x):
        if self.kind == 'InceptionA':
            branches = [self.branch1x1(x), self.branch5x5_2(self.branch5x5_1(x)),
                        self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))]
        elif self.kind == 'InceptionB':
            return torch.cat([
                self.branch3x3(x),
                self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x))),
                F.max_pool1d(x, kernel_size=3, stride=2),
            ], dim=1)
        elif self.kind == 'InceptionC':
            branches = [
                self.branch1x1(x), self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x))),
                self.branch7x7dbl_5(self.branch7x7dbl_4(self.branch7x7dbl_3(
                    self.branch7x7dbl_2(self.branch7x7dbl_1(x))))),
            ]
        elif self.kind == 'InceptionD':
            return torch.cat([
                self.branch3x3_2(self.branch3x3_1(x)),
                self.branch7x7x3_4(self.branch7x7x3_3(self.branch7x7x3_2(self.branch7x7x3_1(x)))),
                F.max_pool1d(x, kernel_size=3, stride=2),
            ], dim=1)
        elif self.kind == 'InceptionE':
            branch3 = self.branch3x3_1(x)
            branch_double = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
            branches = [self.branch1x1(x),
                        torch.cat([self.branch3x3_2a(branch3), self.branch3x3_2b(branch3)], dim=1),
                        torch.cat([self.branch3x3dbl_3a(branch_double),
                                   self.branch3x3dbl_3b(branch_double)], dim=1)]
        else:
            raise ValueError(f'Unsupported Inception block: {self.kind}')
        branches.append(self.branch_pool(F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)))
        return torch.cat(branches, dim=1)


def convert_reference_to_1d(module):
    """Return a separate temporal graph; never patch Torchvision or global functions."""
    if isinstance(module, nn.Conv2d):
        return nn.Conv1d(module.in_channels, module.out_channels, _temporal(module.kernel_size),
                         _temporal(module.stride), _temporal(module.padding), _temporal(module.dilation),
                         module.groups, module.bias is not None, module.padding_mode)
    if isinstance(module, nn.BatchNorm2d):
        return nn.BatchNorm1d(module.num_features, module.eps, module.momentum,
                              module.affine, module.track_running_stats)
    if isinstance(module, nn.MaxPool2d):
        return nn.MaxPool1d(_temporal(module.kernel_size), _temporal(module.stride),
                            _temporal(module.padding), _temporal(module.dilation),
                            module.return_indices, module.ceil_mode)
    if isinstance(module, nn.AvgPool2d):
        if module.divisor_override is not None:
            raise ValueError('Temporal pooling does not support divisor_override')
        return nn.AvgPool1d(_temporal(module.kernel_size), _temporal(module.stride),
                            _temporal(module.padding), module.ceil_mode, module.count_include_pad)
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return nn.AdaptiveAvgPool1d(_temporal(module.output_size))
    if isinstance(module, nn.Dropout2d):
        return nn.Dropout1d(module.p, module.inplace)
    if isinstance(module, LeNetSubsampling):
        return LeNetSubsampling1D(module.weight.shape[1])

    # AlexNet/VGG flatten a fixed image grid before the first hidden layer.
    # Temporal pooling keeps one side of that grid; retain all hidden neurons.
    if isinstance(getattr(module, 'avgpool', None), nn.AdaptiveAvgPool2d) and isinstance(
        getattr(module, 'classifier', None), nn.Sequential
    ):
        bins = module.avgpool.output_size
        if isinstance(bins, (tuple, list)):
            old_area = bins[0] * bins[1]
            for index, layer in enumerate(module.classifier):
                if isinstance(layer, nn.Linear):
                    module.classifier[index] = nn.Linear(
                        layer.in_features // old_area * _temporal(bins), layer.out_features,
                        bias=layer.bias is not None)
                    break

    adapters = {LeNet5: LeNet5_1D, LeNetSparseC3: LeNetSparseC3_1D,
                DenseNet: DenseNet1D, MobileNetV2: MobileNetV2_1D, GoogLeNetAux: GoogLeNetAux1D,
                InceptionAux: InceptionAux1D}
    if type(module) in adapters:
        module = adapters[type(module)](module)
    elif isinstance(module, (InceptionA, InceptionB, InceptionC, InceptionD, InceptionE)):
        module = InceptionBranches1D(module)
    for name, child in list(module.named_children()):
        module._modules[name] = convert_reference_to_1d(child)
    return module
