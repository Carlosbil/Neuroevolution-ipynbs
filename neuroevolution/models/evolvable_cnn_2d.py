"""Evolvable 2D CNN for time-frequency spectrograms."""

import torch
import torch.nn as nn

from neuroevolution.config import ACTIVATION_FUNCTIONS
from .genome_validator import (
    calculate_inception_branch_channels,
    calculate_inception_reduction_channels,
    validate_and_fix_genome,
)


class ChannelLayerNorm2D(nn.Module):
    """Normalize channels of an (N, C, frequency, time) tensor."""

    def __init__(self, channels: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Conv2DUnit(nn.Module):
    """Same-size 2D convolution, normalization and activation."""

    def __init__(self, in_channels, out_channels, kernel_size, activation_name,
                 normalization_type, minimum_kernel_size=3):
        super().__init__()
        kernel_size = max(minimum_kernel_size, kernel_size + (kernel_size % 2 == 0))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = (
            ChannelLayerNorm2D(out_channels) if normalization_type == 'layer'
            else nn.BatchNorm2d(out_channels)
        )
        self.activation = ACTIVATION_FUNCTIONS[activation_name]()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class ResidualConv2DBlock(nn.Module):
    """Residual stack with one spatial pool after the whole block."""

    def __init__(self, in_channels, filters, kernel_sizes, activation_names,
                 normalization_type, residual_projection='auto', apply_dropout=False):
        super().__init__()
        units = []
        current_channels = in_channels
        for out_channels, kernel, activation in zip(filters, kernel_sizes, activation_names):
            units.append(Conv2DUnit(current_channels, out_channels, kernel, activation, normalization_type))
            current_channels = out_channels
        self.units = nn.Sequential(*units)
        if in_channels == current_channels:
            self.shortcut = nn.Identity()
        elif residual_projection == 'auto':
            self.shortcut = nn.Conv2d(in_channels, current_channels, 1)
        else:
            raise ValueError(f'Unsupported residual projection: {residual_projection}')
        self.final_activation = ACTIVATION_FUNCTIONS[activation_names[-1]]()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.1) if apply_dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.pool(self.final_activation(self.units(x) + self.shortcut(x))))


class InceptionConv2DModule(nn.Module):
    """Four optional same-size branches over local time-frequency regions."""

    def __init__(self, in_channels, out_channels, wide_kernel_size, activation_name,
                 normalization_type, reduction_ratio=0.5, pool_branch=True,
                 min_branch_channels=1):
        super().__init__()
        channels = calculate_inception_branch_channels(
            out_channels, pool_branch=pool_branch, min_branch_channels=min_branch_channels,
        )
        reduced = calculate_inception_reduction_channels(
            in_channels, reduction_ratio, min_branch_channels=min_branch_channels,
        )
        self.branches = nn.ModuleDict({
            'pointwise': Conv2DUnit(in_channels, channels['pointwise'], 1,
                                    activation_name, normalization_type, minimum_kernel_size=1),
            'medium': nn.Sequential(
                Conv2DUnit(in_channels, reduced, 1, activation_name,
                           normalization_type, minimum_kernel_size=1),
                Conv2DUnit(reduced, channels['medium'], 3, activation_name, normalization_type),
            ),
            'wide': nn.Sequential(
                Conv2DUnit(in_channels, reduced, 1, activation_name,
                           normalization_type, minimum_kernel_size=1),
                Conv2DUnit(reduced, channels['wide'], wide_kernel_size,
                           activation_name, normalization_type, minimum_kernel_size=5),
            ),
        })
        if pool_branch:
            self.branches['pool'] = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                Conv2DUnit(in_channels, channels['pool'], 1, activation_name,
                           normalization_type, minimum_kernel_size=1),
            )

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches.values()], dim=1)


class EvolvableCNN2D(nn.Module):
    """CNN using 2D kernels on frequency and time; input may omit channel 1."""

    def __init__(self, genome: dict, config: dict):
        super().__init__()
        self.config = config
        self.genome = validate_and_fix_genome(genome, config)
        self.conv_layers = self._build_conv_layers()
        pool_shape = tuple(config.get('pre_fc_pool_shape') or (4, 8))
        self.pre_fc_pool = nn.AdaptiveAvgPool2d(pool_shape)
        self.conv_output_size = self._calculate_conv_output_size()
        self.fc_layers = self._build_fc_layers()

    def _build_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = int(self.config['num_channels'])
        norm = self.genome.get('normalization_type', 'batch')
        topology = self.genome.get('conv_topology', 'sequential')
        count = self.genome['num_conv_layers']
        activations = self.genome['activations']

        if topology == 'inception':
            for i in range(count):
                out_channels = self.genome['filters'][i]
                layers.append(InceptionConv2DModule(
                    in_channels, out_channels, self.genome['kernel_sizes'][i],
                    activations[i % len(activations)], norm,
                    reduction_ratio=self.genome.get('inception_reduction_ratio', 0.5),
                    pool_branch=self.genome.get('inception_pool_branch', True),
                    min_branch_channels=int(self.config.get('inception_min_branch_channels', 1)),
                ))
                layers.append(nn.MaxPool2d(2, 2))
                if i < count - 1:
                    layers.append(nn.Dropout2d(0.1))
                in_channels = out_channels
        elif topology == 'residual':
            block_size = int(self.genome.get('residual_block_size', 2))
            index = 0
            while index < count:
                end = min(count, index + block_size)
                if end - index == 1:
                    out_channels = self.genome['filters'][index]
                    layers.append(Conv2DUnit(
                        in_channels, out_channels, self.genome['kernel_sizes'][index],
                        activations[index % len(activations)], norm,
                    ))
                    layers.append(nn.MaxPool2d(2, 2))
                else:
                    out_channels = self.genome['filters'][end - 1]
                    layers.append(ResidualConv2DBlock(
                        in_channels, self.genome['filters'][index:end],
                        self.genome['kernel_sizes'][index:end],
                        [activations[i % len(activations)] for i in range(index, end)],
                        norm, residual_projection=self.genome.get('residual_projection', 'auto'),
                        apply_dropout=end < count,
                    ))
                in_channels = out_channels
                index = end
        else:
            for i in range(count):
                out_channels = self.genome['filters'][i]
                layers.append(Conv2DUnit(
                    in_channels, out_channels, self.genome['kernel_sizes'][i],
                    activations[i % len(activations)], norm,
                ))
                layers.append(nn.MaxPool2d(2, 2))
                if i < count - 1:
                    layers.append(nn.Dropout2d(0.1))
                in_channels = out_channels
        return layers

    def _calculate_conv_output_size(self):
        frequency_bins = int(self.config['num_frequency_bins'])
        time_frames = int(self.config['sequence_length'])
        x = torch.zeros(1, self.config['num_channels'], frequency_bins, time_frames)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for layer in self.conv_layers:
                    x = layer(x)
                x = self.pre_fc_pool(x)
        finally:
            self.train(was_training)
        return int(x.numel())

    def _build_fc_layers(self):
        layers = nn.ModuleList()
        input_size = self.conv_output_size
        norm = self.genome.get('normalization_type', 'batch')
        for i, nodes in enumerate(self.genome['fc_nodes'][:self.genome['num_fc_layers']]):
            layers.append(nn.Linear(input_size, nodes))
            layers.append(nn.LayerNorm(nodes) if norm == 'layer' else nn.BatchNorm1d(nodes))
            layers.append(nn.ReLU())
            if i < self.genome['num_fc_layers'] - 1:
                layers.append(nn.Dropout(self.genome['dropout_rate']))
            input_size = nodes
        layers.append(nn.Linear(input_size, self.config['num_classes']))
        return layers

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != self.config['num_channels']:
            raise ValueError('Spectrogram input must have shape (N, frequency, time) or (N, 1, frequency, time)')
        for layer in self.conv_layers:
            x = layer(x)
        x = self.pre_fc_pool(x).reshape(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
        return x

    def get_architecture_summary(self) -> str:
        return (
            f"Conv2D Layers: {self.genome['num_conv_layers']} | "
            f"Filters: {self.genome['filters']} | "
            f"Kernel Sizes: {self.genome['kernel_sizes']} | "
            f"FC Layers: {self.genome['num_fc_layers']} | "
            f"Topology: {self.genome.get('conv_topology', 'sequential')}"
        )
