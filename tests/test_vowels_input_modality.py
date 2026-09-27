import copy

import pytest
import torch

from neuroevolution.config import get_default_config
from neuroevolution.models.evolvable_cnn import EvolvableCNN
from neuroevolution.models.evolvable_cnn_1d import EvolvableCNN as EvolvableCNN1D
from neuroevolution.models.evolvable_cnn_2d import EvolvableCNN2D
from neuroevolution.models.genome_validator import (
    architecture_spatial_limit,
    estimate_genome_parameter_count,
    is_genome_valid,
)


def test_adaptive_pool_keeps_raw_audio_dense_layer_bounded():
    config = get_default_config()
    config.update({
        'num_channels': 1,
        'sequence_length': 128,
        'pre_fc_pool_length': 8,
        'min_conv_layers': 1,
        'max_conv_layers': 2,
        'min_fc_layers': 1,
        'max_fc_layers': 1,
        'min_filters': 4,
        'max_filters': 4,
        'min_fc_nodes': 8,
        'max_fc_nodes': 8,
    })
    genome = {
        'id': 'audio-smoke',
        'num_conv_layers': 1,
        'num_fc_layers': 1,
        'filters': [4],
        'kernel_sizes': [3],
        'fc_nodes': [8],
        'activations': ['relu'],
        'dropout_rate': 0.0,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'normalization_type': 'batch',
    }
    model = EvolvableCNN(copy.deepcopy(genome), config)
    assert isinstance(model, EvolvableCNN1D)
    assert model(torch.randn(2, 1, 128)).shape == (2, 2)
    assert model.conv_output_size == 4 * 8
    assert estimate_genome_parameter_count(model.genome, config) == sum(
        parameter.numel() for parameter in model.parameters()
    )


@pytest.mark.parametrize('topology', ['sequential', 'residual', 'inception'])
@pytest.mark.parametrize('normalization', ['batch', 'layer'])
def test_spectrogram_uses_conv2d_and_counts_parameters(topology, normalization):
    config = get_default_config()
    config.update({
        'input_modality': 'spectrogram',
        'num_channels': 1,
        'num_frequency_bins': 16,
        'sequence_length': 32,
        'pre_fc_pool_shape': (2, 4),
        'min_conv_layers': 1,
        'max_conv_layers': 3,
        'min_fc_layers': 1,
        'max_fc_layers': 1,
        'min_filters': 4,
        'max_filters': 8,
        'min_fc_nodes': 8,
        'max_fc_nodes': 8,
    })
    genome = {
        'id': 'mel-smoke',
        'num_conv_layers': 2,
        'num_fc_layers': 1,
        'filters': [4, 8],
        'kernel_sizes': [3, 5],
        'fc_nodes': [8],
        'activations': ['relu', 'relu'],
        'dropout_rate': 0.0,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'normalization_type': normalization,
        'conv_topology': topology,
    }
    model = EvolvableCNN(copy.deepcopy(genome), config)
    assert isinstance(model, EvolvableCNN2D)
    assert any(isinstance(module, torch.nn.Conv2d) for module in model.modules())
    features = torch.randn(2, 16, 32)
    assert model(features).shape == (2, 2)
    assert model(features.unsqueeze(1)).shape == (2, 2)
    assert model.conv_output_size == 8 * 2 * 4
    assert estimate_genome_parameter_count(model.genome, config) == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert is_genome_valid(model.genome, config)


def test_spectrogram_depth_is_limited_by_frequency_axis():
    config = get_default_config()
    config.update({
        'input_modality': 'spectrogram',
        'num_channels': 1,
        'num_frequency_bins': 16,
        'sequence_length': 648,
        'max_conv_layers': 8,
        'max_model_parameters': None,
    })
    genome = {
        'num_conv_layers': 4,
        'num_fc_layers': 1,
        'filters': [8] * 4,
        'kernel_sizes': [3] * 4,
        'fc_nodes': [8],
        'activations': ['relu'] * 4,
        'normalization_type': 'batch',
    }
    assert architecture_spatial_limit(config) == 16
    assert not is_genome_valid(genome, config)
