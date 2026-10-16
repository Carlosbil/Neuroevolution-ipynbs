"""Hidden dense activations must follow the genome in every training entry point."""

import copy
import json

import pytest
import torch
from torch import nn

from neuroevolution.config import get_default_config, get_winner_replay_config
from neuroevolution.models.architecture_formatting import format_fc_activations
from neuroevolution.models.evolvable_cnn import EvolvableCNN


@pytest.mark.parametrize('config_factory', [get_default_config, get_winner_replay_config])
@pytest.mark.parametrize('modality', ['audio', 'spectrogram'])
@pytest.mark.parametrize('topology', ['sequential', 'residual', 'inception'])
def test_hidden_dense_layers_use_genome_activations(config_factory, modality, topology):
    config = config_factory()
    assert config['fc_activation_source'] == 'genome'
    config.update(
        input_modality=modality, num_channels=1, num_frequency_bins=16,
        sequence_length=32, pre_fc_pool_shape=(2, 2), pre_fc_pool_length=4,
        min_conv_layers=1, max_conv_layers=2, min_fc_layers=1, max_fc_layers=4,
        min_filters=4, max_filters=8, min_fc_nodes=8, max_fc_nodes=8,
    )
    genome = {
        'id': 'activation-regression', 'num_conv_layers': 2, 'num_fc_layers': 4,
        'filters': [4, 8], 'kernel_sizes': [3, 5], 'fc_nodes': [8] * 4,
        'activations': ['relu', 'tanh', 'leaky_relu'], 'dropout_rate': 0.0,
        'learning_rate': 0.001, 'optimizer': 'adam', 'normalization_type': 'batch',
        'conv_topology': topology,
    }
    model = EvolvableCNN(copy.deepcopy(genome), config).eval()
    activation_types = (nn.ReLU, nn.Tanh, nn.LeakyReLU, nn.Sigmoid, nn.SELU)
    assert [type(layer) for layer in model.fc_layers if isinstance(layer, activation_types)] == [
        nn.ReLU, nn.Tanh, nn.LeakyReLU, nn.ReLU,
    ]
    assert format_fc_activations(model.genome, config) == '[relu, tanh, leaky_relu, relu]'
    assert isinstance(model.fc_layers[-1], nn.Linear)
    features = torch.randn(2, 1, 32) if modality == 'audio' else torch.randn(2, 16, 32)
    with torch.no_grad():
        logits = model(features)
    assert logits.shape == (2, config['num_classes'])
    assert torch.isfinite(logits).all()


def test_legacy_results_keep_their_explicit_activation_setting():
    genome = {'num_fc_layers': 3, 'activations': ['relu', 'tanh', 'leaky_relu']}
    assert format_fc_activations(genome, {'fc_activation_source': 'relu'}) == '[relu, relu, relu]'
    assert format_fc_activations(genome, {}) == '[relu, tanh, leaky_relu]'


def test_corrected_winner_pass_has_a_separate_output_directory():
    config = get_winner_replay_config()
    assert config['artifact_dir'] == config['artifacts_dir']
    assert config['artifacts_dir'] != get_default_config()['artifacts_dir']
    assert config['artifacts_dir'].endswith('_genome_activations')
    assert 'winner_68e15b97' in config['artifacts_dir']


def test_report_reconstructs_saved_run_with_its_training_config(monkeypatch, tmp_path):
    from scripts import report_winner_scientific as report

    config = get_default_config()
    config.update(fc_activation_source='relu', torch_num_threads=1)
    run = {'status': 'completed', 'config': config}
    run_path = tmp_path / 'results.json'
    run_path.write_text(json.dumps(run), encoding='utf-8')
    (tmp_path / 'scientific_results.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(report, 'get_scientific_evaluation_config', lambda: {
        'run_results_path': str(run_path), 'report_directory': str(tmp_path),
    })
    captured = []
    monkeypatch.setattr(report, 'finalize_report', lambda run, results, output, config, settings: captured.append(config))
    report.main()
    assert captured == [json.loads(json.dumps(config))]
    assert get_winner_replay_config()['fc_activation_source'] == 'genome'
