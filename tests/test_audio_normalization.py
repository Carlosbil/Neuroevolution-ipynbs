"""Regression coverage for old-data audio normalization and singleton batches."""

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from neuroevolution import config as config_module
from neuroevolution.config import get_default_config
from neuroevolution.models.evolvable_cnn import EvolvableCNN
from neuroevolution.models.evolvable_cnn_1d import ChannelLayerNorm1d
from neuroevolution.models.normalization import SingletonSafeBatchNorm1d


@pytest.fixture(autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def audio_genome(normalization):
    return {
        'id': 'audio-normalization-regression',
        'num_conv_layers': 1, 'num_fc_layers': 1,
        'filters': [8], 'kernel_sizes': [3], 'fc_nodes': [16],
        'activations': ['relu'], 'dropout_rate': 0.0,
        'learning_rate': 0.001, 'optimizer': 'adam',
        'normalization_type': normalization,
    }


@pytest.mark.parametrize('normalization', ['batch', 'layer'])
@pytest.mark.parametrize('modality', ['audio', 'spectrogram'])
@pytest.mark.parametrize('topology', ['sequential', 'residual', 'inception'])
def test_old_dataset_training_keeps_singleton_last_batch(monkeypatch, modality, normalization, topology):
    monkeypatch.setattr(config_module, 'DEFAULT_DATASET', 'old_vowels')
    monkeypatch.setattr(config_module, 'DEFAULT_INPUT_MODALITY', modality)
    config = get_default_config()
    shape = (1, 11520) if modality == 'audio' else (64, 44)
    dataset = TensorDataset(torch.randn(3, *shape), torch.tensor([0, 1, 0]))
    loader = DataLoader(dataset, batch_size=2)
    genome = audio_genome(normalization)
    genome['conv_topology'] = topology
    if topology != 'sequential':
        genome.update(num_conv_layers=2, filters=[8, 8], kernel_sizes=[3, 5])
    model = EvolvableCNN(genome, config).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sizes = []
    for x, y in loader:
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        sizes.append(len(y))
    assert sizes == [2, 1]
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(dataset.tensors[0][-1:])).all()


@pytest.mark.parametrize('channels', [8, 15])
def test_audio_layer_norm_uses_channels_at_each_time_step(channels):
    x = torch.randn(1, channels, 11520, requires_grad=True)
    layer = ChannelLayerNorm1d(channels)
    actual = layer(x)
    expected = torch.nn.functional.layer_norm(x.transpose(1, 2), (channels,)).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_batch_norm_preserves_standard_batches_and_checkpoint_state():
    standard = torch.nn.BatchNorm1d(4)
    safe = SingletonSafeBatchNorm1d(4)
    safe.load_state_dict(standard.state_dict(), strict=True)
    x = torch.randn(2, 4)
    torch.testing.assert_close(safe(x), standard(x))
    for key, value in standard.state_dict().items():
        torch.testing.assert_close(safe.state_dict()[key], value)

    snapshot = {key: value.clone() for key, value in safe.state_dict().items()}
    standard.eval()
    singleton = torch.randn(1, 4, requires_grad=True)
    actual = safe(singleton)
    torch.testing.assert_close(actual, standard(singleton))
    actual.square().sum().backward()
    assert singleton.grad is not None and torch.isfinite(singleton.grad).all()
    assert safe.weight.grad is not None and safe.bias.grad is not None
    assert safe.training
    for key, value in safe.state_dict().items():
        torch.testing.assert_close(value, snapshot[key])
    standard.load_state_dict(safe.state_dict(), strict=True)


@pytest.mark.parametrize('normalization', ['batch', 'layer'])
def test_evolution_fold_trains_audio_with_odd_sample_count(monkeypatch, normalization):
    from neuroevolution.evolution import fitness

    monkeypatch.setattr(config_module, 'DEFAULT_DATASET', 'old_vowels')
    monkeypatch.setattr(config_module, 'DEFAULT_INPUT_MODALITY', 'audio')
    config = get_default_config()
    config.update(num_epochs=1, epoch_patience=1, fold_parallel_workers=1, use_amp=False)
    train = DataLoader(TensorDataset(torch.randn(3, 1, 11520), torch.tensor([0, 1, 0])), batch_size=2)
    val = DataLoader(TensorDataset(torch.randn(2, 1, 11520), torch.tensor([0, 1])), batch_size=2)
    monkeypatch.setattr(fitness, 'load_fold_data', lambda *args, **kwargs: (train, val))
    fold, score, model, metrics = fitness.train_fold_in_thread(
        copy.deepcopy(audio_genome(normalization)), 1, config, torch.device('cpu'))
    assert fold == 1 and model is not None and metrics is not None
    assert metrics['best_epoch'] == 1
    assert 0 <= score <= 100


@pytest.mark.parametrize('normalization', ['batch', 'layer'])
def test_final_evaluation_trains_audio_with_odd_sample_count(monkeypatch, normalization):
    from neuroevolution.evaluation.cross_validation import evaluate_single_fold

    monkeypatch.setattr(config_module, 'DEFAULT_DATASET', 'old_vowels')
    monkeypatch.setattr(config_module, 'DEFAULT_INPUT_MODALITY', 'audio')
    config = get_default_config()
    config.update(num_epochs=1, epoch_patience=1, use_amp=False)
    train = DataLoader(TensorDataset(torch.randn(3, 1, 11520), torch.tensor([0, 1, 0])), batch_size=2)
    val = DataLoader(TensorDataset(torch.randn(2, 1, 11520), torch.tensor([0, 1])), batch_size=2)
    test = DataLoader(TensorDataset(torch.randn(2, 1, 11520), torch.tensor([0, 1])), batch_size=2)
    result = evaluate_single_fold(
        audio_genome(normalization), config, train, val, test, 1, torch.device('cpu'), num_epochs=1)
    assert result['best_epoch'] == 1
    assert result['n_samples'] == 2
    assert result['selection_split'] == 'validation'
    assert result['evaluation_split'] == 'test'
