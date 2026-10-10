"""Input compatibility for the central 3.5-second vowels experiment."""

from pathlib import Path

import numpy as np
import pytest
import torch

from neuroevolution import config as config_module
from neuroevolution.config import (
    VOWELS_DATASET_CONFIG,
    VOWELS_HALF_DATASET_CONFIG,
    configure_vowels_input,
    get_default_config,
)
from neuroevolution.models.evolvable_cnn import EvolvableCNN


@pytest.mark.parametrize('modality,shape', [('audio', (1, 154350)), ('spectrogram', (64, 348))])
def test_cropped_inputs_load_and_run_in_both_modalities(monkeypatch, tmp_path, modality, shape):
    monkeypatch.setattr(config_module, 'DEFAULT_INPUT_MODALITY', modality)
    config = get_default_config()
    assert Path(config['data_path']).name == 'vowels_dataset_half'
    assert 'test_vowels_half_' in config['artifacts_dir']
    assert config['audio_duration_s'] == 3.5
    assert config['sequence_length'] == shape[-1]
    directory = tmp_path / config['fold_files_subdirectory']
    directory.mkdir()
    np.save(directory / 'X_train_vowels_fold_1.npy', np.zeros((2, *shape), dtype=np.float32))
    config['data_path'] = str(tmp_path)
    assert configure_vowels_input(config) is config

    genome = {
        'id': 'cropped-input-smoke', 'num_conv_layers': 1, 'num_fc_layers': 1,
        'filters': [4], 'kernel_sizes': [3], 'fc_nodes': [8],
        'activations': ['relu'], 'dropout_rate': 0.0,
        'learning_rate': 0.001, 'optimizer': 'adam', 'normalization_type': 'batch',
    }
    model = EvolvableCNN(genome, config).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, *shape)).shape == (2, 2)


@pytest.mark.parametrize('modality,old_shape', [('audio', (1, 286650)), ('spectrogram', (64, 648))])
def test_original_arrays_are_rejected_for_cropped_experiment(monkeypatch, tmp_path, modality, old_shape):
    monkeypatch.setattr(config_module, 'DEFAULT_INPUT_MODALITY', modality)
    config = get_default_config()
    config['data_path'] = str(tmp_path)
    directory = tmp_path / config['fold_files_subdirectory']
    directory.mkdir()
    np.save(directory / 'X_train_vowels_fold_1.npy', np.zeros((1, *old_shape), dtype=np.float32))
    with pytest.raises(ValueError, match='Unexpected .* input shape'):
        configure_vowels_input(config)


def test_missing_arrays_point_to_cropped_preparation_notebook(tmp_path):
    config = get_default_config()
    config['data_path'] = str(tmp_path)
    with pytest.raises(FileNotFoundError, match='vowels_dataset_por_persona_half.ipynb'):
        configure_vowels_input(config)


def test_original_preparation_keeps_its_output_directory():
    assert Path(VOWELS_DATASET_CONFIG['output_root']).name == 'vowels_dataset'
    assert Path(VOWELS_HALF_DATASET_CONFIG['output_root']).name == 'vowels_dataset_half'
    rate = VOWELS_HALF_DATASET_CONFIG['sample_rate_hz']
    kept_samples = round((6.5 - 1.5 - 1.5) * rate)
    assert VOWELS_HALF_DATASET_CONFIG['audio_samples'] == kept_samples
    frames = 1 + (kept_samples - VOWELS_HALF_DATASET_CONFIG['n_fft']) // VOWELS_HALF_DATASET_CONFIG['hop_length']
    assert VOWELS_HALF_DATASET_CONFIG['spectrogram_frames'] == frames
