"""Training entry points use the selected dataset, modality and CSV ordering."""

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from neuroevolution import config as config_module
from neuroevolution.config import (
    configure_vowels_input, get_data_phase_config, get_default_config,
    get_winner_replay_config, validate_config,
)
from neuroevolution.models.evolvable_cnn import EvolvableCNN


@pytest.mark.parametrize('modality,shape,directory', [
    ('audio', (1, 11520), 'arrays_audio'),
    ('spectrogram', (64, 44), 'arrays'),
])
def test_old_profile_and_forward_pass(monkeypatch, tmp_path, modality, shape, directory):
    monkeypatch.setattr(config_module, 'DEFAULT_DATASET', 'old_vowels')
    monkeypatch.setattr(config_module, 'DEFAULT_INPUT_MODALITY', modality)
    config = get_default_config()
    validate_config(config)
    assert Path(config['data_path']).name == 'old_vowels_dataset'
    assert config['audio_sample_rate_hz'] == 24000 and config['audio_duration_s'] == 0.48
    assert config['audio_trim_left_s'] == config['audio_trim_right_s'] == 0
    assert config['sequence_length'] == shape[-1]
    for phase in ('evolution', 'final_evaluation'):
        selected = get_data_phase_config(config, phase)
        assert selected['dataset_id'] == 'old_vowels'
        assert selected['fold_files_subdirectory'] == directory
    replay = get_winner_replay_config()
    assert replay['data_path'] == config['data_path']
    assert f'old_vowels_{modality}' in replay['artifacts_dir']
    arrays = tmp_path / directory
    arrays.mkdir()
    np.save(arrays / 'X_train_old_vowels_fold_1.npy', np.zeros((2, *shape), dtype=np.float32))
    config['data_path'] = str(tmp_path)
    configure_vowels_input(config)
    genome = {
        'id': 'old-input-smoke', 'num_conv_layers': 1, 'num_fc_layers': 1,
        'filters': [4], 'kernel_sizes': [3], 'fc_nodes': [8],
        'activations': ['relu'], 'dropout_rate': 0.0, 'learning_rate': 0.001,
        'optimizer': 'adam', 'normalization_type': 'batch',
    }
    model = EvolvableCNN(genome, config).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, *shape)).shape == (2, 2)


@pytest.mark.parametrize('modality', ['audio', 'spectrogram'])
def test_scientific_loader_checks_selected_representation(tmp_path, modality):
    # Isolate data loading from report-only plotting/statistics dependencies.
    tree = ast.parse(Path('scripts/report_winner_scientific.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'load_split')
    namespace = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<scientific-loader>', 'exec'), namespace)
    audio = np.array([[1, 2, 3]], dtype=np.float32)
    image = np.full((4, 3), 99, dtype=np.float32)
    np.save(tmp_path / 'audio.npy', audio)
    np.save(tmp_path / 'image.npy', image)
    pd.DataFrame([{'label_binary': 0, 'audio_npy_path': 'audio.npy', 'image_npy_path': 'image.npy'}]).to_csv(
        tmp_path / 'test.csv', index=False)
    arrays = tmp_path / 'arrays'
    arrays.mkdir()
    selected = audio if modality == 'audio' else image
    np.save(arrays / 'X_test_old_vowels.npy', selected[None])
    np.save(arrays / 'y_test_old_vowels.npy', np.array([0]))
    rows, x, y = namespace['load_split'](
        tmp_path, {'input_modality': modality, 'dataset_id': 'old_vowels', 'fold_files_subdirectory': 'arrays'}, 'test')
    assert len(rows) == 1 and np.array_equal(x[0], selected) and y.tolist() == [0]


def test_notebook_loads_variable_fold_sizes_and_real_only_global_test(tmp_path):
    notebook = json.loads(Path('test.ipynb').read_text(encoding='utf-8'))
    config = get_default_config()
    # A small spectrogram dataset with fewer variants exercises CSV-based sizes.
    config.update(input_modality='spectrogram', num_frequency_bins=4, sequence_length=8, num_folds=5,
                  data_path=str(tmp_path), fold_files_subdirectory='.', artifacts_dir=str(tmp_path))
    shape = (4, 8)
    records = []
    for label in ('Control', 'Patologicas'):
        for number in range(50):
            split = 'train' if number < 35 else 'val' if number < 40 else 'test'
            for repetition in range(3 if split == 'test' else 2):
                records.append({'person_id': f'{label}_{number:04d}', 'label': label,
                                'label_binary': int(label == 'Patologicas'), 'split': split,
                                'cv_fold': number // 7 if split == 'train' else -1,
                                'origin': 'real' if split == 'test' or repetition == 0 else 'generated'})
    manifest = pd.DataFrame(records)
    def write(rows, name, split, suffix=''):
        rows.to_csv(tmp_path / name, index=False)
        np.save(tmp_path / f'X_{split}_old_vowels{suffix}.npy', np.zeros((len(rows), *shape), dtype=np.float32))
        np.save(tmp_path / f'y_{split}_old_vowels{suffix}.npy', rows.label_binary.to_numpy())
    for split in ('train', 'val', 'test'):
        write(manifest.loc[manifest.split.eq(split)], f'{split}.csv', split)
    for fold in range(5):
        train = manifest.loc[manifest.split.eq('train')]
        for split, mask in [('train', train.cv_fold.ne(fold)), ('val', train.cv_fold.eq(fold))]:
            write(train.loc[mask], f'fold_{fold}_{split}.csv', split, f'_fold_{fold + 1}')
    namespace = {'csv': csv, 'np': np, 'torch': torch, 'DataLoader': DataLoader, 'TensorDataset': TensorDataset,
                 'CONFIG': config, 'DATASET_ROOT': tmp_path, 'ARRAY_ROOT': tmp_path, 'DATASET_ID': 'old_vowels'}
    helper = next(node for node in ast.parse(''.join(notebook['cells'][3]['source'])).body
                  if isinstance(node, ast.FunctionDef) and node.name == 'resolve_best_genome')
    exec(compile(ast.Module(body=[helper], type_ignores=[]), '<winner-resolver>', 'exec'), namespace)
    namespace['best_genome'] = {'id': 'already-selected', 'fitness': 75.0}
    exec(''.join(notebook['cells'][7]['source']), namespace)
    # Only load final data; never launch architecture search or final training in a unit test.
    source = ''.join(notebook['cells'][19]['source']).split('final_test_results = evaluate_single_fold')[0]
    exec(source, namespace)
    assert len(namespace['final_train_loader'].dataset) == 140
    assert len(namespace['final_val_loader'].dataset) == 20
    assert len(namespace['final_test_loader'].dataset) == 60


@pytest.mark.parametrize('language', ['es', 'en'])
def test_report_describes_actual_data_and_selected_epoch(tmp_path, language, monkeypatch):
    from scripts.report_winner_scientific import saved_run_descriptions

    monkeypatch.setattr(config_module, 'DEFAULT_INPUT_MODALITY', 'spectrogram')
    config = get_default_config()
    config['data_path'] = str(tmp_path)
    pd.DataFrame({'origin': ['real', 'generated']}).to_csv(tmp_path / 'manifest.csv', index=False)
    (tmp_path / 'feature_config.json').write_text(json.dumps({'shape': [64, 44], 'hop_length': 240}))
    (tmp_path / 'dataset_config.json').write_text(json.dumps({'excluded_test_generated_audios': 2399}))
    run = {'config': config, 'genome': {
        'id': 'example', 'num_conv_layers': 1, 'num_fc_layers': 1, 'filters': [4], 'kernel_sizes': [3],
        'fc_nodes': [8], 'activations': ['relu'], 'optimizer': 'adam', 'learning_rate': 0.001,
    }, 'folds': [{'fold': 2, 'metrics': {'best_epoch': 3}}]}
    result = {'protocol': {'selected_fold': 2}, 'folds': [{'fold': 2, 'samples': 1721}],
              'dataset_splits': {'train': {'people': 70, 'recordings': 8609},
                                 'val': {'people': 10, 'recordings': 1230},
                                 'test': {'people': 20, 'recordings': 60}}}
    descriptions = saved_run_descriptions(run, result, language)
    assert '2' in descriptions['selection'] and '3' in descriptions['selection']
    assert '68' not in descriptions['selection']
    assert 'test: 20 / 60' in descriptions['data'] and '2399' in descriptions['data']
    assert '24000 Hz' in descriptions['input'] and '0.48 s' in descriptions['input']
    assert '(64, 44)' in descriptions['input']
    assert '6888 / 1721' in descriptions['cv']
