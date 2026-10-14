"""Exercise the notebook's winner selection without running neuroevolution."""

import ast
import json
from pathlib import Path

import pytest


def notebook():
    return json.loads((Path(__file__).resolve().parents[1] / 'test.ipynb').read_text(encoding='utf-8'))


def resolver_namespace(tmp_path):
    source = ''.join(notebook()['cells'][3]['source'])
    helper = next(node for node in ast.parse(source).body
                  if isinstance(node, ast.FunctionDef) and node.name == 'resolve_best_genome')
    namespace = {'json': json, 'Path': Path, 'CONFIG': {'artifacts_dir': str(tmp_path)}}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), '<winner-resolver>', 'exec'), namespace)
    return namespace


def save_progress(tmp_path, value):
    path = tmp_path / 'evolution_progress.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def test_existing_winner_is_preserved_without_reading_progress(tmp_path):
    save_progress(tmp_path, {'best_individual': {'id': 'disk-winner', 'fitness': 95}})
    current = {'id': 'in-memory', 'fitness': 75, 'filters': [16]}
    resolve = resolver_namespace(tmp_path)['resolve_best_genome']
    assert resolve(current) is current
    (tmp_path / 'evolution_progress.json').unlink()
    assert resolve(current) is current


@pytest.mark.parametrize('candidate', [None, {}, 'invalid'])
def test_recorded_best_individual_is_used_before_population(tmp_path, candidate):
    winner = {'id': 'global-winner', 'fitness': 77.5, 'filters': [16, 32], 'num_conv_layers': 2}
    save_progress(tmp_path, {'best_individual': winner,
                            'population': [{'id': 'population-only', 'fitness': 80}]})
    resolve = resolver_namespace(tmp_path)['resolve_best_genome']
    assert resolve(candidate) == winner


@pytest.mark.parametrize('saved_best', [None, {}, {'id': 'bad', 'fitness': None}])
def test_missing_recorded_winner_uses_highest_valid_population_fitness(tmp_path, saved_best):
    winner = {'id': 'best-population', 'fitness': 85, 'filters': [32]}
    save_progress(tmp_path, {'best_individual': saved_best, 'population': [
        {'id': 'lower', 'fitness': 70}, None, {'id': 'nan', 'fitness': float('nan')},
        {'id': 'infinite', 'fitness': float('inf')}, {'fitness': 100},
        {'id': 'bad-string', 'fitness': 'not-a-score'}, winner,
    ]})
    resolve = resolver_namespace(tmp_path)['resolve_best_genome']
    assert resolve() == winner


def test_missing_progress_reports_the_configured_path(tmp_path):
    resolve = resolver_namespace(tmp_path)['resolve_best_genome']
    with pytest.raises(FileNotFoundError, match='evolution_progress.json'):
        resolve()


@pytest.mark.parametrize('progress', [{}, {'population': []}, {'best_individual': {'id': 'no-score'}}])
def test_progress_without_a_valid_winner_is_rejected(tmp_path, progress):
    save_progress(tmp_path, progress)
    resolve = resolver_namespace(tmp_path)['resolve_best_genome']
    with pytest.raises(ValueError, match='fitness válido'):
        resolve()


def test_final_evaluation_recovers_winner_before_opening_test_data(tmp_path):
    winner = {'id': 'saved-winner', 'fitness': 78, 'num_conv_layers': 2}
    save_progress(tmp_path, {'best_individual': winner})
    namespace = resolver_namespace(tmp_path)
    source = ''.join(notebook()['cells'][19]['source']).split('_saved_checkpoint =')[0]
    exec(source, namespace)
    assert namespace['best_genome'] == winner


def test_missing_winner_stops_before_data_loading(tmp_path):
    namespace = resolver_namespace(tmp_path)
    source = ''.join(notebook()['cells'][19]['source'])
    with pytest.raises(FileNotFoundError, match='evolution_progress.json'):
        exec(source, namespace)
    assert 'final_test_loader' not in namespace


def test_architecture_selection_works_without_evolution_object(tmp_path, monkeypatch):
    from neuroevolution.visualization import reports
    calls = []
    monkeypatch.setattr(reports, 'display_best_architecture', lambda *args: calls.append(args))
    winner = {'id': 'saved-winner', 'fitness': 78}
    save_progress(tmp_path, {'best_individual': winner})
    namespace = resolver_namespace(tmp_path)
    exec(''.join(notebook()['cells'][15]['source']), namespace)
    assert namespace['best_genome'] == winner
    assert calls[0][0] == winner
    assert calls[0][2].generation == 0


def standalone_experiment(tmp_path, monkeypatch, *, checkpoint=True, legacy=False,
                          saved_checkpoint='/previous/server/best_model_gen4_idstandalone_fitness78.00.pth'):
    import torch
    from neuroevolution import config as config_module
    from neuroevolution.evolution.engine import HybridNeuroevolution
    from neuroevolution.models.evolvable_cnn import EvolvableCNN

    config = config_module.get_default_config(str(tmp_path))
    config.update(input_modality='audio', num_channels=1, sequence_length=16,
                  pre_fc_pool_length=4, conv_dropout_rate=0.0)
    genome = {'id': 'standalone', 'fitness': 78.0, 'num_conv_layers': 1,
              'num_fc_layers': 1, 'filters': [4], 'kernel_sizes': [3],
              'fc_nodes': [8], 'activations': ['relu'], 'dropout_rate': 0.0,
              'normalization_type': 'batch', 'optimizer': 'adam', 'learning_rate': 0.001}
    progress = {'best_individual': genome, 'generation': 4,
                'generation_stats': [], 'best_checkpoint_path': saved_checkpoint}
    if not legacy:
        progress['config_snapshot'] = config
    save_progress(tmp_path, progress)
    state = None
    if checkpoint:
        model = EvolvableCNN(genome, config)
        state = model.state_dict()
        torch.save({'genome': genome, 'config': config, 'model_state_dict': state,
                    'generation': 4, 'fitness': 78.0},
                   tmp_path / 'best_model_gen4_idstandalone_fitness78.00.pth')
    default_config = config_module.get_default_config(str(tmp_path))
    monkeypatch.setattr(config_module, 'get_default_config', lambda: dict(default_config))
    monkeypatch.setattr(HybridNeuroevolution, 'evolve', lambda *args: pytest.fail('Evolution must not run'))
    monkeypatch.setattr(HybridNeuroevolution, 'initialize_population', lambda *args: pytest.fail('No population initialization'))
    return config, genome, state


@pytest.mark.parametrize('legacy', [False, True])
def test_architecture_cell_runs_from_empty_kernel_with_full_report(tmp_path, monkeypatch, legacy):
    config, genome, _ = standalone_experiment(tmp_path, monkeypatch, legacy=legacy)
    original_progress = (tmp_path / 'evolution_progress.json').read_bytes()
    namespace = {}
    exec(''.join(notebook()['cells'][15]['source']), namespace)
    assert namespace['best_genome'] == genome
    assert namespace['CONFIG']['input_modality'] == config['input_modality']
    assert namespace['_report_context'].generation == 4
    reports = list(tmp_path.glob('best_architecture_audio_*.json'))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())['best_genome']['id'] == 'standalone'
    assert (tmp_path / 'evolution_progress.json').read_bytes() == original_progress


@pytest.mark.parametrize('saved_path', [
    '/previous/server/best_model_gen4_idstandalone_fitness78.00.pth',
    r'C:\previous\experiment\best_model_gen4_idstandalone_fitness78.00.pth',
])
def test_checkpoint_cell_runs_alone_and_relocates_weights(tmp_path, monkeypatch, saved_path):
    import torch
    _, genome, state = standalone_experiment(tmp_path, monkeypatch, saved_checkpoint=saved_path)
    original_progress = (tmp_path / 'evolution_progress.json').read_bytes()
    namespace = {}
    exec(''.join(notebook()['cells'][17]['source']), namespace)
    assert namespace['loaded_genome'] == genome
    model = namespace['loaded_model']
    assert not model.training
    assert next(model.parameters()).device.type == 'cpu'
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])
    assert (tmp_path / 'evolution_progress.json').read_bytes() == original_progress


def test_checkpoint_cell_handles_json_without_weights_from_empty_kernel(tmp_path, monkeypatch, capsys):
    _, genome, _ = standalone_experiment(tmp_path, monkeypatch, checkpoint=False)
    namespace = {}
    exec(''.join(notebook()['cells'][17]['source']), namespace)
    assert namespace['best_genome'] == genome
    assert namespace['loaded_model'] is None
    assert 'no contiene los pesos' in capsys.readouterr().out


def test_standalone_cell_discovers_only_experiment_when_default_folder_is_missing(tmp_path, monkeypatch):
    from neuroevolution import config as config_module
    from neuroevolution.visualization import reports
    actual = tmp_path / 'audio'
    actual.mkdir()
    standalone_experiment(actual, monkeypatch, checkpoint=False)
    default = dict(config_module.get_default_config())
    default['artifacts_dir'] = str(tmp_path / 'spectrogram')
    monkeypatch.setattr(config_module, 'get_default_config', lambda: dict(default))
    monkeypatch.setattr(reports, 'display_best_architecture', lambda *args: None)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(''.join(notebook()['cells'][15]['source']), namespace)
    assert Path(namespace['CONFIG']['artifacts_dir']) == actual
    assert namespace['best_genome']['id'] == 'standalone'


def test_standalone_cell_accepts_explicit_json_path_with_another_filename(tmp_path, monkeypatch):
    from neuroevolution.visualization import reports
    standalone_experiment(tmp_path, monkeypatch, checkpoint=False)
    saved_path = tmp_path / 'saved_progress.json'
    (tmp_path / 'evolution_progress.json').rename(saved_path)
    monkeypatch.setattr(reports, 'display_best_architecture', lambda *args: None)
    namespace = {'EVOLUTION_PROGRESS_PATH': str(saved_path)}
    exec(''.join(notebook()['cells'][15]['source']), namespace)
    assert namespace['best_genome']['id'] == 'standalone'


def standalone_final_dataset(tmp_path, monkeypatch):
    import csv
    import numpy as np

    config, genome, _ = standalone_experiment(tmp_path, monkeypatch, checkpoint=False)
    dataset = tmp_path / 'dataset'
    arrays = dataset / 'arrays_audio'
    arrays.mkdir(parents=True)
    for split, start, stop, repetitions in [('train', 0, 35, 2),
                                           ('val', 35, 40, 1), ('test', 40, 50, 3)]:
        rows = [{'person_id': f'{label}_{person}', 'label_binary': label,
                 'cv_fold': person // 7 if split == 'train' else -1,
                 'origin': 'generated' if split == 'train' and repetition else 'real'}
                for label in (0, 1) for person in range(start, stop)
                for repetition in range(repetitions)]
        with (dataset / f'{split}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['person_id', 'label_binary', 'origin', 'cv_fold'])
            writer.writeheader()
            writer.writerows(rows)
        targets = np.array([row['label_binary'] for row in rows], dtype=np.int64)
        features = np.repeat((targets * 2 - 1).astype(np.float32)[:, None, None], 16, axis=2)
        np.save(arrays / f'X_{split}_old_vowels.npy', features)
        np.save(arrays / f'y_{split}_old_vowels.npy', targets)
    config.update(data_path=str(dataset), fold_files_subdirectory='arrays_audio',
                  num_epochs=1, epoch_patience=1, batch_size=32, use_amp=False)
    progress_path = tmp_path / 'evolution_progress.json'
    progress = json.loads(progress_path.read_text())
    progress['config_snapshot'] = config
    save_progress(tmp_path, progress)
    return config, genome


def test_final_cell_runs_real_training_from_empty_kernel_and_saves_results(tmp_path, monkeypatch):
    import torch
    config, genome = standalone_final_dataset(tmp_path, monkeypatch)
    namespace = {}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        exec(''.join(notebook()['cells'][19]['source']), namespace)
    finally:
        torch.set_num_threads(previous_threads)
    assert {key: namespace['best_genome'][key] for key in genome} == genome
    result = namespace['final_test_results']
    assert result['n_samples'] == 60
    assert result['best_epoch'] == 1
    assert result['protocol'] == 'single_global_person_held_out_test'
    assert result['train_people'] == 70
    assert result['validation_people'] == 10
    assert result['test_people'] == 20
    assert result['evaluation_split'] == 'test'
    assert len(namespace['final_train_loader'].dataset) == 140
    assert 'fold_results' not in result and 'ensemble_weights' not in result
    assert 'final_fold_loaders' not in namespace
    saved_path = tmp_path / 'old_vowels_final_test.json'
    saved = json.loads(saved_path.read_text())
    assert saved['n_samples'] == result['n_samples']
    assert saved['f1_score'] == result['f1_score']
    assert saved['confusion_matrix'] == result['confusion_matrix']
    assert saved['best_selection_metrics']['confusion_matrix'] == result['best_selection_metrics']['confusion_matrix'].tolist()
    assert namespace['ARRAY_ROOT'] == Path(config['data_path']) / 'arrays_audio'


def test_final_cell_trains_once_without_importing_cv_backend(tmp_path, monkeypatch):
    import builtins
    import torch
    from neuroevolution.evaluation import cross_validation

    _, genome = standalone_final_dataset(tmp_path, monkeypatch)
    original_import = builtins.__import__
    def no_cv_backend(name, *args, **kwargs):
        if name == 'neuroevolution.evaluation.final_cv':
            pytest.fail('Global training must not import a CV backend')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_cv_backend)
    calls = []
    original_train = cross_validation.evaluate_single_fold
    def single_train(*args, **kwargs):
        calls.append((len(args[2].dataset), len(args[3].dataset), len(args[4].dataset)))
        return original_train(*args, **kwargs)
    monkeypatch.setattr(cross_validation, 'evaluate_single_fold', single_train)
    namespace = {}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        exec(''.join(notebook()['cells'][19]['source']), namespace)
    finally:
        torch.set_num_threads(previous_threads)
    assert calls == [(140, 10, 60)]
    result = namespace['final_test_results']
    assert namespace['best_genome']['id'] == genome['id']
    assert result['protocol'] == 'single_global_person_held_out_test'
    assert result['n_samples'] == 60
    assert 'cv_oof_metrics' not in result
    assert not (tmp_path / 'final_cv_standalone').exists()
    assert (tmp_path / 'old_vowels_final_test.json').is_file()


def test_results_cell_runs_from_empty_kernel_using_saved_json(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    standalone_experiment(tmp_path, monkeypatch, checkpoint=False)
    result = {key: 75.0 for key in ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc']}
    result.update(n_samples=60, confusion_matrix=[[20, 10], [5, 25]])
    (tmp_path / 'old_vowels_final_test.json').write_text(json.dumps(result))
    monkeypatch.setattr(plt, 'show', lambda: None)
    namespace = {}
    try:
        exec(''.join(notebook()['cells'][20]['source']), namespace)
        assert namespace['final_test_results'] == result
        assert namespace['matrix'].tolist() == result['confusion_matrix']
    finally:
        plt.close('all')


def test_final_cell_relocates_dataset_after_moving_experiment(tmp_path, monkeypatch):
    config, genome = standalone_final_dataset(tmp_path, monkeypatch)
    from neuroevolution.evaluation import cross_validation
    # Loading is real; reuse the deterministic metric dictionary to isolate relocation.
    monkeypatch.setattr(cross_validation, 'evaluate_single_fold', lambda *args, **kwargs: {
        'fold': 1, 'accuracy': 75, 'sensitivity': 75, 'specificity': 75,
        'f1_score': 75, 'auc': 75, 'n_samples': 60, 'confusion_matrix': [[20, 10], [5, 25]],
    })
    progress_path = tmp_path / 'evolution_progress.json'
    progress = json.loads(progress_path.read_text())
    progress['config_snapshot']['data_path'] = '/previous/server/artifacts/dataset'
    save_progress(tmp_path, progress)
    # The destination dataset keeps its basename next to the experiment directory.
    destination = tmp_path.parent / 'dataset'
    Path(config['data_path']).rename(destination)
    namespace = {}
    exec(''.join(notebook()['cells'][19]['source']), namespace)
    assert namespace['DATASET_ROOT'] == destination
    assert namespace['best_genome'] == genome


def test_all_notebook_code_cells_compile():
    for index, cell in enumerate(notebook()['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'<test.ipynb cell {index}>', 'exec')
