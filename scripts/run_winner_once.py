"""Train the uploaded winner once per fold; settings live in config.py."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from neuroevolution.config import (
    get_winner_replay_config, validate_config, configure_vowels_input,
    with_fold_epoch_callback,
)
from neuroevolution.device_utils import setup_seeds
from neuroevolution.evolution.fitness import train_fold_in_thread
from neuroevolution.models.architecture_formatting import format_fc_activations


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class Tee:
    def __init__(self, original, logfile):
        self.original, self.logfile = original, logfile

    def write(self, message):
        self.original.write(message)
        self.logfile.write(message)
        self.flush()

    def flush(self):
        self.original.flush()
        self.logfile.flush()


def main():
    config = get_winner_replay_config()
    validate_config(config)
    configure_vowels_input(config)
    source = Path(config['winner_progress_path'])
    original = json.loads(source.read_text(encoding='utf-8'))['best_individual']
    if original['id'] != config['winner_expected_id']:
        raise ValueError('The uploaded winner differs from the configured ID')
    genome_keys = (
        'id', 'num_conv_layers', 'num_fc_layers', 'filters', 'kernel_sizes',
        'fc_nodes', 'activations', 'dropout_rate', 'learning_rate', 'optimizer',
        'normalization_type', 'conv_topology', 'residual_enabled',
        'residual_block_size', 'residual_projection', 'inception_enabled',
        'inception_reduction_ratio', 'inception_pool_branch',
    )
    genome = {key: copy.deepcopy(original[key]) for key in genome_keys}
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('The configured CUDA device is unavailable')
    if config['fold_parallel_workers'] != 1:
        raise ValueError('This one-pass runner requires sequential folds')
    torch.set_num_threads(config['torch_num_threads'])
    output = Path(config['artifacts_dir'])
    output.mkdir(parents=True, exist_ok=True)
    # An exclusive persistent marker prevents an accidental second pass.
    with (output / 'started_once.json').open('x', encoding='utf-8') as marker:
        json.dump({'pid': os.getpid(), 'started_at': now()}, marker)
    logfile = (output / 'execution_log.txt').open('a', encoding='utf-8', buffering=1)
    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = Tee(sys.stderr, logfile)
    started = time.monotonic()
    state = {
        'status': 'running', 'started_at': now(), 'pid': os.getpid(),
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'config': config, 'genome': genome, 'folds': [],
        'original_mean_f1': original['fitness'],
        'original_fold_metrics': original['metrics']['fold_metrics'],
        'evaluation_split': 'validation',
    }
    write_json(output / 'results.json', state)
    print(f"[{now()}] One pass: winner={genome['id']}, device={state['device']}, folds={config['num_folds']}")
    print(f"Epoch limit={config['num_epochs']}, patience={config['epoch_patience']}, batch={config['batch_size']}")
    print(f'Hidden FC activations: {format_fc_activations(genome, config)}. Fresh weights for each fold.')
    try:
        for fold in range(1, config['num_folds'] + 1):
            seed = config['seed'] + (fold - 1) * config['fold_seed_stride']
            setup_seeds(seed)
            fold_started = time.monotonic()
            state.update(current_fold=fold, current_epoch=0, current_fold_history=[])
            write_json(output / 'results.json', state)
            print(f"[{now()}] Starting fold {fold}/{config['num_folds']}, seed={seed}")

            def on_epoch(metrics):
                state['current_epoch'] = metrics['epoch']
                state['current_fold_history'].append(metrics)
                state['elapsed_seconds'] = time.monotonic() - started
                write_json(output / 'results.json', state)
                print(f"[{now()}] Fold {fold} epoch {metrics['epoch']}/{config['num_epochs']}: "
                      f"F1={metrics['f1_score']:.2f}% accuracy={metrics['accuracy']:.2f}% "
                      f"specificity={metrics['specificity']:.2f}%")

            _, score, model, metrics = train_fold_in_thread(
                copy.deepcopy(genome), fold, with_fold_epoch_callback(config, on_epoch), device,
            )
            if model is None or metrics is None:
                raise RuntimeError(f'Fold {fold} failed; no fold will be retried automatically')
            checkpoint = output / f'fold_{fold}_best.pt'
            torch.save({
                'genome': genome, 'config': config, 'fold': fold, 'seed': seed,
                'metrics': metrics,
                'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }, checkpoint)
            state['folds'].append({
                'fold': fold, 'seed': seed, 'f1_score': score, 'metrics': metrics,
                'duration_seconds': time.monotonic() - fold_started,
                'checkpoint_path': str(checkpoint),
            })
            write_json(output / 'results.json', state)
            print(f"[{now()}] Fold {fold} saved: best F1={score:.2f}%, epoch={metrics['best_epoch']}")
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        state.update(
            status='completed', completed_at=now(), elapsed_seconds=time.monotonic() - started,
            mean_f1=float(np.mean([fold['f1_score'] for fold in state['folds']])),
            std_f1=float(np.std([fold['f1_score'] for fold in state['folds']])),
        )
        write_json(output / 'results.json', state)
        print(f"[{now()}] COMPLETED: mean F1={state['mean_f1']:.2f}% +/- {state['std_f1']:.2f}%")
    except BaseException:
        state.update(status='failed', failed_at=now(), error=traceback.format_exc())
        write_json(output / 'results.json', state)
        raise


if __name__ == '__main__':
    main()
