"""Audit saved checkpoints and export scientific results without retraining."""
import copy
import hashlib
import json
import importlib.metadata
import platform
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix,
)
from scipy.stats import binomtest
from neuroevolution.config import get_scientific_evaluation_config
from neuroevolution.models.architecture_formatting import format_fc_activations
from neuroevolution.models.evolvable_cnn import EvolvableCNN


def metrics(frame):
    y, pred, p = (frame[key].to_numpy() for key in ('y_true', 'y_pred', 'probability'))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().astype(float)
    div = lambda a, b: float(a / b) if b else 0.0
    sensitivity, specificity = div(tp, tp + fn), div(tn, tn + fp)
    precision, npv = div(tp, tp + fp), div(tn, tn + fn)
    f1, f1_control = div(2 * tp, 2 * tp + fp + fn), div(2 * tn, 2 * tn + fp + fn)
    accuracy = div(tp + tn, len(y))
    expected = div((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), len(y) ** 2)
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {
        'accuracy': 100 * accuracy, 'balanced_accuracy': 50 * (sensitivity + specificity),
        'sensitivity': 100 * sensitivity, 'specificity': 100 * specificity,
        'precision': 100 * precision, 'npv': 100 * npv,
        'f1_score': 100 * f1, 'f1_control': 100 * f1_control,
        'macro_f1': 50 * (f1 + f1_control),
        'weighted_f1': 100 * div(f1 * (tp + fn) + f1_control * (tn + fp), len(y)),
        'roc_auc': 100 * float(roc_auc_score(y, p)),
        'average_precision': 100 * float(average_precision_score(y, p)),
        'mcc': div(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
        'cohen_kappa': div(accuracy - expected, 1 - expected),
        'brier_score': float(np.mean((p - y) ** 2)),
        'log_loss': float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        'false_positive_rate': 100 * (1 - specificity),
        'false_negative_rate': 100 * (1 - sensitivity),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'samples': len(y), 'people': int(frame.person_id.nunique()),
    }


def bootstrap(frame, config):
    rng = np.random.default_rng(config['bootstrap_seed'])
    frame = frame.reset_index(drop=True)
    groups = frame.groupby('person_id', sort=True).indices
    labels = frame.groupby('person_id', sort=True).y_true.first()
    classes = [labels.index[labels.eq(label)].to_numpy() for label in (0, 1)]
    names = [key for key in metrics(frame) if key not in {'tn', 'fp', 'fn', 'tp', 'samples', 'people'}]
    values = {key: [] for key in names}
    for _ in range(config['bootstrap_repetitions']):
        subjects = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in classes])
        positions = np.concatenate([groups[subject] for subject in subjects])
        result = metrics(frame.iloc[positions])
        for key in names:
            values[key].append(result[key])
    tail = (1 - config['confidence_level']) / 2
    return {key: np.quantile(value, [tail, 1 - tail]).tolist() for key, value in values.items()}


def people_predictions(frame, config):
    if config['person_aggregation'] != 'mean_probability':
        raise ValueError('Unsupported person aggregation')
    people = frame.groupby('person_id', sort=True).agg(
        y_true=('y_true', 'first'), probability=('probability', 'mean'), recordings=('y_true', 'size'),
    ).reset_index()
    people['y_pred'] = (people.probability > config['person_probability_threshold']).astype(int)
    return people


def infer(model, x, config):
    predictions, probabilities = [], []
    device = torch.device(config['device'])
    dtype = torch.bfloat16 if config['amp_dtype'] == 'bfloat16' else torch.float16
    model.eval()
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(x), config['batch_size']):
            batch = torch.as_tensor(np.array(x[offset:offset + config['batch_size']], copy=True)).to(device)
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=config['use_amp'] and device.type == 'cuda'):
                logits = model(batch)
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].float().cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return predictions, probabilities, time.monotonic() - started


def load_split(root, config, split, fold=None):
    csv_path = root / (f'fold_{fold - 1}_{split}.csv' if fold else f'{split}.csv')
    rows = pd.read_csv(csv_path)
    suffix = f'_fold_{fold}' if fold else ''
    arrays = root / config['fold_files_subdirectory']
    x = np.load(arrays / f'X_{split}_{config["dataset_id"]}{suffix}.npy', mmap_mode='r')
    y = np.load(arrays / f'y_{split}_{config["dataset_id"]}{suffix}.npy')
    assert len(rows) == len(x) and np.array_equal(rows.label_binary.to_numpy(), y)
    # Verify sample order, including within-class person identities.
    assert all(np.array_equal(x[i], np.load(root / row.image_npy_path))
               for i, row in enumerate(rows.itertuples(index=False)))
    return rows, x, y


def main():
    settings = get_scientific_evaluation_config()
    run_path = Path(settings['run_results_path'])
    run = json.loads(run_path.read_text(encoding='utf-8'))
    if run['status'] != 'completed':
        raise RuntimeError('The training pass has not completed')
    # Reconstruct saved models with the configuration used to train them.
    config = copy.deepcopy(run['config'])
    torch.set_num_threads(config['torch_num_threads'])
    output = Path(settings['report_directory'])
    output.mkdir(parents=True, exist_ok=True)
    saved_analysis = output / 'scientific_results.json'
    if saved_analysis.exists():
        print('Reusing saved predictions; no new test inference or training.', flush=True)
        finalize_report(run, json.loads(saved_analysis.read_text(encoding='utf-8')), output, config, settings)
        return
    root = Path(config['data_path'])
    manifest = pd.read_csv(root / 'manifest.csv')
    split_people = {split: set(manifest.loc[manifest.split.eq(split), 'person_id'])
                    for split in ('train', 'val', 'test')}
    assert all(split_people[a].isdisjoint(split_people[b])
               for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')))
    # Freeze the checkpoint choice before any test predictions are observed.
    chosen = max(run['folds'], key=lambda fold: fold['metrics'][settings['test_model_selection_metric']])
    protocol = {
        'selected_fold': chosen['fold'], 'selection': 'highest saved validation F1; ties use first fold',
        'selected_checkpoint': chosen['checkpoint_path'], 'settings': settings,
        'run_results_sha256': hashlib.sha256(run_path.read_bytes()).hexdigest(),
        'test_tuning': False, 'additional_training': False,
    }
    protocol_path = output / 'evaluation_protocol.json'
    if protocol_path.exists():
        raise RuntimeError('Scientific evaluation already started; inspect existing outputs instead of repeating test')
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    fold_predictions, fold_metrics = [], []
    selected_model = None
    for fold in run['folds']:
        checkpoint = torch.load(fold['checkpoint_path'], map_location='cpu', weights_only=True)
        model = EvolvableCNN(copy.deepcopy(run['genome']), config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config['device'])
        rows, x, y = load_split(root, config, 'val', fold['fold'])
        train_rows = pd.read_csv(root / f'fold_{fold["fold"] - 1}_train.csv')
        assert set(rows.person_id).isdisjoint(train_rows.person_id)
        assert set(rows.person_id).isdisjoint(split_people['test'])
        pred, prob, seconds = infer(model, x, config)
        frame = rows[['person_id', 'vowel', 'repetition', 'source_path']].copy()
        frame['y_true'], frame['y_pred'], frame['probability'], frame['fold'] = y, pred, prob, fold['fold']
        result = metrics(frame)
        for key in ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score'):
            assert np.isclose(result[key], fold['metrics'][key]), f'Checkpoint mismatch: fold {fold["fold"]} {key}'
        assert np.isclose(result['roc_auc'], fold['metrics']['auc'])
        result.update(fold=fold['fold'], best_epoch=fold['metrics']['best_epoch'],
                      epochs_completed=fold['metrics']['epochs_completed'], training_seconds=fold['duration_seconds'],
                      seed=fold['seed'], inference_seconds=seconds)
        fold_metrics.append(result)
        fold_predictions.append(frame)
        print(f"Audited fold {fold['fold']}: F1={result['f1_score']:.2f}% AUC={result['roc_auc']:.2f}%", flush=True)
        if fold['fold'] == chosen['fold']:
            selected_model = model
        else:
            del model
    oof = pd.concat(fold_predictions, ignore_index=True)
    assert len(oof) == len(manifest.loc[manifest.split.eq('train')])
    assert not oof.source_path.duplicated().any() and set(oof.person_id) == split_people['train']
    rows, x, y = load_split(root, config, 'test')
    pred, prob, seconds = infer(selected_model, x, config)
    test = rows[['person_id', 'vowel', 'repetition', 'source_path']].copy()
    test['y_true'], test['y_pred'], test['probability'] = y, pred, prob
    oof_people, test_people = people_predictions(oof, settings), people_predictions(test, settings)
    datasets = {'oof_recordings': oof, 'oof_people': oof_people,
                'test_recordings': test, 'test_people': test_people}
    results = {'protocol': protocol, 'folds': fold_metrics, 'test_inference_seconds': seconds,
               'dataset_splits': {split: {'people': len(split_people[split]),
                                         'recordings': int(manifest.split.eq(split).sum())}
                                  for split in split_people}}
    fold_frame = pd.DataFrame(fold_metrics)
    fold_frame.to_csv(output / 'fold_metrics.csv', index=False)
    names = [key for key in metrics(oof) if key not in {'tn', 'fp', 'fn', 'tp', 'samples', 'people'}]
    results['fold_mean_sample_sd'] = {
        key: {'mean': float(fold_frame[key].mean()), 'sd': float(fold_frame[key].std(ddof=1))} for key in names
    }
    table_rows = []
    for name, frame in datasets.items():
        frame.to_csv(output / f'{name}_predictions.csv', index=False)
        point = metrics(frame)
        print(f"Calculating clustered bootstrap: {name}, n={len(frame)}, F1={point['f1_score']:.2f}%", flush=True)
        intervals = bootstrap(frame, settings)
        results[name] = {'metrics': point, 'confidence_intervals_95': intervals}
        for metric, value in point.items():
            ci = intervals.get(metric, [None, None])
            table_rows.append({'population': name, 'metric': metric, 'value': value,
                               'ci95_low': ci[0], 'ci95_high': ci[1]})
    pd.DataFrame(table_rows).to_csv(output / 'metrics_with_confidence_intervals.csv', index=False)
    (output / 'scientific_results.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    make_figures(run, datasets, output, settings)
    finalize_report(run, results, output, config, settings)
    print(json.dumps({key: results[key]['metrics'] for key in datasets}, indent=2), flush=True)


def finalize_report(run, results, output, config, settings):
    """Refine reporting from saved predictions, never consulting a model again."""
    results['analysis_environment'] = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'libraries': {name: importlib.metadata.version(name) for name in
                      ('numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'torch', 'torchvision')},
    }
    rows, class_rows = [], []
    for population in ('oof_recordings', 'oof_people', 'test_recordings', 'test_people'):
        point = results[population]['metrics']
        intervals = results[population]['confidence_intervals_95']
        methods = {name: 'stratified person-cluster percentile bootstrap' for name in intervals}
        tn, fp, fn, tp = (point[key] for key in ('tn', 'fp', 'fn', 'tp'))
        if population.endswith('_people'):
            denominators = {
                'sensitivity': (tp, tp + fn), 'specificity': (tn, tn + fp),
                'precision': (tp, tp + fp), 'npv': (tn, tn + fn),
                'false_positive_rate': (fp, tn + fp), 'false_negative_rate': (fn, tp + fn),
            }
            for name, (successes, total) in denominators.items():
                interval = binomtest(successes, total).proportion_ci(
                    confidence_level=settings['confidence_level'], method='exact')
                intervals[name] = [100 * interval.low, 100 * interval.high]
                methods[name] = 'Clopper-Pearson exact binomial interval at person level'
        results[population]['confidence_interval_methods'] = methods
        results[population]['confusion_matrix'] = [[tn, fp], [fn, tp]]
        for metric, value in point.items():
            ci = intervals.get(metric, [None, None])
            rows.append({'population': population, 'metric': metric, 'value': value,
                         'ci95_low': ci[0], 'ci95_high': ci[1], 'ci_method': methods.get(metric, '')})
        for label, precision, recall, f1, support in (
            ('Control', point['npv'], point['specificity'], point['f1_control'], tn + fp),
            ('Patologicas', point['precision'], point['sensitivity'], point['f1_score'], tp + fn),
        ):
            class_rows.append({'population': population, 'class': label, 'precision': precision,
                               'recall': recall, 'f1': f1, 'support': support})
    pd.DataFrame(rows).to_csv(output / 'metrics_with_confidence_intervals.csv', index=False)
    pd.DataFrame(class_rows).to_csv(output / 'classification_reports.csv', index=False)
    test = pd.read_csv(output / 'test_recordings_predictions.csv')
    vowels = [{'vowel': vowel, **metrics(frame)} for vowel, frame in test.groupby('vowel')]
    pd.DataFrame(vowels).to_csv(output / 'test_metrics_by_vowel.csv', index=False)
    results['test_metrics_by_vowel'] = vowels
    (output / 'scientific_results.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    write_report(run, results, output, config)


def make_figures(run, datasets, output, config, language='es'):
    translations = {
        'Test: 300 grabaciones': 'Test: 300 recordings', 'Test: 20 personas': 'Test: 20 participants',
        'Patológicas': 'Pathological', 'Predicción': 'Predicted class', 'Clase real': 'True class',
        'Grabaciones': 'Recordings', 'Personas': 'Participants', 'Azar': 'Chance',
        'Prevalencia 50 %': 'Prevalence: 50%', 'ROC del test': 'Test ROC curve',
        'Precisión-recall del test': 'Test precision–recall curve',
        '1 − Especificidad': '1 − Specificity', 'Sensibilidad': 'Sensitivity / recall',
        'Precisión': 'Precision', 'Época': 'Epoch', 'Validación (%)': 'Validation (%)',
    }
    label = lambda text: translations.get(text, text) if language == 'en' else text
    suffix = '_en' if language == 'en' else ''
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, name, title in zip(axes, ('test_recordings', 'test_people'), ('Test: 300 grabaciones', 'Test: 20 personas')):
        frame = datasets[name]
        cm = confusion_matrix(frame.y_true, frame.y_pred, labels=[0, 1])
        axis.imshow(cm, cmap='Blues', vmin=0)
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=18,
                          color='white' if cm[i, j] > cm.max() / 2 else '#222222')
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Control', label('Patológicas')],
                 yticklabels=['Control', label('Patológicas')], xlabel=label('Predicción'), ylabel=label('Clase real'), title=label(title))
    fig.tight_layout()
    fig.savefig(output / f'test_confusion_matrices{suffix}.png', dpi=config['figure_dpi'])
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, label, color, linestyle in (
        ('test_recordings', 'Grabaciones', '#176B9A', '-'), ('test_people', 'Personas', '#C58024', '--'),
    ):
        frame = datasets[name]
        fpr, tpr, _ = roc_curve(frame.y_true, frame.probability)
        precision, recall, _ = precision_recall_curve(frame.y_true, frame.probability)
        axes[0].plot(fpr, tpr, color=color, linestyle=linestyle,
                     label=f'{translations.get(label, label) if language == "en" else label}: AUC={roc_auc_score(frame.y_true, frame.probability):.3f}')
        axes[1].plot(recall, precision, color=color, linestyle=linestyle,
                     label=f'{translations.get(label, label) if language == "en" else label}: AP={average_precision_score(frame.y_true, frame.probability):.3f}')
    translate = lambda text: translations.get(text, text) if language == 'en' else text
    axes[0].plot([0, 1], [0, 1], color='#666666', linestyle=':', label=translate('Azar'))
    axes[1].axhline(.5, color='#666666', linestyle=':', label=translate('Prevalencia 50 %'))
    for axis, title, xlabel, ylabel in zip(axes, ('ROC del test', 'Precisión-recall del test'),
                                        ('1 − Especificidad', 'Sensibilidad'), ('Sensibilidad', 'Precisión')):
        axis.set(title=translate(title), xlabel=translate(xlabel), ylabel=translate(ylabel), xlim=(0, 1), ylim=(0, 1.03))
        axis.legend(loc='lower right' if axis == axes[0] else 'lower left')
    fig.tight_layout()
    fig.savefig(output / f'test_roc_pr{suffix}.png', dpi=config['figure_dpi'])
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for axis, fold in zip(axes.flat, run['folds']):
        history = pd.DataFrame(fold['metrics']['epoch_history'])
        axis.plot(history.epoch, history.f1_score, color='#176B9A', label='F1')
        axis.plot(history.epoch, history.auc, color='#C58024', linestyle='--', label='ROC-AUC')
        axis.axvline(fold['metrics']['best_epoch'], color='#555555', linestyle=':', label='Checkpoint')
        axis.set(title=f"Fold {fold['fold']}", xlabel=translate('Época'), ylabel=translate('Validación (%)'), ylim=(0, 100))
    axes.flat[0].legend(fontsize=8)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(output / f'validation_training_curves{suffix}.png', dpi=config['figure_dpi'])
    plt.close(fig)


def write_report(run, result, output, config):
    labels = {'accuracy': 'Exactitud', 'balanced_accuracy': 'Exactitud equilibrada',
              'sensitivity': 'Sensibilidad / recall', 'specificity': 'Especificidad',
              'precision': 'Precisión / VPP', 'npv': 'Valor predictivo negativo',
              'f1_score': 'F1 clase patológica', 'f1_control': 'F1 clase control',
              'macro_f1': 'F1 macro', 'weighted_f1': 'F1 ponderado', 'roc_auc': 'ROC-AUC',
              'average_precision': 'Average precision (AP)', 'mcc': 'MCC', 'cohen_kappa': 'Kappa de Cohen',
              'brier_score': 'Brier score', 'log_loss': 'Log loss',
              'false_positive_rate': 'Tasa falsos positivos', 'false_negative_rate': 'Tasa falsos negativos'}
    lines = ['# Resultados científicos de la arquitectura 68e15b97', '',
             '## Resultado final en el test reservado', '',
             'Modelo fijado antes de consultar el test: checkpoint del fold 5, época 68, elegido por el mayor F1 de validación. '
             'No hubo entrenamiento adicional, elección de umbral ni ajustes después de observar el test. '
             'El modelo se entrenó con 56 personas; no es un modelo reentrenado con las 70 personas de desarrollo.', '',
             '| Métrica | Grabaciones: estimación [IC 95 %] | Personas: estimación [IC 95 %] |',
             '|---|---:|---:|']
    for key, label in labels.items():
        cells = []
        for population in ('test_recordings', 'test_people'):
            point = result[population]['metrics'][key]
            low, high = result[population]['confidence_intervals_95'][key]
            digits = 3 if key in {'mcc', 'cohen_kappa', 'brier_score', 'log_loss'} else 2
            cells.append(f'{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]')
        lines.append(f'| {label} | {cells[0]} | {cells[1]} |')
    lines += ['', 'Las métricas de proporción y las AUC se expresan en %. MCC, kappa, Brier y log loss conservan su escala propia. '
              'AP es average precision, no el área trapezoidal de la curva PR. Brier y log loss: menor es mejor.', '',
              'Para las grabaciones y las métricas compuestas, los IC son percentiles 2,5 y 97,5 de 2.000 remuestreos estratificados por clase a nivel de persona. '
              'Cada persona arrastra sus 15 grabaciones. En test se remuestrean 10 controles y 10 personas patológicas; '
              'no se tratan las 300 grabaciones como 300 sujetos independientes. Los IC son condicionales al modelo guardado '
              'y al balance de clases; no incluyen variabilidad por reentrenamiento. En personas, sensibilidad, especificidad, VPP, VPN '
              'y tasas de error usan IC binomiales exactos de Clopper–Pearson. Así, 10/10 positivos detectados no produce un IC '
              'engañoso de [100, 100]: el IC de sensibilidad es [69,15, 100] %. VPP y VPN corresponden al balance 50/50 de esta muestra, '
              'no a una prevalencia poblacional estimada.', '',
              'Para la evaluación por persona se promedian las probabilidades de sus 15 audios y se predice patología '
              f"cuando la media supera {get_scientific_evaluation_config()['person_probability_threshold']}. Esta agregación es un análisis secundario predefinido, "
              'no una regla optimizada con el test.', '',
              '![Matrices de confusión](test_confusion_matrices.png)', '',
              '![ROC y precisión-recall](test_roc_pr.png)', '',
              'Matriz de confusión por grabación (filas: real Control/Patologicas; columnas: predicción Control/Patologicas): '
              f"{result['test_recordings']['confusion_matrix']}. Por persona: {result['test_people']['confusion_matrix']}.", '',
              '## Validación cruzada interna: cinco folds', '',
              '| Fold | Exactitud | Sensibilidad | Especificidad | Precisión | F1 | ROC-AUC | Mejor época / total | Tiempo (min) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for fold in result['folds']:
        cells = [f"{fold[key]:.2f}" for key in ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc')]
        lines.append(f"| {fold['fold']} | " + ' | '.join(cells) +
                     f" | {fold['best_epoch']} / {fold['epochs_completed']} | {fold['training_seconds'] / 60:.2f} |")
    lines += ['', '| Métrica | Media ± DE muestral entre folds |', '|---|---:|']
    for key, label in labels.items():
        value = result['fold_mean_sample_sd'][key]
        digits = 3 if key in {'mcc', 'cohen_kappa', 'brier_score', 'log_loss'} else 2
        lines.append(f"| {label} | {value['mean']:.{digits}f} ± {value['sd']:.{digits}f} |")
    lines += ['', 'La DE usa n−1 (cinco folds). El JSON de entrenamiento usa n para la dispersión del F1: '
              f"{run['std_f1']:.2f} puntos. Son convenciones distintas; ±DE no es un intervalo de confianza.", '',
              'Cada fold evalúa 210 grabaciones de 14 personas (7 controles y 7 patológicas), '
              'entrenando con 840 grabaciones de otras 56 personas. Las predicciones fuera de fold reúnen '
              '1.050 grabaciones de 70 personas, cada audio evaluado solo por el modelo que no vio su persona al entrenar. '
              'Las métricas agregadas OOF y sus IC están en scientific_results.json y metrics_with_confidence_intervals.csv; '
              'la media de métricas por fold puede diferir de calcularlas sobre todas las predicciones juntas.', '',
              '![Curvas de validación por época](validation_training_curves.png)', '',
              '## Datos y protocolo', '',
              '100 personas, 50 Control y 50 Patologicas; 1.500 grabaciones, cinco vocales × tres repeticiones por persona. '
              'Partición estratificada y separada por persona: train 70 (1.050 audios), val global 10 (150), test 20 (300). '
              'Se verificaron ausencia de solapamientos de personas, etiquetas y orden de los arrays frente a los CSV y los .npy individuales. '
              'Clase positiva: Patologicas (1); negativa: Control (0).', '',
              'Entrada: log-mel de potencia, 64 × 648; audio mono 44.100 Hz, duración 6,5 s con recorte/relleno. '
              'STFT: n_fft=1.024, hop=441; 64 bandas mel entre 80 y 8.000 Hz; transformación log1p(1000 × mel_power). '
              'La duración se fijó con el percentil 95 de train, redondeado a décimas de segundo; 48/1.050 audios de train se recortaron. '
              'Los audios cortos se rellenan con ceros a ambos lados; para los largos se conserva la ventana de 6,5 s de máxima energía. '
              'Los filtros mel triangulares se normalizan por la suma de sus coeficientes; el PCM se escala por 32.768 antes de la STFT. '
              'No se aplicaron aumentos de datos en esta pasada.', '',
              '## Arquitectura y entrenamiento', '',
              'CNN 2D residual: cinco convoluciones, filtros [188, 221, 64, 205, 115], kernels [5, 7, 3, 15, 9]. '
              'Dos bloques residuales de dos convoluciones y una convolución final; proyecciones 1×1 cuando cambia el número de canales. '
              'Activaciones convolucionales cíclicas [ReLU, tanh, LeakyReLU, ReLU, tanh]; batch normalization. '
              'Max-pooling después de cada bloque y de la última convolución, dropout convolucional 0,1. '
              'Pooling medio adaptativo 4×8; capas densas [488, 278, 247] y salida de dos logits. '
              f"Activaciones de las densas ocultas: {format_fc_activations(run['genome'], run['config'])}; "
              'dropout denso 0,425527 entre capas ocultas. '
              '9.081.459 parámetros entrenables.', '',
              'AdamW, learning rate 5×10⁻⁵, weight decay 0,01; pérdida cross-entropy; batch 8. '
              'Betas por defecto de AdamW (0,9; 0,999), epsilon 10⁻⁸; sin pesos de clase. '
              'Máximo 100 épocas; validación cada época y selección del mejor F1, mejora mínima 0,01 puntos; '
              'parada tras 18 épocas sin mejora. Sin scheduler ni preentrenamiento. '
              'Una pasada de los cinco folds, secuencial, semillas 42–46 y pesos nuevos en cada fold.', '',
              'Predicciones por grabación: argmax de los dos logits; las probabilidades softmax se obtienen con AMP BF16 '
              'y se exportan como float32, reproduciendo el cálculo del entrenamiento. No se optimiza un umbral en test.', '',
              f"NVIDIA GeForce RTX 5080; PyTorch {run['torch_version']}; CUDA {run['cuda_version']}; AMP BF16. "
              f"Tiempo total: {run['elapsed_seconds'] / 60:.2f} min; {sum(f['metrics']['epochs_completed'] for f in run['folds'])} épocas ejecutadas. "
              'La configuración efectiva y los checkpoints se conservan en results.json y fold_N_best.pt.', '',
              f"El forward por lotes y las transferencias de los 300 audios de test tardaron {result['test_inference_seconds']:.3f} s "
              '(modelo ya cargado y GPU previamente usada). Esta medida excluye extracción de log-mel y carga desde disco; '
              'no es latencia de un sistema completo.', '',
              f"Entorno de análisis: {json.dumps(result['analysis_environment'], ensure_ascii=False)}.", '',
              '## Alcance e interpretación', '',
              f"F1 medio actual {run['mean_f1']:.2f} % frente a {run['original_mean_f1']:.2f} % en la ejecución original "
              f"(diferencia {run['mean_f1'] - run['original_mean_f1']:+.2f} puntos). Es una comparación descriptiva, no una prueba de significación. "
              'El estado del generador aleatorio del proceso evolutivo original no está disponible y su ejecución paralela '
              'no permite exigir una reproducción numérica exacta.', '',
              'La arquitectura ya se eligió usando estos folds: su validación interna mide desarrollo y puede ser optimista '
              'por selección de arquitectura y checkpoint. El test reservado es la evaluación independiente para el modelo fijado, '
              'pero solo contiene 20 personas. No hay repeticiones de entrenamiento, validación externa, comparación entrenada con baselines '
              'ni edades/sexo/gravedad clínica en los manifiestos examinados. Por ello no se atribuyen significación entre modelos ni validez clínica externa. '
              'No se registraron pérdidas de entrenamiento por época, energía ni FLOPs; no se inventan esos valores.', '',
              '## Resultados descriptivos por vocal en test', '',
              '| Vocal | Grabaciones | Exactitud (%) | Sensibilidad (%) | Especificidad (%) | F1 (%) | ROC-AUC (%) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for vowel in result['test_metrics_by_vowel']:
        lines.append(f"| {vowel['vowel']} | {vowel['samples']} | " + ' | '.join(
            f"{vowel[key]:.2f}" for key in ('accuracy', 'sensitivity', 'specificity', 'f1_score', 'roc_auc')) + ' |')
    lines += ['', 'Cortes descriptivos sin pruebas de significación ni selección posterior de vocales. '
              'Cada vocal representa a las mismas 20 personas, con tres repeticiones por persona.', '',
              '## Texto de resultados para un artículo', '']
    t, p = result['test_recordings']['metrics'], result['test_people']['metrics']
    lines.append(f"La arquitectura seleccionada alcanzó un F1 medio de {run['mean_f1']:.2f} % "
                 f"(DE muestral {result['fold_mean_sample_sd']['f1_score']['sd']:.2f}) en cinco folds internos separados por persona. "
                 f"El checkpoint seleccionado mediante validación obtuvo, en el test reservado de 20 participantes y 300 grabaciones, "
                 f"exactitud del {t['accuracy']:.2f} %, sensibilidad del {t['sensitivity']:.2f} %, especificidad del {t['specificity']:.2f} %, "
                 f"F1 del {t['f1_score']:.2f} % y ROC-AUC de {t['roc_auc'] / 100:.3f}. "
                 f"La agregación secundaria de las 15 grabaciones por participante obtuvo exactitud del {p['accuracy']:.2f} % "
                 f"y F1 del {p['f1_score']:.2f} %. Los intervalos de confianza se estimaron mediante bootstrap estratificado por participante.")
    lines += ['', '## Fuentes metodológicas y archivos auditables', '',
              '- [Definiciones de métricas, scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html).',
              '- [Selección de modelos y sesgo de validación no anidada, scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).',
              '- Datos: results.json, cinco fold_N_best.pt, manifest.csv, fold_N_val.csv, test.csv y arrays .npy locales.',
              '- Salidas: fold_metrics.csv; metrics_with_confidence_intervals.csv; predicciones de cada población; '
              'scientific_results.json; evaluation_protocol.json; figuras PNG.',
              '- Script reproducible: scripts/report_winner_scientific.py. No modifica los modelos ni vuelve a entrenarlos.', '']
    (output / 'informe_cientifico.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
