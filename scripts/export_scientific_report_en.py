"""English presentation of saved scientific results; no model evaluation."""
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from neuroevolution.config import get_scientific_evaluation_config
from neuroevolution.models.architecture_formatting import format_fc_activations
from report_winner_scientific import make_figures, saved_run_descriptions


def main():
    settings = get_scientific_evaluation_config()
    output = Path(settings['report_directory'])
    result = json.loads((output / 'scientific_results.json').read_text(encoding='utf-8'))
    run = json.loads(Path(settings['run_results_path']).read_text(encoding='utf-8'))
    datasets = {key: pd.read_csv(output / f'{key}_predictions.csv') for key in
                ('oof_recordings', 'oof_people', 'test_recordings', 'test_people')}
    make_figures(run, datasets, output, settings, language='en')
    labels = {
        'accuracy': 'Accuracy', 'balanced_accuracy': 'Balanced accuracy',
        'sensitivity': 'Sensitivity / recall', 'specificity': 'Specificity',
        'precision': 'Precision / positive predictive value', 'npv': 'Negative predictive value',
        'f1_score': 'F1, pathological class', 'f1_control': 'F1, control class',
        'macro_f1': 'Macro F1', 'weighted_f1': 'Weighted F1', 'roc_auc': 'ROC-AUC',
        'average_precision': 'Average precision (AP)', 'mcc': 'Matthews correlation coefficient',
        'cohen_kappa': "Cohen's kappa", 'brier_score': 'Brier score', 'log_loss': 'Log loss',
        'false_positive_rate': 'False-positive rate', 'false_negative_rate': 'False-negative rate',
    }
    descriptions = saved_run_descriptions(run, result, language='en')
    lines = ['# Scientific results for architecture 68e15b97', '',
             '## Final evaluation on the held-out test set', '',
             descriptions['selection'], '',
             '| Metric | Recordings: estimate [95% CI] | Participants: estimate [95% CI] |',
             '|---|---:|---:|']
    for key, label in labels.items():
        cells = []
        for population in ('test_recordings', 'test_people'):
            value = result[population]['metrics'][key]
            lower, upper = result[population]['confidence_intervals_95'][key]
            digits = 3 if key in {'mcc', 'cohen_kappa', 'brier_score', 'log_loss'} else 2
            cells.append(f'{value:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]')
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Proportion-based metrics and AUCs are expressed as percentages. MCC, kappa, Brier score and log loss retain their native scales. '
              'AP means average precision, not trapezoidal PR-AUC. Lower Brier score and log loss are better.', '',
              'For recording-level and composite metrics, 95% confidence intervals are the 2.5th and 97.5th percentiles of 2,000 '
              'class-stratified participant-cluster bootstrap samples. Resampling a participant includes all their recordings. '
              'The test bootstrap samples 10 controls and 10 pathological participants with replacement; the recordings are not '
              'treated as independent subjects. These intervals are conditional on the saved model and class balance and do not '
              'include retraining variability.', '',
              'Participant-level sensitivity, specificity, predictive values and error rates use exact Clopper–Pearson binomial intervals. '
              'Detecting 10/10 positive participants therefore gives a sensitivity CI of [69.15%, 100%], rather than the misleading '
              '[100%, 100%] percentile-bootstrap interval. Predictive values correspond to this balanced 50/50 sample, not an estimated population prevalence.', '',
              'Participant-level predictions average the probabilities of all their recordings and assign the pathological class when '
              'the mean exceeds 0.5. This was a predefined secondary analysis, not a rule optimized on the test set.', '',
              '![Test confusion matrices](test_confusion_matrices_en.png)', '',
              '![Test ROC and precision–recall curves](test_roc_pr_en.png)', '',
              'Confusion matrices use rows = true Control/Pathological and columns = predicted Control/Pathological. '
              f"Recordings: {result['test_recordings']['confusion_matrix']}; participants: {result['test_people']['confusion_matrix']}.", '',
              '## Internal five-fold cross-validation', '',
              '| Fold | Accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC | Best epoch / total | Training time (min) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for fold in result['folds']:
        cells = [f'{fold[key]:.2f}' for key in ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc')]
        lines.append(f"| {fold['fold']} | " + ' | '.join(cells) +
                     f" | {fold['best_epoch']} / {fold['epochs_completed']} | {fold['training_seconds'] / 60:.2f} |")
    lines += ['', '| Metric | Mean ± sample SD across folds |', '|---|---:|']
    for key, label in labels.items():
        value = result['fold_mean_sample_sd'][key]
        digits = 3 if key in {'mcc', 'cohen_kappa', 'brier_score', 'log_loss'} else 2
        lines.append(f"| {label} | {value['mean']:.{digits}f} ± {value['sd']:.{digits}f} |")
    lines += ['', f"Sample SD uses n−1 for the five folds. The training JSON uses n and reports an F1 dispersion of {run['std_f1']:.2f} percentage points. "
              'These are different conventions; mean ± SD is not a confidence interval.', '',
              descriptions['cv'], '',
              '![Validation trajectories](validation_training_curves_en.png)', '',
              '## Dataset and preprocessing', '',
              descriptions['data'], '',
              descriptions['input'], '',
              '## Architecture and training', '',
              descriptions['architecture'], '',
              descriptions['training'], '',
              'Recording-level predictions use argmax of the two logits. Softmax probabilities are computed under AMP BF16 and exported '
              'as float32, reproducing the training evaluation. No threshold was tuned on the test set.', '',
              f"Hardware/runtime: {run['device']}, PyTorch {run['torch_version']}, CUDA {run['cuda_version']}, AMP BF16. "
              f"Total training duration was {run['elapsed_seconds'] / 60:.2f} min across {sum(f['metrics']['epochs_completed'] for f in run['folds'])} epochs. "
              'Effective settings and checkpoints are preserved in results.json and fold_N_best.pt.', '',
              f"Batched forward passes and transfers for the {result['test_recordings']['metrics']['samples']} test recordings took {result['test_inference_seconds']:.3f} s, "
              'with the model already loaded and the GPU previously used. This excludes feature extraction and disk loading '
              'and is not end-to-end system latency.', '',
              f"Analysis environment: {json.dumps(result['analysis_environment'])}.", '',
              '## Interpretation and limitations', '',
              f"Mean F1 was {run['mean_f1']:.2f}% versus {run['original_mean_f1']:.2f}% in the original run "
              f"({run['mean_f1'] - run['original_mean_f1']:+.2f} percentage points). This is a descriptive comparison, not a significance test. "
              'The original evolution-run random-generator state is unavailable, and its parallel execution prevents requiring exact numerical reproduction.', '',
              'The architecture was selected using these folds, so internal validation is a development result that may be optimistic '
              'because of architecture and checkpoint selection. The held-out test independently evaluates the fixed checkpoint but contains '
              'only 20 participants. There were no repeated training runs, external validation, trained baseline comparisons, or age, sex '
              'and clinical-severity fields in the examined manifests. Comparative statistical significance and external clinical validity '
              'are therefore not established. Per-epoch training loss, energy use and FLOPs were not recorded.', '',
              f"Observed false positives: recordings {result['test_recordings']['metrics']['fp']}; participants {result['test_people']['metrics']['fp']}.", '',
              '## Descriptive test results by vowel', '',
              '| Vowel | Recordings | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1 (%) | ROC-AUC (%) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for vowel in result['test_metrics_by_vowel']:
        lines.append(f"| {vowel['vowel']} | {vowel['samples']} | " + ' | '.join(
            f'{vowel[key]:.2f}' for key in ('accuracy', 'sensitivity', 'specificity', 'f1_score', 'roc_auc')) + ' |')
    lines += ['', 'These are descriptive subgroups without significance tests or subsequent vowel selection. '
              'Each vowel includes the same 20 participants, with three repetitions each.', '',
              '## Results paragraph for a manuscript', '',
              f"Mean internal-validation F1: {run['mean_f1']:.2f}% (sample SD {result['fold_mean_sample_sd']['f1_score']['sd']:.2f}). "
              f"Held-out test: {result['test_people']['metrics']['people']} participants, {result['test_recordings']['metrics']['samples']} recordings. "
              f"Accuracy {result['test_recordings']['metrics']['accuracy']:.2f}%; F1 {result['test_recordings']['metrics']['f1_score']:.2f}%; "
              f"ROC-AUC {result['test_recordings']['metrics']['roc_auc'] / 100:.3f}. "
              f"Participant-level accuracy {result['test_people']['metrics']['accuracy']:.2f}%; F1 {result['test_people']['metrics']['f1_score']:.2f}%.", '',
              '## Methodological references and auditable outputs', '',
              '- [Metric definitions, scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html).',
              '- [Model selection and non-nested validation bias, scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).',
              '- Evidence: results.json, five fold_N_best.pt checkpoints, manifest.csv, fold_N_val.csv, test.csv, and local .npy arrays.',
              '- Outputs: fold_metrics.csv; classification_reports.csv; metrics_with_confidence_intervals.csv; '
              'test_metrics_by_vowel.csv; prediction tables; scientific_results.json; evaluation_protocol.json; PNG figures.',
              '- Reproducible analysis: scripts/report_winner_scientific.py. English export: scripts/export_scientific_report_en.py.', '']
    target = output / 'scientific_report_en.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    with ZipFile(output.parent / 'scientific_report_en.zip', 'w', ZIP_DEFLATED) as archive:
        archive.write(target, target.name)
        for path in sorted(output.iterdir()):
            if path.suffix in {'.csv', '.json'} or path.name.endswith('_en.png'):
                archive.write(path, path.name)
        archive.write(Path(settings['run_results_path']), 'training_results.json')
    print(target)


if __name__ == '__main__':
    main()
