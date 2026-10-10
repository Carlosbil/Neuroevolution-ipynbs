"""English presentation of saved scientific results; no model evaluation."""
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from neuroevolution.config import get_scientific_evaluation_config
from neuroevolution.models.architecture_formatting import format_fc_activations
from report_winner_scientific import make_figures


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
    lines = ['# Scientific results for architecture 68e15b97', '',
             '## Final evaluation on the held-out test set', '',
             'The fold 5 checkpoint from epoch 68 was selected using the highest validation F1 before examining test predictions. '
             'There was no additional training, test-based threshold optimization, or model adjustment after observing test results. '
             'This checkpoint was trained on 56 participants; it was not retrained on all 70 development participants.', '',
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
              'class-stratified participant-cluster bootstrap samples. Resampling a participant includes all 15 recordings. '
              'The test bootstrap samples 10 controls and 10 pathological participants with replacement; the 300 recordings are not '
              'treated as 300 independent subjects. These intervals are conditional on the saved model and class balance and do not '
              'include retraining variability.', '',
              'Participant-level sensitivity, specificity, predictive values and error rates use exact Clopper–Pearson binomial intervals. '
              'Detecting 10/10 positive participants therefore gives a sensitivity CI of [69.15%, 100%], rather than the misleading '
              '[100%, 100%] percentile-bootstrap interval. Predictive values correspond to this balanced 50/50 sample, not an estimated population prevalence.', '',
              'Participant-level predictions average the probabilities of all 15 recordings and assign the pathological class when '
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
    lines += ['', 'Sample SD uses n−1 for the five folds. The training JSON uses n and reports an F1 dispersion of 7.86 percentage points. '
              'These are different conventions; mean ± SD is not a confidence interval.', '',
              'Each fold trained on 840 recordings from 56 participants and validated on 210 recordings from 14 different participants '
              '(7 controls and 7 pathological participants). The out-of-fold predictions cover 1,050 recordings from 70 participants, '
              'each evaluated only by the model whose training excluded that participant. Pooled OOF metrics and their confidence intervals '
              'are included in scientific_results.json and metrics_with_confidence_intervals.csv. Mean fold metrics can differ from metrics '
              'computed on pooled predictions.', '',
              '![Validation trajectories](validation_training_curves_en.png)', '',
              '## Dataset and preprocessing', '',
              'The dataset contains 100 participants: 50 controls and 50 pathological participants, with 1,500 recordings '
              '(five vowels × three repetitions per participant). The participant-disjoint, stratified split contains 70 training '
              'participants (1,050 recordings), 10 global-validation participants (150 recordings), and 20 test participants (300 recordings). '
              'The global-validation subset was not used in this one-pass training or checkpoint selection. '
              'Checks confirmed disjoint participant sets, labels, and the exact order of arrays against CSV manifests and individual .npy files. '
              'The positive class is Patologicas (1), and the negative class is Control (0).', '',
              'Inputs are 64 × 648 log-mel power spectrograms from mono audio sampled at 44,100 Hz and normalized to 6.5 s. '
              'Duration was fixed using the training-set 95th percentile rounded up to a tenth of a second; 48/1,050 training recordings '
              'were cropped. Short recordings are zero-padded on both sides; longer recordings retain the highest-energy 6.5 s window. '
              'PCM samples are divided by 32,768. STFT uses n_fft=1,024 and hop=441; 64 triangular mel filters cover 80–8,000 Hz, '
              'each normalized by its coefficient sum. The final transform is log1p(1000 × mel_power). No data augmentation was used.', '',
              '## Architecture and training', '',
              'The residual 2D CNN has five convolutions with filters [188, 221, 64, 205, 115] and kernel sizes [5, 7, 3, 15, 9]. '
              'There are two residual blocks of two convolutions followed by one final convolution. Channel changes use 1×1 projections. '
              'Convolutional activations are [ReLU, tanh, LeakyReLU, ReLU, tanh], with batch normalization. Max-pooling follows each block '
              'and the final convolution; convolutional dropout is 0.1. Adaptive average pooling produces 4×8 feature maps. '
              'The dense layers have [488, 278, 247] nodes, followed by two output logits. '
              f"Hidden dense activations are {format_fc_activations(run['genome'], run['config'])}; "
              'dropout between hidden dense layers is 0.425527. The model has 9,081,459 trainable parameters.', '',
              'Training used AdamW with learning rate 5×10⁻⁵, weight decay 0.01, default betas (0.9, 0.999), epsilon 10⁻⁸, '
              'unweighted cross-entropy loss, and batch size 8. The limit was 100 epochs, with validation every epoch and checkpoint '
              'selection by the highest validation F1. Early stopping used patience 18 and a minimum improvement of 0.01 percentage points. '
              'No learning-rate scheduler or pretrained weights were used. One sequential pass covered all five folds, with seeds 42–46 '
              'and freshly initialized weights in each fold.', '',
              'Recording-level predictions use argmax of the two logits. Softmax probabilities are computed under AMP BF16 and exported '
              'as float32, reproducing the training evaluation. No threshold was tuned on the test set.', '',
              f"Hardware/runtime: NVIDIA GeForce RTX 5080, PyTorch {run['torch_version']}, CUDA {run['cuda_version']}, AMP BF16. "
              f"Total training duration was {run['elapsed_seconds'] / 60:.2f} min across {sum(f['metrics']['epochs_completed'] for f in run['folds'])} epochs. "
              'Effective settings and checkpoints are preserved in results.json and fold_N_best.pt.', '',
              f"Batched forward passes and transfers for the 300 test recordings took {result['test_inference_seconds']:.3f} s, "
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
              'The main observed weakness is false-positive classification: 68/150 control recordings and 4/10 control participants '
              'were predicted as pathological.', '',
              '## Descriptive test results by vowel', '',
              '| Vowel | Recordings | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1 (%) | ROC-AUC (%) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for vowel in result['test_metrics_by_vowel']:
        lines.append(f"| {vowel['vowel']} | {vowel['samples']} | " + ' | '.join(
            f'{vowel[key]:.2f}' for key in ('accuracy', 'sensitivity', 'specificity', 'f1_score', 'roc_auc')) + ' |')
    lines += ['', 'These are descriptive subgroups without significance tests or subsequent vowel selection. '
              'Each vowel includes the same 20 participants, with three repetitions each.', '',
              '## Results paragraph for a manuscript', '',
              'The selected architecture achieved a mean F1 score of 76.40% (sample SD 8.79 percentage points) across five internal, '
              'participant-disjoint folds. The validation-selected checkpoint achieved 73.00% accuracy, 91.33% sensitivity, '
              '54.67% specificity, 77.18% F1, and a ROC-AUC of 0.786 on the held-out test set of 20 participants and 300 recordings. '
              'A predefined secondary analysis averaging the 15 recording probabilities per participant achieved 80.00% accuracy '
              'and 83.33% F1. Confidence intervals used participant-cluster bootstrap resampling, with exact binomial intervals '
              'for participant-level sensitivity, specificity, predictive values and error rates.', '',
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
