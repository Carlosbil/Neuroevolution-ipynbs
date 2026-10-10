"""Compare saved results with the user's reference screenshot; no training."""
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from neuroevolution.config import get_scientific_evaluation_config


def main():
    output = Path(get_scientific_evaluation_config()['report_directory'])
    results = json.loads((output / 'scientific_results.json').read_text(encoding='utf-8'))
    image_path = Path('C:/Users/carlo/AppData/Local/Temp/codex-clipboard-255416e3-63ee-41e3-8caf-75d406988521.png')
    keys = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'roc_auc']
    names = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'AUC']
    references = {
        'ResNet': [70., 63., 76., 67., 70.],
        'LSTM-FCN': [72., 78., 67., 73., 72.],
        'InceptionTime': [69., 71., 67., 69., 69.],
        'CDIL-CNN': [73., 82., 63., 75., 73.],
    }
    parenthetical = {
        'ResNet': [7, 11, 12, 7, None],
        'LSTM-FCN': [8, 21, 18, 11, None],
        'InceptionTime': [18, 19, 16, 19, None],
        'CDIL-CNN': [13, 17, 20, 13, 13],
    }
    provenance = {
        'source': str(image_path), 'source_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
        'score_units': 'percent; screenshot decimal scores multiplied by 100',
        'reference_evaluation_protocol': 'not provided',
        'reference_auc_definition': 'not specified in screenshot',
        'parenthetical_statistic_definition': 'not specified in screenshot; not assumed to be SD or CI',
        'missing_data': 'AUC parenthetical values for first three models are cropped; CDIL-CNN AUC variability is visible in target table',
        'references': {model: {key: {'score_percent': score, 'parenthetical_percent': spread}
                              for key, score, spread in zip(keys, scores, parenthetical[model])}
                       for model, scores in references.items()},
    }
    (output / 'reference_results_transcribed.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    rows = [{'Model / evaluation': model, **dict(zip(names, scores))} for model, scores in references.items()]
    cv = {key: results['fold_mean_sample_sd'][key]['mean'] for key in keys}
    test = {key: results['test_recordings']['metrics'][key] for key in keys}
    for name, values in (('Proposed: internal 5-fold CV', cv), ('Proposed: held-out test, recordings', test)):
        rows.append({'Model / evaluation': name, **{label: values[key] for key, label in zip(keys, names)}})
    pd.DataFrame(rows).to_csv(output / 'reference_model_comparison.csv', index=False)
    targets, detailed = [], []
    for position, (key, name) in enumerate(zip(keys, names)):
        best_model = max(references, key=lambda model: references[model][position])
        target = references[best_model][position]
        targets.append({
            'Metric': name, 'Target (%)': target, 'Reference': best_model,
            'CV (%)': cv[key], 'CV difference (pp)': cv[key] - target,
            'CV status': 'Above' if cv[key] > target else 'Equal' if cv[key] == target else 'Below',
            'Test (%)': test[key], 'Test difference (pp)': test[key] - target,
            'Test status': 'Above' if test[key] > target else 'Equal' if test[key] == target else 'Below',
        })
        for model, scores in references.items():
            for protocol, values in (('Internal CV', cv), ('Held-out test, recordings', test)):
                detailed.append({'metric': name, 'reference_model': model, 'reference_percent': scores[position],
                                 'proposed_evaluation': protocol, 'proposed_percent': values[key],
                                 'difference_percentage_points': values[key] - scores[position]})
    pd.DataFrame(targets).to_csv(output / 'target_comparison.csv', index=False)
    pd.DataFrame(detailed).to_csv(output / 'all_reference_differences.csv', index=False)
    lines = [
        '# Comparison with the supplied reference results', '',
        '## Table 1. Reported point estimates and proposed-model results', '',
        'All values are percentages, including AUC (e.g., 0.73 = 73.00%). '
        'Reference values are transcribed from the supplied screenshot, not from independently rerun models. '
        'Internal CV and held-out test results are distinct evaluations.', '',
        '| Model / evaluation | Accuracy | Sensitivity | Specificity | F1-score | AUC |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f"| {row['Model / evaluation']} | " + ' | '.join(f'{row[name]:.2f}' for name in names) + ' |')
    lines += ['', 'Proposed-model internal CV sample SDs: ' + '; '.join(
        f"{name} {results['fold_mean_sample_sd'][key]['sd']:.2f} percentage points" for key, name in zip(keys, names)) + '.', '',
        '## Table 2. Differences from the strongest listed target for each metric', '',
        'Positive differences indicate a higher reported point estimate. Differences are percentage points, not relative percentage improvements.', '',
        '| Metric | Target (%) | Reference model | Internal CV (%) | CV Δ (pp) | Held-out test (%) | Test Δ (pp) |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in targets:
        lines.append(f"| {row['Metric']} | {row['Target (%)']:.2f} | {row['Reference']} | "
                     f"{row['CV (%)']:.2f} | {row['CV difference (pp)']:+.2f} | "
                     f"{row['Test (%)']:.2f} | {row['Test difference (pp)']:+.2f} |")
    lines += ['', '## Table 3. Target assessment at the reported point-estimate level', '',
              '| Metric | Internal CV | Held-out test |', '|---|---|---|']
    for row in targets:
        lines.append(f"| {row['Metric']} | {row['CV status']} | {row['Test status']} |")
    cv_above = sum(row['CV status'] == 'Above' for row in targets)
    test_above = sum(row['Test status'] == 'Above' for row in targets)
    lines += ['', f'Internal CV is numerically above {cv_above}/5 listed targets. The held-out recording-level test is above '
              f'{test_above}/5, equals the accuracy target, and is below the specificity target. '
              'The model does not exceed all five targets.', '',
              'The specificity shortfall is 13.90 percentage points in internal CV and 21.33 points on the held-out test. '
              'Specificity is also below every individual reference model shown. The internal-CV accuracy advantage is only 0.24 '
              'percentage points against a reference rounded to two decimal places on the 0–1 scale.', '',
              '## Table 4. Reference variability exactly as displayed', '',
              '| Reference model | Accuracy | Sensitivity | Specificity | F1-score | AUC |',
              '|---|---:|---:|---:|---:|---:|']
    for model, values in parenthetical.items():
        cells = [f'{value}%' if value is not None else 'Not visible' for value in values]
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'These are the parenthetical percentages in the screenshot. Its statistical definition is not supplied, '
              'so they are not labeled as SD, confidence intervals or error rates. The first three AUC variability values are cropped. '
              'The CDIL-CNN AUC value of 13% is visible in the minimum-target table.', '',
              '## Scientific interpretation', '',
              'The comparison establishes numerical differences between reported point estimates, not statistically significant superiority. '
              'The screenshot does not specify the reference dataset, participant split, evaluation unit, cross-validation protocol, '
              'or AUC definition. The proposed model reports ROC-AUC. These details must match before making a direct scientific comparison.', '',
              'The proposed internal folds participated in architecture and checkpoint selection. The held-out test evaluated a fixed '
              'checkpoint on 20 previously unseen participants and 300 recordings. These two estimates must not be presented as the '
              'same protocol or combined into a single result.', '',
              'The secondary participant-level results are not substituted into the baseline ranking because the reference evaluation '
              'unit is unknown. Participant-level accuracy = 80.00%, sensitivity = 100.00%, specificity = 60.00%, F1 = 83.33%, '
              'and ROC-AUC = 88.00%.', '',
              '## Suggested manuscript wording', '',
              'Compared numerically with the supplied reference values, the proposed architecture achieved higher internal '
              'cross-validation point estimates for accuracy, sensitivity, F1-score and AUC, while specificity remained below '
              'the reference target. On the held-out recording-level test, sensitivity, F1-score and ROC-AUC exceeded the listed '
              'targets, accuracy matched the target, and specificity was lower. These comparisons are descriptive; statistical '
              'superiority is not established because matched reference protocols and predictions are unavailable.', '',
              '## Auditable files', '',
              '- reference_results_transcribed.json: source image fingerprint, transcription, units and missing values.',
              '- reference_model_comparison.csv: all model point estimates.',
              '- target_comparison.csv: strongest listed target, differences and statuses.',
              '- all_reference_differences.csv: differences against each reference model.',
              '- Source for proposed results: scientific_results.json.', '']
    report = output / 'benchmark_comparison_en.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    with ZipFile(output.parent / 'benchmark_comparison_en.zip', 'w', ZIP_DEFLATED) as archive:
        for name in ('benchmark_comparison_en.md', 'reference_results_transcribed.json',
                     'reference_model_comparison.csv', 'target_comparison.csv', 'all_reference_differences.csv'):
            archive.write(output / name, name)
    print(json.dumps(targets, indent=2))
    print(report)


if __name__ == '__main__':
    main()
