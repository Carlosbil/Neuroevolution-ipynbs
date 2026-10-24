"""Compare audio predictions with a majority vote over each person's recordings."""

from collections import defaultdict

import numpy as np

from ..evolution.fitness import compute_classification_metrics


COMPARISON_METRICS = ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'auc')


def build_audio_predictions(rows, predictions):
    """Bind sequential test-loader predictions to the corresponding CSV rows."""
    targets, labels, probabilities = (predictions[key] for key in ('y_true', 'y_pred', 'y_prob'))
    if not (len(rows) == len(targets) == len(labels) == len(probabilities)):
        raise ValueError('Las predicciones y las filas de test.csv deben tener la misma longitud')
    records = []
    for index, (row, target, label, probability) in enumerate(zip(rows, targets, labels, probabilities)):
        if target != int(row['label_binary']):
            raise ValueError(f'Orden de etiquetas distinto de test.csv en la fila {index}')
        if target not in (0, 1) or label not in (0, 1):
            raise ValueError('Las etiquetas y predicciones deben ser binarias (0 o 1)')
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('La probabilidad patológica debe ser finita y estar entre 0 y 1')
        if not row.get('person_id'):
            raise ValueError(f'Falta person_id en la fila {index} de test.csv')
        records.append({
            'test_index': index, 'person_id': row['person_id'],
            'audio_id': row.get('source_audio_id') or row.get('source_path') or f'test_row_{index}',
            'origin': row.get('origin'), 'repetition': row.get('repetition'),
            'label_binary': int(target), 'predicted_label': int(label),
            'probability_pathological': float(probability), 'correct': bool(target == label),
        })
    return records


def _metrics(targets, predictions, probabilities):
    result = compute_classification_metrics(targets, predictions, probabilities)
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    result['confusion_matrix'] = [
        [int(np.sum((targets == truth) & (predictions == predicted))) for predicted in (0, 1)]
        for truth in (0, 1)
    ]
    result['n_samples'] = len(targets)
    return result


def compare_audio_and_person(results, *, expected_audios_per_person=3):
    """Use a fixed vote within each run; aggregate paired metrics across runs."""
    runs = results.get('run_results', [results])
    if not runs:
        raise ValueError('No hay repeticiones para comparar')
    compared_runs = []
    reference_identity = None
    for index, run in enumerate(runs, start=1):
        records = run.get('audio_predictions')
        if not records:
            raise ValueError(
                'Estos resultados no contienen predicciones por audio. '
                'Vuelve a ejecutar la sección 8 con el código actualizado y después esta celda.'
            )
        groups = defaultdict(list)
        identity = []
        for record in records:
            person = record.get('person_id')
            if not person or record.get('origin') != 'real':
                raise ValueError('La comparación necesita person_id y solo audios reales de test')
            if record['label_binary'] not in (0, 1) or record['predicted_label'] not in (0, 1):
                raise ValueError('Las etiquetas y predicciones deben ser binarias (0 o 1)')
            probability = record['probability_pathological']
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('La probabilidad patológica debe ser finita y estar entre 0 y 1')
            groups[person].append(record)
            identity.append((person, record['audio_id'], record['label_binary']))
        if reference_identity is not None and identity != reference_identity:
            raise ValueError('Las repeticiones deben evaluar los mismos audios, personas y etiquetas en el mismo orden')
        reference_identity = identity
        people = []
        for person, audios in sorted(groups.items()):
            if len(audios) != expected_audios_per_person or len(audios) % 2 == 0:
                raise ValueError(f'{person}: se esperaban {expected_audios_per_person} audios reales, recibidos {len(audios)}')
            if len({audio['audio_id'] for audio in audios}) != len(audios):
                raise ValueError(f'{person}: hay audios duplicados')
            if len({audio['label_binary'] for audio in audios}) != 1:
                raise ValueError(f'{person}: sus audios tienen etiquetas reales contradictorias')
            target = audios[0]['label_binary']
            votes = sum(audio['predicted_label'] for audio in audios)
            prediction = int(votes > len(audios) / 2)
            correct_audios = sum(audio['predicted_label'] == target for audio in audios)
            people.append({
                'person_id': person, 'label_binary': target, 'predicted_label': prediction,
                'n_audios': len(audios), 'correct_audios': correct_audios,
                'incorrect_audios': len(audios) - correct_audios,
                'votes_pathological': votes, 'votes_control': len(audios) - votes,
                'correct': prediction == target,
                'probability_pathological': float(np.mean([audio['probability_pathological'] for audio in audios])),
                'audio_predictions': [audio['predicted_label'] for audio in audios],
                'audio_ids': [audio['audio_id'] for audio in audios],
            })
        audio_metrics = _metrics(
            [audio['label_binary'] for audio in records], [audio['predicted_label'] for audio in records],
            [audio['probability_pathological'] for audio in records],
        )
        person_metrics = _metrics(
            [person['label_binary'] for person in people], [person['predicted_label'] for person in people],
            [person['probability_pathological'] for person in people],
        )
        compared_runs.append({
            'run': run.get('run', index), 'seed': run.get('seed'),
            'audio_metrics': audio_metrics, 'person_metrics': person_metrics, 'people': people,
        })
    ddof = 1 if len(compared_runs) > 1 else 0
    summaries = {}
    for level in ('audio', 'person', 'difference_person_minus_audio'):
        summaries[level] = {}
        for metric in COMPARISON_METRICS:
            values = [
                run['person_metrics'][metric] - run['audio_metrics'][metric]
                if level == 'difference_person_minus_audio' else run[f'{level}_metrics'][metric]
                for run in compared_runs
            ]
            summaries[level][metric] = {
                'values': values, 'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=ddof)),
            }
    return {
        'protocol': 'audio_vs_person_majority_vote_on_same_test',
        'n_runs': len(compared_runs), 'std_ddof': ddof,
        'n_audios': len(records), 'n_people': len(people),
        'audios_per_person': expected_audios_per_person,
        'positive_label': 'Patologico (1)',
        'person_decision': 'strict_majority_of_audio_predictions',
        'person_auc_score': 'mean_probability_pathological_over_person_audios',
        'metrics_summary': summaries, 'run_results': compared_runs,
    }
