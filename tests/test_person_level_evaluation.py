"""Majority voting must group by person and keep repeated models separate."""

from copy import deepcopy

import pytest

from neuroevolution.evaluation.person_level import build_audio_predictions, compare_audio_and_person


def example_run():
    # Interleave the people: grouping consecutive triples would be incorrect.
    people = [('control-a', 0, [0, 0, 1]), ('control-b', 0, [0, 0, 0]),
              ('patient-a', 1, [1, 1, 0]), ('patient-b', 1, [1, 1, 0])]
    rows, targets, predictions = [], [], []
    for repetition in range(3):
        for person, label, predicted in people:
            rows.append({'person_id': person, 'label_binary': str(label), 'origin': 'real',
                         'repetition': str(repetition + 1), 'source_audio_id': f'{person}-{repetition}'})
            targets.append(label)
            predictions.append(predicted[repetition])
    records = build_audio_predictions(rows, {
        'y_true': targets, 'y_pred': predictions, 'y_prob': [0.8 if p else 0.2 for p in predictions],
    })
    return {'run': 1, 'seed': 42, 'audio_predictions': records}


def test_person_majority_improves_metrics_and_keeps_audio_errors():
    comparison = compare_audio_and_person(example_run())
    run = comparison['run_results'][0]
    assert comparison['n_people'] == 4 and comparison['n_audios'] == 12
    assert comparison['std_ddof'] == 0
    assert run['audio_metrics']['accuracy'] == 75
    assert run['audio_metrics']['f1_score'] == pytest.approx(100 * 8 / 11)
    assert run['audio_metrics']['confusion_matrix'] == [[5, 1], [2, 4]]
    for metric in ('accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score'):
        assert run['person_metrics'][metric] == 100
    assert run['person_metrics']['confusion_matrix'] == [[2, 0], [0, 2]]
    person = next(person for person in run['people'] if person['person_id'] == 'patient-a')
    assert person['audio_predictions'] == [1, 1, 0]
    assert person['correct_audios'] == 2 and person['incorrect_audios'] == 1
    assert person['votes_pathological'] == 2 and person['predicted_label'] == 1 and person['correct']


def test_decision_uses_majority_even_when_mean_probability_is_below_half():
    run = example_run()
    patient = [audio for audio in run['audio_predictions'] if audio['person_id'] == 'patient-a']
    for audio, probability in zip(patient, [0.51, 0.51, 0.01]):
        audio['probability_pathological'] = probability
    compared = compare_audio_and_person(run)['run_results'][0]
    person = next(person for person in compared['people'] if person['person_id'] == 'patient-a')
    assert person['probability_pathological'] == pytest.approx(1.03 / 3)
    assert person['predicted_label'] == 1
    # Person AUC ranks mean probabilities rather than hard majority labels.
    assert compared['person_metrics']['auc'] == pytest.approx(75)


def test_control_with_two_pathological_votes_is_a_false_positive():
    run = example_run()
    control = [audio for audio in run['audio_predictions'] if audio['person_id'] == 'control-a']
    control[0]['predicted_label'] = 1
    person = next(person for person in compare_audio_and_person(run)['run_results'][0]['people']
                  if person['person_id'] == 'control-a')
    assert person['predicted_label'] == 1 and not person['correct']
    assert person['correct_audios'] == 1 and person['incorrect_audios'] == 2


def test_summary_uses_paired_repetitions_without_pooling_models():
    first = example_run()
    perfect = deepcopy(first)
    perfect.update(run=2, seed=43)
    for audio in perfect['audio_predictions']:
        audio['predicted_label'] = audio['label_binary']
        audio['probability_pathological'] = 0.9 if audio['label_binary'] else 0.1
    comparison = compare_audio_and_person({'run_results': [first, perfect]})
    summary = comparison['metrics_summary']
    assert comparison['n_runs'] == 2 and comparison['std_ddof'] == 1
    assert summary['audio']['accuracy']['values'] == [75, 100]
    assert summary['audio']['accuracy']['mean'] == 87.5
    assert summary['audio']['accuracy']['std'] == pytest.approx(25 / 2 ** 0.5)
    assert summary['person']['accuracy'] == {'values': [100, 100], 'mean': 100, 'std': 0}
    assert summary['difference_person_minus_audio']['accuracy']['values'] == [25, 0]
    assert summary['difference_person_minus_audio']['accuracy']['mean'] == 12.5
    assert summary['difference_person_minus_audio']['accuracy']['std'] == pytest.approx(25 / 2 ** 0.5)


@pytest.mark.parametrize('problem,match', [
    ('missing_audio', 'se esperaban 3'), ('mixed_labels', 'contradictorias'),
    ('generated', 'solo audios reales'), ('duplicate', 'duplicados'),
    ('invalid_probability', 'probabilidad'), ('missing_person', 'person_id'),
])
def test_invalid_person_groups_are_rejected(problem, match):
    run = example_run()
    records = run['audio_predictions']
    if problem == 'missing_audio':
        records.pop()
    elif problem == 'mixed_labels':
        records[0]['label_binary'] = 1
    elif problem == 'generated':
        records[0]['origin'] = 'generated'
    elif problem == 'duplicate':
        records[4]['audio_id'] = records[0]['audio_id']
    elif problem == 'invalid_probability':
        records[0]['probability_pathological'] = float('nan')
    elif problem == 'missing_person':
        records[0]['person_id'] = ''
    with pytest.raises(ValueError, match=match):
        compare_audio_and_person(run)


def test_different_test_recordings_between_repetitions_are_rejected():
    first = example_run()
    second = deepcopy(first)
    second['audio_predictions'][0]['audio_id'] = 'another-recording'
    with pytest.raises(ValueError, match='mismos audios'):
        compare_audio_and_person({'run_results': [first, second]})


def test_old_results_explain_that_predictions_must_be_saved():
    with pytest.raises(ValueError, match='sección 8'):
        compare_audio_and_person({'run_results': [{'accuracy': 75}]})


def test_binding_predictions_checks_csv_order_and_lengths():
    rows = [{'person_id': 'one', 'label_binary': '0', 'origin': 'real'}]
    with pytest.raises(ValueError, match='misma longitud'):
        build_audio_predictions(rows, {'y_true': [], 'y_pred': [], 'y_prob': []})
    with pytest.raises(ValueError, match='Orden de etiquetas'):
        build_audio_predictions(rows, {'y_true': [1], 'y_pred': [0], 'y_prob': [0.2]})


def test_dataset_really_has_three_real_recordings_per_test_person():
    import csv
    from collections import Counter
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'artifacts/old_vowels_dataset/test.csv'
    if not path.is_file():
        pytest.skip('Dataset files are generated locally')
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    counts = Counter(row['person_id'] for row in rows)
    assert len(rows) == 60 and len(counts) == 20 and set(counts.values()) == {3}
    assert all(row['origin'] == 'real' for row in rows)
