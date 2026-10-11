"""Export a small, inspectable snapshot for the explanatory frontend.

Run from the repository root: .venv/Scripts/python docs/tools/export_frontend_data.py
Only reads prepared artifacts and source files; never trains or modifies datasets.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts' / 'vowels_dataset'


def rows(name):
    with (DATA / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def excerpt(path, start, end):
    source = (ROOT / path).read_text(encoding='utf-8')
    lines = source.splitlines()
    return dict(path=path, line=start, code='\n'.join(lines[start - 1:end]),
                sha256=hashlib.sha256(source.encode()).hexdigest()[:12])


def main():
    manifest = rows('manifest.csv')
    people = rows('people.csv')
    assert len(manifest) == 1500 and len(people) == 100
    partitions = {}
    for split in ('train', 'val', 'test'):
        members = [row for row in manifest if row['split'] == split]
        partitions[split] = dict(people=len({r['person_id'] for r in members}),
                                 recordings=len(members))
    assert partitions == {'train': {'people':70,'recordings':1050},
                          'val': {'people':10,'recordings':150},
                          'test': {'people':20,'recordings':300}}
    person_sets = [{r['person_id'] for r in manifest if r['split'] == split}
                   for split in ('train', 'val', 'test')]
    assert all(person_sets[i].isdisjoint(person_sets[j]) for i in range(3) for j in range(i+1,3))
    example = next(r for r in manifest if r['split'] == 'train' and r['label'] == 'Control' and r['vowel'] == 'A')
    waveform = np.load(DATA / example['audio_npy_path'], mmap_mode='r')[0]
    mel = np.load(DATA / example['image_npy_path'], mmap_mode='r')
    assert waveform.shape == (286650,) and mel.shape == (64, 648)
    # Min/max preserves amplitude peaks in each display bin. Mel averages 4 time frames.
    wave_bins = [dict(low=round(float(part.min()), 5), high=round(float(part.max()), 5))
                 for part in np.array_split(waveform, 480)]
    mel_display = mel.reshape(64, 162, 4).mean(axis=2)
    config_source = (ROOT / 'neuroevolution/config.py').read_text(encoding='utf-8')
    tree = ast.parse(config_source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_default_config')
    defaults = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                try:
                    defaults[ast.literal_eval(key)] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    pass
    nb = json.loads((ROOT / 'vowels_dataset_por_persona.ipynb').read_text(encoding='utf-8'))
    test_nb = json.loads((ROOT / 'test.ipynb').read_text(encoding='utf-8'))
    def cell_excerpt(notebook, index, path):
        return dict(path=path, cell=index + 1, code=''.join(notebook['cells'][index]['source']))
    sources = {
        'extract': cell_excerpt(nb, 5, 'vowels_dataset_por_persona.ipynb'),
        'split': cell_excerpt(nb, 9, 'vowels_dataset_por_persona.ipynb'),
        'folds': cell_excerpt(nb, 11, 'vowels_dataset_por_persona.ipynb'),
        'duration': cell_excerpt(nb, 17, 'vowels_dataset_por_persona.ipynb'),
        'mel': cell_excerpt(nb, 22, 'vowels_dataset_por_persona.ipynb'),
        'audio': cell_excerpt(nb, 26, 'vowels_dataset_por_persona.ipynb'),
        'binding': excerpt('neuroevolution/config.py', 36, 86),
        'genome': excerpt('neuroevolution/genetics/genome.py', 36, 141),
        'model': excerpt('neuroevolution/models/evolvable_cnn.py', 1, 17),
        'conv1d': excerpt('neuroevolution/models/evolvable_cnn_1d.py', 274, 329),
        'conv2d': excerpt('neuroevolution/models/evolvable_cnn_2d.py', 110, 218),
        'blocks1d': excerpt('neuroevolution/models/evolvable_cnn_1d.py', 56, 197),
        'blocks2d': excerpt('neuroevolution/models/evolvable_cnn_2d.py', 40, 109),
        'initialization': excerpt('neuroevolution/evolution/engine.py', 479, 556),
        'fitness': excerpt('neuroevolution/evolution/fitness.py', 411, 542),
        'crossover': excerpt('neuroevolution/genetics/crossover.py', 29, 87),
        'mutation': excerpt('neuroevolution/genetics/mutation.py', 1, 125),
        'selection': excerpt('neuroevolution/evolution/engine.py', 976, 1044),
        'final': cell_excerpt(test_nb, 12, 'test.ipynb'),
    }
    # Locate cells by content so source evidence survives notebook cell insertion.
    for key, needle, notebook, path in [
        ('audio', 'AUDIO_ARRAY_ROOT =', nb, 'vowels_dataset_por_persona.ipynb'),
        ('final', 'def global_loader', test_nb, 'test.ipynb'),
        ('monitor', 'class MonitoredVowelsEvolution', test_nb, 'test.ipynb'),
    ]:
        index = next(i for i,c in enumerate(notebook['cells']) if needle in ''.join(c['source']))
        sources[key] = cell_excerpt(notebook, index, path)
    snapshot = dict(
        exported_at=datetime.now(timezone.utc).isoformat(), people=len(people), recordings=len(manifest),
        partitions=partitions, defaults=defaults,
        audio=json.loads((DATA / 'audio_config.json').read_text()),
        mel=json.loads((DATA / 'feature_config.json').read_text()),
        example=dict(label=example['label'], vowel=example['vowel'],
                     path=example['processed_path'], audio_path=example['audio_npy_path'],
                     mel_path=example['image_npy_path'], waveform=wave_bins,
                     spectrogram=np.round(mel_display, 5).tolist()),
        sources=sources,
    )
    destination = ROOT / 'docs' / 'frontend-data.js'
    destination.write_text('// Generated from local project files. Refresh with docs/tools/export_frontend_data.py\n'
                           + 'window.PROJECT_DATA = ' + json.dumps(snapshot, ensure_ascii=False, separators=(',', ':'))
                           + ';\n', encoding='utf-8')
    print(f'Exported {len(manifest)} recordings, {len(people)} people, real previews and {len(sources)} source excerpts.')


if __name__ == '__main__':
    main()
