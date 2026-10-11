"""Subject grouping and paired representations for real/BigVSAN data."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_old_vowels_dataset import (
    assign_reference_splits, exclude_generated_test, log_mel, mel_filters,
    partition_rows, remove_previous_generated_test, validate_groups,
)


def reference_people():
    records = []
    for label, binary in (("Control", 0), ("Patologicas", 1)):
        for number in range(50):
            split = "train" if number < 35 else "val" if number < 40 else "test"
            records.append({"person_id": f"{label}_{number:04d}", "label": label,
                            "label_binary": binary, "split": split,
                            "cv_fold": number // 7 if split == "train" else pd.NA})
    return pd.DataFrame(records).astype({"cv_fold": "Int64"})


def source_rows(people):
    records = []
    for person in people.itertuples(index=False):
        for repetition in range(1, 4):
            for iteration in (0, 1, 2):
                records.append({"person_id": person.person_id, "label": person.label,
                                "label_binary": person.label_binary,
                                "source_audio_id": f"{person.person_id}_A{repetition}",
                                "source_path": f"{person.person_id}_A{repetition}_{iteration}.wav",
                                "origin": "real" if iteration == 0 else "generated"})
    return pd.DataFrame(records)


def test_real_and_all_generated_variants_reuse_reference_person_assignments(tmp_path):
    reference = reference_people()
    path = tmp_path / "people.csv"
    reference.to_csv(path, index=False)
    # Same numeric ID in the two classes refers to different people.
    manifest, people = assign_reference_splits(source_rows(reference), path)
    assert people.set_index("person_id").split.equals(reference.set_index("person_id").split)
    assert manifest.groupby("person_id").split.nunique().eq(1).all()
    assert manifest.groupby("person_id").cv_fold.nunique(dropna=False).eq(1).all()
    assert manifest.groupby("source_audio_id").split.nunique().eq(1).all()
    for csv_name, _, _, rows in partition_rows(manifest):
        if csv_name.startswith("fold_"):
            assert set(rows.split) == {"train"}
        assert rows.groupby(["label", "origin"]).person_id.nunique().nunique() == 1


@pytest.mark.parametrize("column", ["split", "cv_fold"])
def test_generated_variant_cannot_leak_into_another_partition_or_fold(tmp_path, column):
    reference = reference_people()
    path = tmp_path / "people.csv"
    reference.to_csv(path, index=False)
    manifest, _ = assign_reference_splits(source_rows(reference), path)
    index = manifest.index[manifest.origin.eq("generated")][0]
    manifest.loc[index, column] = "test" if column == "split" else 4
    with pytest.raises(ValueError, match="Person shared|different CV folds"):
        validate_groups(manifest)


def test_reference_label_mismatch_is_rejected(tmp_path):
    reference = reference_people()
    path = tmp_path / "people.csv"
    reference.to_csv(path, index=False)
    source = source_rows(reference)
    source.loc[0, "label_binary"] = 1
    with pytest.raises(ValueError, match="labels differ"):
        assign_reference_splits(source, path)


def test_test_generated_variants_are_discarded_without_changing_train_val_or_folds(tmp_path):
    reference = reference_people()
    path = tmp_path / "people.csv"
    reference.to_csv(path, index=False)
    original, people = assign_reference_splits(source_rows(reference), path)
    retained, updated_people, excluded = exclude_generated_test(original, people)
    pd.testing.assert_frame_equal(
        retained.loc[retained.split.ne("test")].reset_index(drop=True),
        original.loc[original.split.ne("test")].reset_index(drop=True),
    )
    assert len(excluded) == 120 and set(excluded.split) == {"test"}
    assert set(excluded.origin) == {"generated"}
    assert len(retained.loc[retained.split.eq("test")]) == 60
    assert set(retained.loc[retained.split.eq("test"), "origin"]) == {"real"}
    assert set(excluded.source_path).isdisjoint(retained.source_path)
    assert updated_people.loc[updated_people.split.eq("test"), "generated_audios"].eq(0).all()
    pd.testing.assert_frame_equal(updated_people[reference.columns], reference)


def test_cleanup_removes_only_previous_generated_test_exports(tmp_path):
    (tmp_path / "dataset_config.json").write_text(json.dumps({"feature_id": "old_vowels"}))
    kept = []
    for prefix in ("", "npy", "npy_audio"):
        for label in ("Control", "Patologicas"):
            generated = tmp_path / prefix / "test" / label / "generated"
            generated.mkdir(parents=True)
            (generated / "old-export").write_text("generated")
            for split, origin in (("test", "real"), ("train", "generated"), ("val", "generated")):
                keep = tmp_path / prefix / split / label / origin / "keep"
                keep.parent.mkdir(parents=True, exist_ok=True)
                keep.write_text("keep")
                kept.append(keep)
    remove_previous_generated_test(tmp_path)
    assert all(path.is_file() for path in kept)
    assert not list(tmp_path.glob("**/test/*/generated"))


def test_cleanup_refuses_an_unrecognized_dataset(tmp_path):
    target = tmp_path / "test" / "Control" / "generated"
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="unrecognized"):
        remove_previous_generated_test(tmp_path)
    assert target.is_dir()


def test_log_mel_shape_silence_and_frequency_localization():
    filters = mel_filters()
    silent = log_mel(np.zeros(11520, dtype=np.float32), filters)
    assert silent.shape == (64, 44) and silent.dtype == np.float32
    assert np.count_nonzero(silent) == 0
    time = np.arange(11520, dtype=np.float32) / 24000
    signal = np.sin(2 * np.pi * 1000 * time).astype(np.float32)
    features = log_mel(signal, filters)
    edges = 700 * (10 ** (np.linspace(2595 * np.log10(1 + 80 / 700),
                                    2595 * np.log10(1 + 8000 / 700), 66) / 2595) - 1)
    peak = int(features.mean(axis=1).argmax())
    assert edges[peak] < 1000 < edges[peak + 2]
    assert np.isfinite(features).all()
