"""Export real/BigVSAN audio using the exact subject splits of vowels_dataset.

Run from the repository root with:
    python scripts/prepare_old_vowels_dataset.py
The companion notebook uses these same functions; neither source is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import wave

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FOLDERS = {
    "control_files_short_24khz": ("Control", "real"),
    "pathological_files_short_24khz": ("Patologicas", "real"),
    "pretrained_40_1e5_BigVSAN_generated_control": ("Control", "generated"),
    "pretrained_40_1e5_BigVSAN_generated_pathological": ("Patologicas", "generated"),
}
NAME_PATTERN = re.compile(
    r"^AVPEPUDEA(?P<control>C?)(?P<number>\d{4})(?P<vowel>[aeiou])"
    r"(?P<repetition>[123])_shortened(?:_iter(?P<iteration>\d+)_generated)?\.wav$",
    re.IGNORECASE,
)
FEATURE_ID = "old_vowels"
N_FOLDS = 5
SAMPLE_RATE = 24_000
N_SAMPLES = 11_520
N_MELS, N_FFT, HOP_LENGTH = 64, 1024, 240
MIN_HZ, MAX_HZ, POWER_SCALE = 80.0, 8000.0, 1000.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inventory(source_root: Path) -> tuple[pd.DataFrame, dict]:
    """Read only the four named folders, ignoring AppleDouble metadata."""
    records, ignored = [], []
    for folder, (label, origin) in SOURCE_FOLDERS.items():
        directory = source_root / folder
        require(directory.is_dir(), f"Missing source folder: {directory}")
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_root)
            if "__MACOSX" in relative.parts or path.name.startswith("._"):
                ignored.append(relative.as_posix())
                continue
            require(path.suffix.lower() == ".wav", f"Unexpected source file: {relative}")
            match = NAME_PATTERN.fullmatch(path.name)
            require(match is not None, f"Unexpected audio name: {relative}")
            name_label = "Control" if match["control"] else "Patologicas"
            iteration = int(match["iteration"]) if match["iteration"] else 0
            require(name_label == label, f"Folder/filename label mismatch: {relative}")
            require((iteration > 0) == (origin == "generated"), f"Origin mismatch: {relative}")
            require(iteration == 0 or 1 <= iteration <= 40, f"Unexpected iteration: {relative}")
            with wave.open(str(path), "rb") as audio:
                require(
                    (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(),
                     audio.getnframes(), audio.getcomptype())
                    == (SAMPLE_RATE, 1, 2, N_SAMPLES, "NONE"),
                    f"Expected 0.48 s, 24 kHz, mono PCM16: {relative}",
                )
            base_stem = path.stem.split("_iter")[0]
            records.append({
                "person_id": f"{label}_{match['number']}",
                "label": label, "label_binary": int(label == "Patologicas"),
                "origin": origin, "vowel": match["vowel"].upper(),
                "repetition": int(match["repetition"]), "generation_iteration": iteration,
                "source_audio_id": base_stem, "source_path": relative.as_posix(),
                "sample_rate_hz": SAMPLE_RATE, "raw_samples": N_SAMPLES,
                "raw_duration_s": N_SAMPLES / SAMPLE_RATE,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    require(bool(records), "No source audio found")
    manifest = pd.DataFrame(records).sort_values(
        ["label", "person_id", "vowel", "repetition", "generation_iteration"]
    ).reset_index(drop=True)
    require(not manifest.duplicated(["source_audio_id", "generation_iteration"]).any(),
            "Duplicate recording/iteration in source")
    real = manifest.loc[manifest.origin.eq("real")]
    require(real.person_id.nunique() == 100 and len(real) == 300,
            "Expected 100 people and 300 real recordings")
    for person, rows in real.groupby("person_id"):
        require(set(zip(rows.vowel, rows.repetition)) == {("A", 1), ("A", 2), ("A", 3)},
                f"Incomplete real recordings: {person}")
    generated = manifest.loc[manifest.origin.eq("generated")]
    require(set(generated.source_audio_id) <= set(real.source_audio_id),
            "Generated recording has no real source recording")
    present = set(zip(generated.source_audio_id, generated.generation_iteration))
    missing = [
        {"source_audio_id": audio_id, "generation_iteration": iteration}
        for audio_id in sorted(real.source_audio_id) for iteration in range(1, 41)
        if (audio_id, iteration) not in present
    ]
    return manifest, {"ignored_metadata_files": ignored, "missing_generated_variants": missing}


def assign_reference_splits(manifest: pd.DataFrame, reference_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One lookup per person, shared by real audio and all its generated variants."""
    if not reference_csv.is_file():
        raise FileNotFoundError(f"Run vowels_dataset_por_persona.ipynb first: {reference_csv}")
    people = pd.read_csv(reference_csv, dtype={"person_id": str, "cv_fold": "Int64"})
    require(set(("person_id", "label", "label_binary", "split", "cv_fold")) <= set(people),
            "Reference CSV lacks person/split/fold columns")
    people = people[["person_id", "label", "label_binary", "split", "cv_fold"]].copy()
    require(not people.person_id.duplicated().any(), "Duplicate person in reference")
    require(set(people.person_id) == set(manifest.person_id),
            "Source people differ from the vowels_dataset reference")
    require(set(people.split) == {"train", "val", "test"}, "Unexpected reference splits")
    expected_counts = {"train": 35, "val": 5, "test": 10}
    for split, count in expected_counts.items():
        rows = people.loc[people.split.eq(split)]
        require(rows.label.value_counts().to_dict() == {"Control": count, "Patologicas": count},
                f"Reference must have {count} people per class in {split}")
    require(people.loc[people.split.ne("train"), "cv_fold"].isna().all(),
            "Reserved reference people must not belong to a CV fold")
    train = people.loc[people.split.eq("train")]
    require(train.cv_fold.notna().all() and set(train.cv_fold) == set(range(N_FOLDS)),
            "Invalid reference CV folds")
    for fold in range(N_FOLDS):
        require(train.loc[train.cv_fold.eq(fold), "label"].value_counts().to_dict()
                == {"Control": 7, "Patologicas": 7}, f"Unbalanced reference fold {fold}")
    joined = manifest.merge(people, on=["person_id", "label", "label_binary"],
                            how="left", validate="many_to_one", sort=False)
    require(joined.split.notna().all(), "Reference labels differ from source labels")
    joined["cv_fold"] = joined.cv_fold.astype("Int64")
    counts = joined.groupby(["person_id", "origin"]).size().unstack(fill_value=0)
    people = people.merge(counts.rename(columns={"real": "real_audios", "generated": "generated_audios"}),
                          on="person_id", validate="one_to_one")
    validate_groups(joined)
    return joined, people


def partition_rows(manifest: pd.DataFrame):
    for split in ("train", "val", "test"):
        yield f"{split}.csv", split, "", manifest.loc[manifest.split.eq(split)]
    for fold in range(N_FOLDS):
        inner_val = manifest.cv_fold.eq(fold).fillna(False)
        for split, mask in (("train", ~inner_val), ("val", inner_val)):
            yield (f"fold_{fold}_{split}.csv", split, f"_fold_{fold + 1}",
                   manifest.loc[manifest.split.eq("train") & mask])


def exclude_generated_test(manifest: pd.DataFrame, people: pd.DataFrame):
    """Discard test subjects' generated variants without reassigning any person."""
    remove = manifest.split.eq("test") & manifest.origin.eq("generated")
    excluded = manifest.loc[remove].copy().reset_index(drop=True)
    retained = manifest.loc[~remove].copy().reset_index(drop=True)
    counts = retained.groupby(["person_id", "origin"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["real", "generated"], fill_value=0).rename(
        columns={"real": "real_audios", "generated": "generated_audios"})
    updated_people = people.drop(columns=["real_audios", "generated_audios"]).merge(
        counts, on="person_id", validate="one_to_one")
    validate_groups(retained)
    return retained, updated_people, excluded


def remove_previous_generated_test(output_root: Path) -> None:
    """Remove only the six generated-test export directories from a previous run."""
    marker = output_root / "dataset_config.json"
    require(marker.is_file() and json.loads(marker.read_text())["feature_id"] == FEATURE_ID,
            "Refusing to clean an unrecognized output dataset")
    root = output_root.resolve()
    targets = [output_root / prefix / "test" / label / "generated"
               for prefix in (Path(), Path("npy_audio"), Path("npy"))
               for label in ("Control", "Patologicas")]
    # Resolve and check every absolute target before performing any recursive deletion.
    for target in targets:
        expected = root / target.relative_to(output_root)
        require(target.resolve() == expected and target.resolve().is_relative_to(root),
                f"Unsafe generated-test output path: {target}")
    for target in targets:
        if target.exists():
            require(target.is_dir(), f"Expected an output directory: {target}")
            shutil.rmtree(target)


def validate_groups(manifest: pd.DataFrame) -> None:
    require(manifest.groupby("person_id").split.nunique().eq(1).all(),
            "Person shared between global partitions")
    require(manifest.groupby("person_id").cv_fold.nunique(dropna=False).eq(1).all(),
            "Real/generated recordings of a person have different CV folds")
    require(manifest.groupby("source_audio_id").split.nunique().eq(1).all(),
            "Generated variant separated from real source")
    require(manifest.loc[manifest.split.ne("train"), "cv_fold"].isna().all(),
            "Reserved people found in CV folds")
    train = manifest.loc[manifest.split.eq("train")]
    require(train.cv_fold.notna().all(), "Training person without a CV fold")
    reserved = set(manifest.loc[manifest.split.ne("train"), "person_id"])
    for fold in range(N_FOLDS):
        inner_val = set(train.loc[train.cv_fold.eq(fold), "person_id"])
        inner_train = set(train.loc[~train.cv_fold.eq(fold), "person_id"])
        require(inner_train.isdisjoint(inner_val), f"Person leakage in fold {fold}")
        require((inner_train | inner_val).isdisjoint(reserved), f"Held-out leakage in fold {fold}")


def mel_filters() -> np.ndarray:
    frequencies = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    edges = 700.0 * (10.0 ** (np.linspace(to_mel(MIN_HZ), to_mel(MAX_HZ), N_MELS + 2) / 2595.0) - 1.0)
    filters = np.zeros((N_MELS, len(frequencies)), dtype=np.float32)
    for band in range(N_MELS):
        left, center, right = edges[band:band + 3]
        filters[band] = np.maximum(0.0, np.minimum(
            (frequencies - left) / (center - left), (right - frequencies) / (right - center)))
        filters[band] /= max(float(filters[band].sum()), 1e-12)
    return filters


def log_mel(waveform: np.ndarray, filters: np.ndarray) -> np.ndarray:
    from scipy.signal import stft

    _, _, spectrum = stft(waveform, fs=SAMPLE_RATE, nperseg=N_FFT,
                          noverlap=N_FFT - HOP_LENGTH, boundary=None, padded=False)
    power = np.abs(spectrum).astype(np.float32) ** 2
    return np.ascontiguousarray(np.log1p(POWER_SCALE * (filters @ power)), dtype=np.float32)


def export_recordings(manifest: pd.DataFrame, source_root: Path, output_root: Path) -> pd.DataFrame:
    filters, paths = mel_filters(), []
    for index, row in enumerate(manifest.itertuples(index=False)):
        source = source_root / row.source_path
        require(hashlib.sha256(source.read_bytes()).hexdigest() == row.source_sha256,
                f"Source changed after inventory: {source}")
        relative = Path(row.split) / row.label / row.origin / row.vowel / source.name
        processed = output_root / relative
        processed.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, processed)
        with wave.open(str(processed), "rb") as audio:
            pcm = np.frombuffer(audio.readframes(N_SAMPLES), dtype="<i2")
        require(pcm.shape == (N_SAMPLES,), f"Truncated audio: {source}")
        waveform = np.ascontiguousarray(pcm.astype(np.float32)[None, :] / 32768.0)
        features = log_mel(waveform[0], filters)
        require(features.shape == (N_MELS, 44) and np.isfinite(features).all(),
                f"Invalid features: {source}")
        audio_relative = Path("npy_audio") / relative.with_suffix(".npy")
        image_relative = Path("npy") / relative.with_suffix(".npy")
        for path, array in ((audio_relative, waveform), (image_relative, features)):
            (output_root / path).parent.mkdir(parents=True, exist_ok=True)
            np.save(output_root / path, array)
        paths.append((relative.as_posix(), audio_relative.as_posix(), image_relative.as_posix()))
        if (index + 1) % 1000 == 0:
            print(f"Recordings: {index + 1}/{len(manifest)}", flush=True)
    manifest = manifest.copy()
    manifest[["processed_path", "audio_npy_path", "image_npy_path"]] = paths
    manifest["feature_npy_path"] = manifest.image_npy_path
    manifest["processed_samples"] = N_SAMPLES
    manifest["processed_duration_s"] = N_SAMPLES / SAMPLE_RATE
    manifest["audio_channels"], manifest["audio_samples"] = 1, N_SAMPLES
    manifest["feature_channels"], manifest["feature_frames"] = N_MELS, 44
    return manifest


def export_arrays(manifest: pd.DataFrame, people: pd.DataFrame, output_root: Path) -> None:
    require(manifest.loc[manifest.split.eq("test"), "origin"].eq("real").all(),
            "Test must contain real audio only")
    manifest.to_csv(output_root / "manifest.csv", index=False)
    people.to_csv(output_root / "people.csv", index=False)
    for csv_name, split, suffix, rows in partition_rows(manifest):
        rows.to_csv(output_root / csv_name, index=False)
        for directory, shape, column in (
            ("arrays_audio", (1, N_SAMPLES), "audio_npy_path"),
            ("arrays", (N_MELS, 44), "image_npy_path"),
        ):
            root = output_root / directory
            root.mkdir(exist_ok=True)
            x = np.lib.format.open_memmap(root / f"X_{split}_{FEATURE_ID}{suffix}.npy",
                                         mode="w+", dtype=np.float32, shape=(len(rows), *shape))
            for i, path in enumerate(rows[column]):
                x[i] = np.load(output_root / path, allow_pickle=False)
            x.flush()
            del x
            np.save(root / f"y_{split}_{FEATURE_ID}{suffix}.npy", rows.label_binary.to_numpy(dtype=np.int64))
        print(f"Exported {csv_name}: {len(rows)} recordings", flush=True)


def distribution(manifest: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for csv_name, _, _, rows in partition_rows(manifest):
        summary = rows.groupby(["label", "origin"]).agg(
            people=("person_id", "nunique"), audios=("source_path", "size")).reset_index()
        summary.insert(0, "partition", Path(csv_name).stem)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def verify_output(output_root: Path) -> dict:
    """Check every row of every array against its CSV and individual representation."""
    manifest = pd.read_csv(output_root / "manifest.csv", dtype={"cv_fold": "Int64"})
    validate_groups(manifest)
    test = manifest.loc[manifest.split.eq("test")]
    require(not test.empty and test.origin.eq("real").all(), "Test must contain real audio only")
    for row in manifest.itertuples(index=False):
        path = output_root / row.processed_path
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row.source_sha256,
                f"Processed WAV differs from source: {path}")
        with wave.open(str(path), "rb") as audio:
            require((audio.getframerate(), audio.getnframes(), audio.getnchannels(), audio.getsampwidth())
                    == (SAMPLE_RATE, N_SAMPLES, 1, 2), f"Unexpected processed WAV: {path}")
            pcm = np.frombuffer(audio.readframes(N_SAMPLES), dtype="<i2")
        require(np.array_equal(np.load(output_root / row.audio_npy_path)[0], pcm.astype(np.float32) / 32768.0),
                f"Waveform differs from PCM: {path}")
    for csv_name, split, suffix, rows in partition_rows(manifest):
        saved = pd.read_csv(output_root / csv_name)
        require(saved.source_path.tolist() == rows.source_path.tolist(), f"CSV order differs: {csv_name}")
        for directory, shape, column in (
            ("arrays_audio", (1, N_SAMPLES), "audio_npy_path"),
            ("arrays", (N_MELS, 44), "image_npy_path"),
        ):
            root = output_root / directory
            require(not list(root.glob("*test*fold*.npy")), "Test must be global, not per fold")
            x = np.load(root / f"X_{split}_{FEATURE_ID}{suffix}.npy", mmap_mode="r")
            y = np.load(root / f"y_{split}_{FEATURE_ID}{suffix}.npy", mmap_mode="r")
            require(x.shape == (len(rows), *shape) and x.dtype == np.float32, f"Invalid array: {csv_name}")
            require(y.dtype == np.int64 and np.array_equal(y, rows.label_binary.to_numpy()),
                    f"Labels differ: {csv_name}")
            for i, path in enumerate(rows[column]):
                require(np.isfinite(x[i]).all() and np.array_equal(x[i], np.load(output_root / path)),
                        f"Array differs from individual recording: {csv_name}, row {i}")
        print(f"Verified {csv_name}: {len(rows)} recordings", flush=True)
    for directory, suffix in (("npy_audio", "*.npy"), ("npy", "*.npy")):
        require(len(list((output_root / directory).rglob(suffix))) == len(manifest), "Stale/missing individual arrays")
    require(len(list(output_root.glob("*/*/*/*/*.wav"))) == len(manifest), "Stale/missing WAVs")
    result = {"verified": True, "people": int(manifest.person_id.nunique()),
              "audios": len(manifest), "real_audios": int(manifest.origin.eq("real").sum()),
              "generated_audios": int(manifest.origin.eq("generated").sum()),
              "test_real_audios": len(test), "test_generated_audios": 0,
              "excluded_test_generated_audios": len(pd.read_csv(output_root / "excluded_test_generated.csv")),
              "person_leakage": False, "num_folds": N_FOLDS,
              "audio_shape": [1, N_SAMPLES], "spectrogram_shape": [N_MELS, 44],
              "array_csv_alignment": "all rows in both modalities and all partitions checked"}
    (output_root / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def prepare(source_root: Path = PROJECT_ROOT / "old_data_do_not_use",
            output_root: Path = PROJECT_ROOT / "artifacts" / "old_vowels_dataset",
            reference_csv: Path = PROJECT_ROOT / "artifacts" / "vowels_dataset" / "people.csv") -> dict:
    source_root, output_root, reference_csv = map(Path, (source_root, output_root, reference_csv))
    require(not output_root.resolve().is_relative_to(source_root.resolve()), "Output must be outside source")
    require(output_root.resolve() != reference_csv.parent.resolve(), "Output must not overwrite reference dataset")
    manifest, audit = inventory(source_root)
    manifest, people = assign_reference_splits(manifest, reference_csv)
    manifest, people, excluded = exclude_generated_test(manifest, people)
    audit["excluded_test_generated_count"] = len(excluded)
    audit["excluded_test_generated_by_label"] = excluded.label.value_counts().to_dict()
    audit["excluded_test_generated_reason"] = "Removed (quitados): final test contains real audio only"
    audit["excluded_test_generated_csv"] = "excluded_test_generated.csv"
    print(distribution(manifest).head(12).to_string(index=False), flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / "dataset_config.json"
    if any(output_root.iterdir()):
        require(marker.is_file() and json.loads(marker.read_text())["feature_id"] == FEATURE_ID,
                f"Output directory is not an existing {FEATURE_ID} dataset: {output_root}")
    config = {
        "feature_id": FEATURE_ID, "source_root": str(source_root.resolve()),
        "reference_people_csv": str(reference_csv.resolve()),
        "reference_sha256": hashlib.sha256(reference_csv.read_bytes()).hexdigest(),
        "seed": 42, "train_fraction": 0.7, "validation_fraction": 0.1, "test_fraction": 0.2,
        "num_folds": N_FOLDS, "assignment": "exact person/split/cv_fold copied from reference",
        "sample_rate_hz": SAMPLE_RATE, "duration_s": N_SAMPLES / SAMPLE_RATE,
        "source_folders": SOURCE_FOLDERS,
        "generated_policy": "train/val variants follow their real source person; test variants removed, not reassigned",
        "test_policy": "real audio only", "excluded_test_generated_audios": len(excluded),
        "generator_training_subjects": "not documented in the source folders",
    }
    marker.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    remove_previous_generated_test(output_root)
    excluded.to_csv(output_root / "excluded_test_generated.csv", index=False)
    (output_root / "source_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest = export_recordings(manifest, source_root, output_root)
    export_arrays(manifest, people, output_root)
    distribution(manifest).to_csv(output_root / "distribution.csv", index=False)
    feature_config = {"representation": "log_mel_power", "source_sample_rate_hz": SAMPLE_RATE,
                      "source_samples": N_SAMPLES, "n_mels": N_MELS, "n_fft": N_FFT,
                      "hop_length": HOP_LENGTH, "min_hz": MIN_HZ, "max_hz": MAX_HZ,
                      "shape": [N_MELS, 44], "dtype": "float32", "scale": "log1p(1000 * mel_power)",
                      "stft": {"window": "hann", "boundary": None, "padded": False}}
    audio_config = {"representation": "pcm16_waveform_scaled_float32", "source_sample_rate_hz": SAMPLE_RATE,
                    "source_samples": N_SAMPLES, "shape": [1, N_SAMPLES], "dtype": "float32", "scale": "pcm16 / 32768"}
    for name, value in (("feature_config.json", feature_config), ("audio_config.json", audio_config)):
        (output_root / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    result = verify_output(output_root)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "old_data_do_not_use")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts" / "old_vowels_dataset")
    parser.add_argument("--reference-people", type=Path, default=PROJECT_ROOT / "artifacts" / "vowels_dataset" / "people.csv")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_output(args.output_root), indent=2))
    else:
        prepare(args.source_root, args.output_root, args.reference_people)
