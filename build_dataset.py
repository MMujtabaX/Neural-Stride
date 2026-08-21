"""
================================================================================
NeuralStride AI - Dataset Builder (Leg Raise, multi-label)
================================================================================

Converts recorded JSONL sessions (dataset/legraise/*.jsonl + dataset/manifest.csv)
into model-ready tensors for a multi-label CNN-LSTM. Does NOT train anything —
its only job is: read manifest -> slice per-rep windows -> build multi-label
targets -> subject-grouped split -> augment train only -> save arrays -> print
a verification summary.

Runs in venv-train (numpy only, no TensorFlow/mediapipe needed at build time):
    py -3.12 -m venv venv-train
    venv-train\\Scripts\\pip install numpy
    venv-train\\Scripts\\python build_dataset.py

WINDOW_LEN and EXERCISE are config, not hardcoded assumptions, so squat/scaption
can reuse this file later with different values.
================================================================================
"""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from feature_utils import FEATURE_COUNT, MIRROR_PAIRS, mirror_features

# ============================================================
# CONFIGURATION
# ============================================================
EXERCISE = "Side Leg Raise"
WINDOW_LEN = 32            # leg raise reps are 13-23 frames; pad/truncate to 32
NUM_FEATURES = 22
LABEL_CODES = ["KNEE_BENT", "SHALLOW"]   # multi-label order; target = [knee_bent, shallow]
QUALITY_FLOOR = 0.5        # drop individual frames below this feature_quality before windowing
MIN_VALID_FRAMES = 8       # if a rep has fewer good frames than this, drop the rep (too degraded)
SEED = 42

# Angle-family features (0-13: joint angles, back lean, symmetry diffs) get
# degree-scale noise; geometry features (14-21, divided by hip_width) get a
# much smaller sigma to match their normalized scale.
NOISE_SIGMA_ANGLES = 1.5
NOISE_SIGMA_GEOMETRY = 0.02
TIME_WARP_SCALES = (0.9, 1.1)

FAULT_TYPE_TO_LABEL = {"knee_bent": "KNEE_BENT", "shallow": "SHALLOW"}

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "dataset" / "manifest.csv"
DATA_DIR = PROJECT_ROOT / "dataset" / "legraise"
OUT_DIR = PROJECT_ROOT / "dataset" / "built" / "legraise"

assert NUM_FEATURES == FEATURE_COUNT, "NUM_FEATURES config must match feature_utils.FEATURE_COUNT"

NOISE_SIGMA = np.array(
    [NOISE_SIGMA_ANGLES] * 14 + [NOISE_SIGMA_GEOMETRY] * (NUM_FEATURES - 14)
)


@dataclass
class RepSample:
    feat: np.ndarray   # (T, NUM_FEATURES) unpadded, quality-filtered
    conf: np.ndarray   # (T, NUM_FEATURES) unpadded
    label: np.ndarray  # (len(LABEL_CODES),)
    subject: str
    filename: str
    split: str


# ============================================================
# Step 1: manifest
# ============================================================
def load_manifest():
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}")
        sys.exit(1)
    with open(MANIFEST_PATH, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["exercise"] == EXERCISE]

    missing = [row["filename"] for row in rows if not (DATA_DIR / row["filename"]).exists()]
    if missing:
        print("ERROR: manifest references files missing from dataset/legraise/:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    return rows


# ============================================================
# Step 2: load reps from a JSONL session
# ============================================================
def load_jsonl(path):
    frames = {}
    reps = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rtype = rec.get("type")
            if rtype == "frame":
                frames[rec["frame"]] = rec
            elif rtype == "rep" and rec.get("completed"):
                reps.append(rec)
    return frames, reps


def extract_rep_frames(frames, rep):
    """Frames of a completed rep, in order, with feature_quality >= QUALITY_FLOOR."""
    valid = []
    for fn in range(rep["start_frame"], rep["end_frame"] + 1):
        rec = frames.get(fn)
        if rec is None or rec.get("feature_quality", 0.0) < QUALITY_FLOOR:
            continue
        valid.append(rec)
    return valid


def build_label(rep):
    issues = set(rep.get("issues", []))
    return np.array([1 if code in issues else 0 for code in LABEL_CODES], dtype=np.int64)


# ============================================================
# Step 3: windowing (pad/truncate to WINDOW_LEN)
# ============================================================
def pad_or_truncate(feat, conf, window_len):
    """Centered truncate if longer than window_len (peak of the rep is
    mid-sequence); end-pad with zeros if shorter. Returns (feat, conf, mask)."""
    t = feat.shape[0]
    if t == window_len:
        return feat, conf, np.ones(window_len, dtype=np.float32)
    if t > window_len:
        excess = t - window_len
        start = excess // 2
        end = start + window_len
        return feat[start:end], conf[start:end], np.ones(window_len, dtype=np.float32)
    pad_len = window_len - t
    feat_p = np.vstack([feat, np.zeros((pad_len, feat.shape[1]))])
    conf_p = np.vstack([conf, np.zeros((pad_len, conf.shape[1]))])
    mask = np.concatenate([np.ones(t, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
    return feat_p, conf_p, mask


# ============================================================
# Step 4: process manifest rows -> RepSamples grouped by split
# ============================================================
def process_manifest(manifest_rows):
    reps_by_split = {"train": [], "test": []}
    stats = {
        "files": [],           # (filename, subject, split, completed_reps)
        "total_completed": 0,
        "dropped_degraded": 0,
        "fault_mismatches": [],
    }

    for row in manifest_rows:
        frames, reps = load_jsonl(DATA_DIR / row["filename"])
        stats["files"].append((row["filename"], row["subject"], row["split"], len(reps)))
        stats["total_completed"] += len(reps)

        file_label_hits = {code: 0 for code in LABEL_CODES}
        for rep in reps:
            valid_frames = extract_rep_frames(frames, rep)
            if len(valid_frames) < MIN_VALID_FRAMES:
                stats["dropped_degraded"] += 1
                continue

            feat = np.array([r["features"] for r in valid_frames], dtype=np.float64)
            conf = np.array([r["feature_confidence"] for r in valid_frames], dtype=np.float64)
            label = build_label(rep)
            for i, code in enumerate(LABEL_CODES):
                if label[i]:
                    file_label_hits[code] += 1

            reps_by_split[row["split"]].append(
                RepSample(feat=feat, conf=conf, label=label,
                          subject=row["subject"], filename=row["filename"], split=row["split"])
            )

        expected_code = FAULT_TYPE_TO_LABEL.get(row["fault_type"])
        if expected_code and file_label_hits[expected_code] == 0:
            stats["fault_mismatches"].append(
                f"{row['filename']}: manifest fault_type={row['fault_type']} "
                f"but zero reps carry {expected_code}"
            )

    return reps_by_split, stats


def assert_no_subject_leak(reps_by_split):
    train_subjects = {r.subject for r in reps_by_split["train"]}
    test_subjects = {r.subject for r in reps_by_split["test"]}
    overlap = train_subjects & test_subjects
    if overlap:
        print(f"ERROR: subject(s) appear in both train and test splits: {sorted(overlap)}")
        sys.exit(1)
    return train_subjects, test_subjects


# ============================================================
# Step 5: augmentation (train only)
# ============================================================
def mirror_confidence(conf_vec):
    """Same index swaps as mirror_features, no negation — confidence has no sign."""
    mirrored = list(conf_vec)
    for i, j in MIRROR_PAIRS:
        mirrored[i], mirrored[j] = conf_vec[j], conf_vec[i]
    return mirrored


def augment_mirror(X, conf, mask, label):
    X_m = np.array([mirror_features(frame) for frame in X])
    conf_m = np.array([mirror_confidence(frame) for frame in conf])
    return X_m, conf_m, mask.copy(), label.copy()


def augment_time_warp(feat, conf, label, rng):
    t = feat.shape[0]
    scale = rng.choice(TIME_WARP_SCALES)
    new_t = max(2, int(round(t * scale)))
    old_idx = np.linspace(0, t - 1, t)
    new_idx = np.linspace(0, t - 1, new_t)
    feat_w = np.empty((new_t, feat.shape[1]), dtype=np.float64)
    conf_w = np.empty((new_t, conf.shape[1]), dtype=np.float64)
    for j in range(feat.shape[1]):
        feat_w[:, j] = np.interp(new_idx, old_idx, feat[:, j])
        conf_w[:, j] = np.interp(new_idx, old_idx, conf[:, j])
    X_w, conf_p, mask_w = pad_or_truncate(feat_w, conf_w, WINDOW_LEN)
    return X_w, conf_p, mask_w, label.copy()


def augment_noise(feat, conf, label, rng):
    noise = rng.normal(0.0, NOISE_SIGMA, size=feat.shape)
    feat_n = feat + noise
    X_n, conf_n, mask_n = pad_or_truncate(feat_n, conf, WINDOW_LEN)
    return X_n, conf_n, mask_n, label.copy()


def build_split_arrays(samples, augment, rng):
    X_list, conf_list, mask_list, y_list = [], [], [], []
    for s in samples:
        X, conf, mask = pad_or_truncate(s.feat, s.conf, WINDOW_LEN)
        X_list.append(X); conf_list.append(conf); mask_list.append(mask); y_list.append(s.label)

        if augment:
            for variant in (
                augment_mirror(X, conf, mask, s.label),
                augment_time_warp(s.feat, s.conf, s.label, rng),
                augment_noise(s.feat, s.conf, s.label, rng),
            ):
                Xv, confv, maskv, yv = variant
                X_list.append(Xv); conf_list.append(confv); mask_list.append(maskv); y_list.append(yv)

    X = np.stack(X_list).astype(np.float32)
    C = np.stack(conf_list).astype(np.float32)
    M = np.stack(mask_list).astype(np.float32)
    Y = np.stack(y_list).astype(np.int64)
    return X, C, M, Y


# ============================================================
# Summary / verification
# ============================================================
def label_counts(samples):
    counts = {code: 0 for code in LABEL_CODES}
    clean = both = 0
    for s in samples:
        total = int(s.label.sum())
        if total == 0:
            clean += 1
        if total == len(LABEL_CODES):
            both += 1
        for i, code in enumerate(LABEL_CODES):
            if s.label[i]:
                counts[code] += 1
    return counts, clean, both


def print_summary(stats, reps_by_split, arrays, train_subjects, test_subjects):
    print("\n=== Files processed ===")
    for fname, subject, split, n_reps in stats["files"]:
        print(f"  [{split:5s}] {subject:10s} {fname}  (completed reps: {n_reps})")

    print("\n=== Rep totals ===")
    print(f"  Total completed reps found: {stats['total_completed']}")
    print(f"  Dropped as degraded (< {MIN_VALID_FRAMES} valid frames): {stats['dropped_degraded']}")

    print("\n=== Per-split rep counts (before augmentation) ===")
    for split in ("train", "test"):
        print(f"  {split}: {len(reps_by_split[split])} reps")

    print("\n=== Per-label positive counts per split (before augmentation) ===")
    for split in ("train", "test"):
        counts, clean, both = label_counts(reps_by_split[split])
        print(f"  {split}:")
        for code in LABEL_CODES:
            print(f"    {code}=1: {counts[code]}")
        print(f"    clean [0,0]: {clean}")
        print(f"    both {[1] * len(LABEL_CODES)}: {both}")
        for code in LABEL_CODES:
            if counts[code] == 0:
                sev = "ERROR" if split == "train" else "WARNING"
                print(f"    [{sev}] {code} has zero positive examples in {split}")

    n_before = len(reps_by_split["train"])
    n_after = arrays["y_train"].shape[0]
    factor = n_after / n_before if n_before else 0.0
    print("\n=== Augmentation ===")
    print(f"  Train size before augmentation: {n_before}")
    print(f"  Train size after augmentation:  {n_after}")
    print(f"  Multiplication factor: {factor:.2f}x")

    print("\n=== fault_type vs actual-label cross-check ===")
    if stats["fault_mismatches"]:
        for m in stats["fault_mismatches"]:
            print(f"  [WARNING] {m}")
    else:
        print("  No mismatches found.")

    print("\n=== Final array shapes ===")
    for name, arr in arrays.items():
        print(f"  {name}: {arr.shape} dtype={arr.dtype}")

    print("\n=== CONFIRM ===")
    overlap = train_subjects & test_subjects
    assert not overlap, f"subject leak: {overlap}"
    print(f"  [OK] no subject overlap (train={sorted(train_subjects)}, test={sorted(test_subjects)})")

    for name, arr in arrays.items():
        assert np.isfinite(arr).all(), f"{name} contains NaN/inf"
    print("  [OK] no NaN/inf in any saved array")

    for name in ("y_train", "y_test"):
        values = set(np.unique(arrays[name]).tolist())
        assert values.issubset({0, 1}), f"{name} has non-binary label values: {values}"
    print(f"  [OK] all label rows are binary combinations of {LABEL_CODES}")

    print(f"\nSeed used: {SEED}")


# ============================================================
# Main
# ============================================================
def main():
    rng = np.random.default_rng(SEED)

    manifest_rows = load_manifest()
    reps_by_split, stats = process_manifest(manifest_rows)
    train_subjects, test_subjects = assert_no_subject_leak(reps_by_split)

    X_train, conf_train, mask_train, y_train = build_split_arrays(
        reps_by_split["train"], augment=True, rng=rng
    )
    X_test, conf_test, mask_test, y_test = build_split_arrays(
        reps_by_split["test"], augment=False, rng=rng
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X_train": X_train, "conf_train": conf_train, "mask_train": mask_train, "y_train": y_train,
        "X_test": X_test, "conf_test": conf_test, "mask_test": mask_test, "y_test": y_test,
    }
    for name, arr in arrays.items():
        np.save(OUT_DIR / f"{name}.npy", arr)

    with open(OUT_DIR / "label_names.json", "w") as f:
        json.dump({"label_codes": LABEL_CODES, "window_len": WINDOW_LEN}, f, indent=2)

    print_summary(stats, reps_by_split, arrays, train_subjects, test_subjects)
    print(f"\nSaved arrays + label_names.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
