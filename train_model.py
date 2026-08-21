"""
================================================================================
NeuralStride AI - CNN-LSTM Trainer (Leg Raise, multi-label)
================================================================================

Trains a small multi-label CNN-LSTM on the arrays build_dataset.py produced in
dataset/built/legraise/. Two independent logit outputs: [KNEE_BENT, SHALLOW]
(order comes from label_names.json, never hardcoded; see WEIGHTED LOSS NOTE
below for why the outputs are logits, not sigmoid probabilities). This is a
proof-of-concept
on a small dataset — the goal is an honest, well-evaluated model, not maximal
accuracy, so evaluation is reported per-label with confusion matrices at
several thresholds rather than a single overall-accuracy number.

Runs in venv-train (needs tensorflow in addition to numpy):
    venv-train\\Scripts\\pip install tensorflow
    venv-train\\Scripts\\python train_model.py

MASKING NOTE: padded timesteps make up a large fraction of this window (reps
are 13-23 frames inside a 32-frame window), so ignoring padding correctly
matters a lot here. Keras's automatic mask propagation (from a Masking layer)
does NOT survive Conv1D/BatchNormalization/MaxPooling1D — those layers don't
propagate it, and Conv1D's bias can turn zero-padded timesteps non-zero. So
this file carries the true mask through by hand (MaskMultiply, PoolMask below)
instead of relying on a Masking layer, and hands a correctly-downsampled mask
to the LSTM explicitly.

SUBJECT-GROUPED VALIDATION: build_dataset.py's saved .npy arrays don't carry
per-row subject labels. Rather than duplicate its manifest/JSONL-parsing logic
(and risk it drifting out of sync), this file imports build_dataset directly
and replays process_manifest() to recover which subject each row came from
(train and test), then cross-checks the recovered labels against the saved
y_train.npy/y_test.npy as a consistency guard — see recover_subjects().

WEIGHTED LOSS NOTE: train is imbalanced per-label (KNEE_BENT vs SHALLOW), and
Keras's class_weight doesn't support per-label weights for multi-label output.
Instead this file uses tf.nn.weighted_cross_entropy_with_logits, which needs
raw logits (not post-sigmoid probabilities) to stay numerically stable. So the
final Dense layer has NO activation during training/saving — the model file
on disk outputs logits, not probabilities. Sigmoid is applied by hand after
model.predict() (see sigmoid() below) for both the accuracy metric and eval.
Any future inference wrapper loading legraise_cnn_lstm.keras must do the same.
================================================================================
"""

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers

import build_dataset as bd

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
VAL_SUBJECT = "Tashfeen"   # TRAIN subject held out for validation (configurable)
AUG_FACTOR = 4              # build_dataset.build_split_arrays: original + mirror + time_warp + noise, in that fixed order

BATCH_SIZE = 16
MAX_EPOCHS = 150
EARLY_STOP_PATIENCE = 15
THRESHOLDS = (0.3, 0.5, 0.7)

PROJECT_ROOT = Path(__file__).resolve().parent
BUILT_DIR = PROJECT_ROOT / "dataset" / "built" / "legraise"
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# Custom layers — hand-carried masking (see module docstring)
# ============================================================
@keras.saving.register_keras_serializable(package="legraise")
class MaskMultiply(layers.Layer):
    """Zeros out padded timesteps of a (batch, T, C) tensor using a
    (batch, T) 0/1 mask. Re-applied after the Conv1D/BatchNorm block because
    those layers can turn zero-padded input non-zero (bias, BN shift)."""

    def call(self, inputs):
        x, mask = inputs
        return x * tf.expand_dims(mask, axis=-1)


@keras.saving.register_keras_serializable(package="legraise")
class PoolMask(layers.Layer):
    """Downsamples a (batch, T) 0/1 mask to match MaxPooling1D(2)'s output
    timestep count. Padding is a contiguous run at the END (build_dataset.py
    pads there, never in the middle), so a pooled step is real iff its first
    source step was real — hence the stride-2 slice from index 0."""

    def call(self, mask):
        return mask[:, 0::2]


@keras.saving.register_keras_serializable(package="legraise")
class WeightedBinaryCrossentropy(keras.losses.Loss):
    """Per-label weighted BCE on logits: loss = -[pos_weight*y*log(sigmoid(x))
    + (1-y)*log(1-sigmoid(x))], via tf.nn.weighted_cross_entropy_with_logits
    (numerically stable on raw logits, unlike computing this by hand on
    clipped probabilities). Averaged across labels then across the batch."""

    def __init__(self, pos_weight, name="weighted_bce_logits", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = [float(w) for w in pos_weight]

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        pos_weight_tensor = tf.constant(self.pos_weight, dtype=tf.float32)
        per_label = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=y_pred, pos_weight=pos_weight_tensor
        )
        return tf.reduce_mean(per_label, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"pos_weight": self.pos_weight})
        return config


@keras.saving.register_keras_serializable(package="legraise")
def binary_accuracy_from_logits(y_true, y_pred):
    """binary_accuracy, but applies sigmoid first since the model outputs logits."""
    return keras.metrics.binary_accuracy(y_true, tf.sigmoid(y_pred))


def sigmoid(x):
    """Numerically-stable sigmoid for turning saved-model logits back into
    probabilities (mirrors what any future inference wrapper must also do)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def compute_pos_weights(y_pos_source, label_codes):
    """pos_weight[label] = num_negative / num_positive, computed from the
    actual rows handed to model.fit (post val-split, post-augmentation —
    augmentation preserves labels so the ratio is unaffected either way)."""
    weights = []
    print("Per-label pos_weight (computed from TRAIN-fit rows):")
    for i, code in enumerate(label_codes):
        pos = int(y_pos_source[:, i].sum())
        neg = int(y_pos_source.shape[0] - pos)
        w = neg / pos if pos > 0 else 1.0
        weights.append(w)
        print(f"  {code}: pos_weight={w:.4f}  (pos={pos}, neg={neg})")
    return weights


# ============================================================
# Seeds
# ============================================================
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    print(f"Seed: {seed}")


# ============================================================
# Load config + arrays
# ============================================================
def load_label_config():
    with open(BUILT_DIR / "label_names.json") as f:
        cfg = json.load(f)
    label_codes = cfg["label_codes"]
    window_len = cfg["window_len"]
    assert label_codes == bd.LABEL_CODES, "label_names.json out of sync with build_dataset.LABEL_CODES"
    assert window_len == bd.WINDOW_LEN, "label_names.json out of sync with build_dataset.WINDOW_LEN"
    return label_codes, window_len


def load_arrays():
    arrays = {}
    for split in ("train", "test"):
        for name in ("X", "conf", "mask", "y"):
            arrays[f"{name}_{split}"] = np.load(BUILT_DIR / f"{name}_{split}.npy")
    return arrays


def recover_subjects(arrays):
    """Replay build_dataset's manifest/rep parsing (imported, not duplicated)
    to recover each row's subject — TRAIN rows are expanded x AUG_FACTOR (the
    exact order build_split_arrays() writes augmented variants in); TEST rows
    are 1:1 since build_dataset.py never augments test. Returns
    (train_subjects, test_subjects); recomputed labels are cross-checked
    against the saved y_train.npy/y_test.npy so any drift in build_dataset.py
    is caught loudly instead of silently misgrouping validation or the
    per-subject breakdown."""
    manifest_rows = bd.load_manifest()
    reps_by_split, _ = bd.process_manifest(manifest_rows)

    train_reps = reps_by_split["train"]
    train_subjects, train_labels = [], []
    for rep in train_reps:
        train_subjects.extend([rep.subject] * AUG_FACTOR)
        train_labels.extend([rep.label] * AUG_FACTOR)
    train_subjects = np.array(train_subjects)
    train_labels = np.stack(train_labels)

    test_reps = reps_by_split["test"]
    test_subjects = np.array([r.subject for r in test_reps])
    test_labels = np.stack([r.label for r in test_reps])

    n_train_rows = arrays["y_train"].shape[0]
    if train_subjects.shape[0] != n_train_rows:
        raise RuntimeError(
            f"Recovered {train_subjects.shape[0]} train-row subjects but X_train has "
            f"{n_train_rows} rows. build_dataset.py's augmentation must have "
            "changed — update AUG_FACTOR (or the recovery logic) in train_model.py, "
            "or rerun build_dataset.py if dataset/built/legraise/ is stale."
        )
    n_test_rows = arrays["y_test"].shape[0]
    if test_subjects.shape[0] != n_test_rows:
        raise RuntimeError(
            f"Recovered {test_subjects.shape[0]} test-row subjects but X_test has "
            f"{n_test_rows} rows — rerun build_dataset.py if dataset/built/legraise/ is stale."
        )

    assert np.array_equal(train_labels, arrays["y_train"]), (
        "Recomputed TRAIN labels from build_dataset.py don't match saved y_train.npy — "
        "dataset/built/legraise/ looks stale; rerun build_dataset.py."
    )
    assert np.array_equal(test_labels, arrays["y_test"]), (
        "Recomputed TEST labels from build_dataset.py don't match saved y_test.npy — "
        "dataset/built/legraise/ looks stale; rerun build_dataset.py."
    )
    return train_subjects, test_subjects


# ============================================================
# Normalisation — TRAIN-only stats, feature channels only (not confidence)
# ============================================================
def compute_norm_stats(X_feat_train, mask_train):
    real_frames = X_feat_train[mask_train.astype(bool)]
    mean = real_frames.mean(axis=0).astype(np.float32)
    std = real_frames.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def normalize(X_feat, mean, std):
    return (X_feat - mean) / std


def build_model_inputs(X_feat_norm, conf):
    """Concatenate normalized features and (un-normalized) confidence along
    the channel axis: (N, T, 22) + (N, T, 22) -> (N, T, 44)."""
    return np.concatenate([X_feat_norm, conf], axis=-1).astype(np.float32)


# ============================================================
# Model
# ============================================================
def build_model(window_len, num_channels, num_labels, pos_weight):
    if window_len % 2 != 0:
        raise ValueError("window_len must be even for this architecture's single MaxPooling1D(2)")

    feat_input = keras.Input(shape=(window_len, num_channels), name="features")
    mask_input = keras.Input(shape=(window_len,), name="mask")

    x = MaskMultiply()([feat_input, mask_input])
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = MaskMultiply()([x, mask_input])
    x = layers.MaxPooling1D(2)(x)

    pooled_mask = PoolMask()(mask_input)
    x = layers.Bidirectional(layers.LSTM(32))(x, mask=pooled_mask)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # Linear (logits) output — see WEIGHTED LOSS NOTE at top of file.
    outputs = layers.Dense(num_labels, activation=None)(x)

    model = keras.Model(inputs=[feat_input, mask_input], outputs=outputs, name="legraise_cnn_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=WeightedBinaryCrossentropy(pos_weight),
        metrics=[binary_accuracy_from_logits],
    )
    return model


# ============================================================
# Evaluation — precision/recall/F1/confusion per label, no sklearn needed
# ============================================================
def binary_confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def roc_auc(y_true, y_score):
    """Rank-sum (Mann-Whitney U) AUC — avoids an sklearn dependency."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score))
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_subject_breakdown(y_test, y_prob, test_subjects, label_codes, threshold=0.5):
    """Per-label precision/recall/F1 at `threshold`, computed separately for
    each test subject — shows whether the model generalises across bodies or
    just fits the one it saw most of (here: Mujtaba vs Rohail)."""
    breakdown = {}
    for subject in sorted(set(test_subjects.tolist())):
        mask = test_subjects == subject
        breakdown[subject] = {}
        for i, code in enumerate(label_codes):
            yt = y_test[mask, i]
            yp = (y_prob[mask, i] >= threshold).astype(int)
            conf = binary_confusion(yt, yp)
            precision, recall, f1 = precision_recall_f1(conf["tp"], conf["fp"], conf["fn"])
            breakdown[subject][code] = {
                **conf, "precision": precision, "recall": recall, "f1": f1,
                "n_positive": int(yt.sum()), "n_rows": int(mask.sum()),
            }
    return breakdown


def evaluate(y_test, y_prob, label_codes, test_subjects=None):
    metrics = {"labels": {}, "thresholds": list(THRESHOLDS)}
    for i, code in enumerate(label_codes):
        yt, yp = y_test[:, i], y_prob[:, i]
        label_metrics = {
            "n_positive": int(yt.sum()),
            "n_negative": int(len(yt) - yt.sum()),
            "auc": roc_auc(yt, yp),
            "by_threshold": {},
        }
        for thr in THRESHOLDS:
            pred = (yp >= thr).astype(int)
            conf = binary_confusion(yt, pred)
            precision, recall, f1 = precision_recall_f1(conf["tp"], conf["fp"], conf["fn"])
            label_metrics["by_threshold"][str(thr)] = {**conf, "precision": precision, "recall": recall, "f1": f1}
        metrics["labels"][code] = label_metrics

    y_pred_05 = (y_prob >= 0.5).astype(int)
    metrics["exact_match_accuracy"] = float(np.mean(np.all(y_pred_05 == y_test, axis=1)))

    pred_clean = (y_pred_05.sum(axis=1) == 0).astype(int)
    true_clean = (y_test.sum(axis=1) == 0).astype(int)
    conf_clean = binary_confusion(true_clean, pred_clean)
    p, r, f1 = precision_recall_f1(conf_clean["tp"], conf_clean["fp"], conf_clean["fn"])
    metrics["clean_detection"] = {**conf_clean, "precision": p, "recall": r, "f1": f1, "n_true_clean": int(true_clean.sum())}

    metrics["n_test"] = int(len(y_test))
    metrics["n_test_both_faults"] = int(np.sum(y_test.sum(axis=1) == y_test.shape[1]))

    if test_subjects is not None:
        metrics["per_subject"] = per_subject_breakdown(y_test, y_prob, test_subjects, label_codes)

    return metrics


def print_evaluation(metrics, label_codes):
    print("\n=== TEST SET EVALUATION (held-out subject, never trained/validated on) ===")
    per_label_counts = ", ".join(f"{c}={metrics['labels'][c]['n_positive']}" for c in label_codes)
    print(f"  n_test={metrics['n_test']} ({per_label_counts}, both={metrics['n_test_both_faults']})")
    print("  Small-sample caveat: per-fault numbers are indicative; the 'both faults' count")
    print("  especially is small and noisy — don't over-read single-rep swings.")

    for code in label_codes:
        lm = metrics["labels"][code]
        print(f"\n  --- {code} ---")
        auc_str = f"{lm['auc']:.3f}" if lm["auc"] is not None else "n/a (single class in test)"
        print(f"  AUC: {auc_str}  (n_pos={lm['n_positive']}, n_neg={lm['n_negative']})")
        print(f"  {'thr':>5} {'prec':>7} {'rec':>7} {'f1':>7}   TP  TN  FP  FN")
        for thr in metrics["thresholds"]:
            bt = lm["by_threshold"][str(thr)]
            print(f"  {thr:>5} {bt['precision']:>7.3f} {bt['recall']:>7.3f} {bt['f1']:>7.3f}   "
                  f"{bt['tp']:>3} {bt['tn']:>3} {bt['fp']:>3} {bt['fn']:>3}")

    cd = metrics["clean_detection"]
    print("\n  --- Clean-rep detection (predicted [0,0], threshold 0.5) ---")
    print(f"  precision={cd['precision']:.3f} recall={cd['recall']:.3f} f1={cd['f1']:.3f}  "
          f"(TP={cd['tp']} TN={cd['tn']} FP={cd['fp']} FN={cd['fn']}, n_true_clean={cd['n_true_clean']})")

    print(f"\n  Exact-match accuracy (both labels correct, threshold 0.5, strict): "
          f"{metrics['exact_match_accuracy']:.3f}")

    if "per_subject" in metrics:
        print("\n  --- Per-subject breakdown (threshold 0.5) — generalisation check ---")
        for subject, subj_metrics in metrics["per_subject"].items():
            print(f"  {subject}:")
            for code in label_codes:
                sm = subj_metrics[code]
                print(f"    {code}: recall={sm['recall']:.3f} precision={sm['precision']:.3f} "
                      f"f1={sm['f1']:.3f}  (n_pos={sm['n_positive']}/{sm['n_rows']}, "
                      f"TP={sm['tp']} TN={sm['tn']} FP={sm['fp']} FN={sm['fn']})")

    print(f"\n  Val subject: {metrics['val_subject']}   Seed: {metrics['seed']}")
    print(f"  Epochs run: {metrics['epochs_run']}   "
          f"final train_loss={metrics['final_train_loss']:.4f}  final val_loss={metrics['final_val_loss']:.4f}  "
          f"best val_loss={metrics['best_val_loss']:.4f}")


def maybe_plot_curve(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping training-curve PNG (history JSON still saved).")
        return
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("binary_crossentropy")
    plt.title("Leg Raise CNN-LSTM training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curve plot to {path}")


# ============================================================
# Main
# ============================================================
def main():
    set_seeds(SEED)

    label_codes, window_len = load_label_config()
    arrays = load_arrays()

    train_subjects, test_subjects = recover_subjects(arrays)

    val_row_mask = train_subjects == VAL_SUBJECT
    if not val_row_mask.any():
        raise ValueError(f"VAL_SUBJECT={VAL_SUBJECT!r} not found among train subjects: {sorted(set(train_subjects))}")
    print(f"Validation subject (held out from TRAIN, subject-grouped): {VAL_SUBJECT}")
    print(f"  train rows: {int((~val_row_mask).sum())}   val rows: {int(val_row_mask.sum())}")

    mean, std = compute_norm_stats(arrays["X_train"], arrays["mask_train"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(BUILT_DIR / "norm_stats.npz", mean=mean, std=std)
    np.savez(MODELS_DIR / "legraise_norm_stats.npz", mean=mean, std=std)
    print(f"Normalisation stats (train-only, real frames only) saved to "
          f"{BUILT_DIR / 'norm_stats.npz'} and {MODELS_DIR / 'legraise_norm_stats.npz'}")

    feat_train_all = build_model_inputs(normalize(arrays["X_train"], mean, std), arrays["conf_train"])
    feat_test = build_model_inputs(normalize(arrays["X_test"], mean, std), arrays["conf_test"])

    feat_tr, feat_val = feat_train_all[~val_row_mask], feat_train_all[val_row_mask]
    mask_tr, mask_val = arrays["mask_train"][~val_row_mask], arrays["mask_train"][val_row_mask]
    y_tr = arrays["y_train"][~val_row_mask].astype(np.float32)
    y_val = arrays["y_train"][val_row_mask].astype(np.float32)

    pos_weight = compute_pos_weights(y_tr, label_codes)

    model = build_model(window_len, feat_tr.shape[-1], len(label_codes), pos_weight)
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
    )
    history = model.fit(
        {"features": feat_tr, "mask": mask_tr}, y_tr,
        validation_data=({"features": feat_val, "mask": mask_val}, y_val),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[early_stop],
        verbose=2,
    )

    model_path = MODELS_DIR / "legraise_cnn_lstm.keras"
    model.save(model_path)
    print(f"\nSaved model to {model_path}")

    with open(MODELS_DIR / "legraise_training_history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    maybe_plot_curve(history, MODELS_DIR / "legraise_training_curve.png")

    y_logit_test = model.predict({"features": feat_test, "mask": arrays["mask_test"]}, verbose=0)
    y_prob_test = sigmoid(y_logit_test)
    y_test = arrays["y_test"]

    metrics = evaluate(y_test, y_prob_test, label_codes, test_subjects=test_subjects)
    metrics["val_subject"] = VAL_SUBJECT
    metrics["seed"] = SEED
    metrics["pos_weight"] = {code: pos_weight[i] for i, code in enumerate(label_codes)}
    metrics["epochs_run"] = len(history.history["loss"])
    metrics["final_train_loss"] = float(history.history["loss"][-1])
    metrics["final_val_loss"] = float(history.history["val_loss"][-1])
    metrics["best_val_loss"] = float(min(history.history["val_loss"]))

    with open(MODELS_DIR / "legraise_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print_evaluation(metrics, label_codes)
    print(f"\nSaved metrics to {MODELS_DIR / 'legraise_metrics.json'}")


if __name__ == "__main__":
    main()
