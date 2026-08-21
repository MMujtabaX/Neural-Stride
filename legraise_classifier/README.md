# legraise_classifier

Self-contained inference wrapper around the trained NeuralStride leg-raise
CNN-LSTM. Drop this whole folder into another project (e.g. a FastAPI
backend) — it does not import anything else from the NeuralStride repo, only
`numpy`, `tensorflow`/`keras`, and Python's standard library.

## What this is (and isn't)

This package is **the model half only**. The rule engine
(`detectors/legraise.py` in the main project) still owns:
- Live per-frame feedback (voice cues, on-screen "Straighten your knee")
- Rep counting and the UP/MID/DOWN state machine
- The user-facing fault flag during the set (KNEE_BENT / SHALLOW), which is
  exact and instant because it's a deterministic threshold check, not a
  prediction

This package adds, **on top of** the rule engine, once a rep is complete:
- A continuous **quality score** per rep (not just pass/fail)
- **Fault probabilities** — how likely each fault is present, not a severity
  score
- **Session-level progress metrics** (average quality over a workout, trend
  over time) that the rule engine has no notion of

If you only need the existing pass/fail behavior, you don't need this
package. Use it when you want a quality score, a probability instead of a
hard flag, or session-level aggregation.

## Install

```
pip install tensorflow==2.18.1 numpy
```
(This must match the training environment's TensorFlow version —
**TF 2.18.1, keras 3.15.1** — the `.keras` file's custom layers were saved
against that version. A different TF major/minor version may still load it,
but hasn't been verified.)

## Usage

```python
from legraise_classifier import LegRaiseClassifier, aggregate_session

clf = LegRaiseClassifier()  # loads legraise_cnn_lstm.keras + norm stats + label config from this folder

# rep_frames: list of frames for ONE completed rep, in order.
# Each frame is a sequence of 33 landmarks (MediaPipe Pose order, index 0-32).
# Each landmark can be a MediaPipe NormalizedLandmark object OR a dict:
#   {"x": 0.51, "y": 0.42, "z": -0.31, "visibility": 0.98}
result = clf.predict_rep(rep_frames)
```

`result`:
```python
{
  "faults": {
      "knee_bent": bool,        # p_knee >= threshold_knee
      "shallow":   bool,        # p_shallow >= threshold_shallow
  },
  "fault_probabilities": {      # probability the fault is PRESENT, 0-1
      "knee_bent": 0.82,
      "shallow":   0.03,
  },
  "fault_percentages": {        # same numbers as %, for display
      "knee_bent": 82.0,        # "knee-bend likelihood 82%"
      "shallow":   3.0,         # "shallow-raise likelihood 3%"
  },
  "rep_quality": 0.1746,        # (1-p_knee) * (1-p_shallow), 0-1, higher = better
  "rep_quality_percent": 17.46,
  "predicted_clean": False,     # True iff neither fault flag is set
}
```

**Read the percentages honestly**: `fault_percentages.knee_bent = 3.0` means
*"the model is 3% confident this rep had a bent knee"* — i.e. very likely a
straight-knee rep. It does **not** mean "the knee was bent by 3%" or "3%
knee bend severity". These are presence-probabilities, not severity scores.
Display them as e.g. "knee-bend likelihood: 3%", never as "knee bend: 3%".

### Thresholds

Constructor params, default `threshold_knee=0.5`, `threshold_shallow=0.5`:

```python
clf = LegRaiseClassifier(threshold_knee=0.3, threshold_shallow=0.5)
```

On the held-out test set (100 reps, 2 subjects), KNEE_BENT recall/precision
at each threshold:

| threshold | precision | recall |
|---|---|---|
| 0.3 | 0.962 | 0.694 |
| 0.5 | 0.960 | 0.667 |
| 0.7 | 1.000 | 0.639 |

Precision barely moves across thresholds, but recall drops as the threshold
rises. **If missing a real knee-bend is worse than an occasional false
alarm, use `threshold_knee=0.3`.** SHALLOW's precision/recall is comparably
flat across thresholds (0.79-0.81 / 0.94), so 0.5 is fine there.

### Session summary

```python
rep_results = [clf.predict_rep(frames) for frames in reps_from_a_session]
summary = aggregate_session(rep_results)
```

```python
{
  "total_reps": 12,
  "avg_rep_quality_percent": 71.4,
  "avg_fault_percent": {"knee_bent": 18.2, "shallow": 9.5},
  "flagged_reps": {"knee_bent": 3, "shallow": 1},
  "clean_reps": 8,
}
```

`aggregate_session` is a pure function — no I/O, no model calls — so it's
safe to call from a request handler, a background job, or a test.

## Known limitation: subtle knee-bends

The model under-detects knee-bends that are only mildly past the
straight-knee threshold. In evaluation, a test subject whose knee-bends
averaged only ~11° past the rule engine's 168° threshold got 5.6% recall;
after re-recording clearer bends (~25° past threshold, in line with the rest
of the dataset) recall rose to 55.6%. **Operating range: the model is
reliable on bends that are clearly, visibly bent (roughly 20°+ past 168°,
i.e. knee angle ≲ 148°). Very marginal bends (168° down to ~150°) are likely
to be missed.** This is a training-data-coverage limitation, not a bug — the
rule engine's exact threshold check does not have this blind spot, which is
part of why it stays the source of truth for live per-rep flags.

## Files in this folder

| file | purpose |
|---|---|
| `classifier.py` | `LegRaiseClassifier`, `aggregate_session`, the custom Keras layers needed to load the model |
| `feature_extract.py` | Self-contained port of the project's feature-extraction math (no mediapipe/project dependency) |
| `legraise_cnn_lstm.keras` | Trained model weights + architecture |
| `legraise_norm_stats.npz` | Train-fit per-feature mean/std (must not be refit — see classifier.py docstring) |
| `label_names.json` | Label order (`["KNEE_BENT", "SHALLOW"]`) and `WINDOW_LEN` (32) |

## Updating the model later

If the model gets retrained, re-copy `legraise_cnn_lstm.keras`,
`legraise_norm_stats.npz`, and `label_names.json` from `models/` and
`dataset/built/legraise/` in the main project into this folder — the norm
stats and model must always come from the *same* training run (mismatched
normalization stats will silently produce wrong predictions, no error).
