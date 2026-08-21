# NeuralStride — Model Integration Contract & Handoff (Leg Raise)

**From:** Mujtaba (model / Phase 2–3)
**To:** Furqan (backend / frontend / Phase 4+)
**Subject:** How to integrate the leg-raise CNN-LSTM into the FastAPI backend

---

## 1. What you're getting

- `legraise_classifier/` — a self-contained Python package. Import it, feed it a completed rep's
  landmark frames, get back fault flags, fault probabilities, and a rep-quality score.
- Its `README.md` — usage, thresholds, honest labelling, known limitations.
- This document — how the model and the rule engine divide responsibility so they never contradict.

The package is self-contained: it carries its own model file, normalization stats, and feature
logic. It only needs `numpy` and `tensorflow` (2.18.1, same as `main.py`'s documented stack). It does
NOT need the rest of the project on the path.

---

## 2. The division of responsibility (the contract)

The system has two detectors. They are complementary, not competing. Wire them so each owns its lane:

### Rule engine (Phase 1) owns:
- **Live, per-frame feedback** during a rep (the "straighten your knee" / "raise higher" coaching).
- **Voice.**
- **The user-facing fault flag / rep-flag count** — what the user is told is wrong, in the moment.
- This is what drives the live red-shading in the Three.js avatar (frame-level, immediate).

### The model owns:
- **Per-rep quality score** (0–100%) — a graded measure of how good the rep was.
- **Per-fault likelihood** (0–100% per fault) — the probability the rep had each fault.
- **Session-level metrics** — average quality, per-fault averages, trend over time.
- This is what drives **progress tracking / dashboards** — the thing binary rules cannot provide.

### Why this split matters
The rule engine is fast and sensitive — better for instant live feedback, and it catches subtle
faults the model may rate below its flag threshold. The model provides continuous, trackable quality
metrics the rule engine cannot. **The summary the user sees for "what was wrong this rep" should come
from the rule engine** (so live feedback and summary never contradict). **The quality score and
progress trend come from the model.** Present them as different things: "form coaching" (rule engine)
vs "rep quality score / progress" (model).

### On disagreement (important)
On some reps the two will differ — e.g. a subtle knee-bend the rule engine flags but the model rates
~34% (below a 0.5 flag threshold). This is expected and by design. Handle it as:
- Live coaching + flag: rule engine wins (it caught it).
- Quality score: model's graded value (a subtle fault correctly yields a middling, not zero, quality).
They don't contradict because they answer different questions ("what to fix now" vs "how good was
the rep overall").

---

## 3. How to call the model (backend)

The model scores a **completed rep**, not a live frame stream. So:

1. Your backend already receives per-frame landmarks (from browser MediaPipe or the Python capture).
2. Maintain a buffer of the raw landmarks for the rep currently in progress. Use the rule engine's
   rep-start / rep-complete signals (the same `RepWindowTracker` logic Phase 1 uses) to know when a
   rep begins and ends.
3. On rep completion, pass that rep's buffered frames to `predict_rep()`:

```python
from legraise_classifier import LegRaiseClassifier

clf = LegRaiseClassifier(threshold_knee=0.3, threshold_shallow=0.5)  # tune thresholds as desired

result = clf.predict_rep(rep_frames)   # rep_frames = list of that rep's per-frame landmarks
# result contains: faults, fault_probabilities, fault_percentages, rep_quality, rep_quality_percent, predicted_clean
```

4. Store `result` per rep (in the DB) and feed the list of results to `aggregate_session()` for the
   session summary that drives the dashboard.

### Threshold tuning
Defaults are 0.5/0.5. For `knee_bent`, a **0.3** threshold catches more (higher recall) at some cost
to precision — recommended if you'd rather flag borderline bends than miss them. This only affects the
boolean `faults` flag; the quality score and probabilities are unaffected. Your call per product need.

### Honest labelling on the UI
The percentages are **probability the fault is present** — "knee-bend likelihood 3%", NOT "the knee is
3% bent". Label them as likelihood/confidence, not severity, on any dashboard.

---

## 4. Known limitation (state it honestly in the product/report)
The model under-detects **very subtle** knee-bends (bends barely past the detection threshold) — it
was trained on clearer bends. The rule engine covers this case live. Documented via per-subject
evaluation: knee-bend recall ~0.56 (Mujtaba, subtle) vs ~0.78 (Rohail, clearer); shallow ~0.94 on
both. This is why the hybrid design exists.

---

## 5. What this does NOT cover (yet)
- Only **Side Leg Raise**. Other exercises fall back to Phase 1 (rule engine) only.
- **Scaption** model is planned next; it will ship as a parallel `scaption_classifier/` package with
  the same interface, so your integration code generalizes.
- Live per-frame model scoring is intentionally NOT provided — the model scores completed reps; live
  frame-level feedback is the rule engine's job.

---

## 6. Environment note
The model needs TensorFlow 2.18.1. It coexists with mediapipe 0.10.14 in a combined env (`venv-live`
here — mediapipe + opencv + TF installed together, protobuf resolved to 4.25.x). If your backend runs
the model, it needs TF installed; if MediaPipe runs in the browser instead, your backend only needs
TF (no mediapipe), which is cleaner. Decide where MediaPipe runs (browser vs backend) — that's the one
architecture point where our work meets.
