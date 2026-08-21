"""
================================================================================
NeuralStride AI - Live App + Model Scoring
================================================================================

Full Phase 1 live experience (main.py: camera window, skeleton overlay, HUD,
voice, rep counting, CSV/JSONL export, WebSocket) PLUS the trained leg-raise
model's per-rep quality/fault scoring shown on screen and in the console.

WHY A SEPARATE ENTRY SCRIPT (not a --model flag on main.py):
    Running `python main.py` stays completely untouched -- no new flag to
    parse, no behavior branch to reason about, no risk of the model path
    affecting a plain Phase 1 session. The only change made to main.py was
    extracting its camera/loop/summary body into
    run_live_session(..., extensions=None) so THIS script can import and
    reuse that exact code instead of copy-pasting it -- when extensions is
    None (what main() itself still passes), run_live_session() reproduces
    the old inline behavior exactly. See main.py's run_live_session()
    docstring for the four hook points that make this work.

WHAT THIS ADDS ON TOP (via the ModelScoringExtensions hook object below):
    - on_frame():       buffers each frame's raw landmarks in a rolling
                         window (harness glue, not rep segmentation -- rep
                         boundaries still come entirely from
                         rep_window.RepWindowTracker inside run_live_session)
    - on_rep_event():   when a rep completes, hands its buffered frame
                         window to LegRaiseClassifier.predict_rep() once
                         (not per-frame) and prints the result
    - draw_overlay():   draws a small corner panel on the video frame with
                         the last rep's quality/fault %, and the running
                         session average quality
    - on_session_end(): prints the model's aggregate_session summary right
                         after the rule engine's existing console summary

HYBRID CONTRACT (see legraise_classifier/README.md):
    The rule engine still owns every LIVE decision -- voice feedback, HUD
    state, per-frame fault flags, rep counting. None of that is touched
    here. The model only ever scores a rep AFTER RepWindowTracker reports it
    complete (post-hoc), and only ever adds display/print output -- it never
    drives voice and never feeds back into rule-engine behavior.

ENVIRONMENT: venv-live (mediapipe 0.10.14 + tensorflow 2.18.1 together --
    the project's other two venvs each have only one of those).

RUNNING:
    venv-live\\Scripts\\python main_with_model.py
================================================================================
"""

import cv2

from main import show_exercise_menu_and_select, prompt_session_metadata, run_live_session
from legraise_classifier import LegRaiseClassifier, aggregate_session

FRAME_BUFFER_HORIZON = 300  # keep this many recent frames -- generous vs. any single rep's span

# Corner panel placement -- top-right, below the existing HUD's FPS counter
# (display_utils.draw_hud puts FPS at (560,15)) and well above the existing
# bottom-right elapsed-time text (490,470), so nothing overlaps.
PANEL_X0, PANEL_Y0 = 345, 40
PANEL_X1, PANEL_Y1 = 635, 155


class ModelScoringExtensions:
    """
    Hook object passed to main.run_live_session(extensions=...). Implements
    the four methods run_live_session calls at defined points in its loop
    (see that function's docstring) -- this class owns all model-side state
    (frame buffer, per-rep results) so run_live_session itself stays
    completely unaware the model exists.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.frame_buffer = {}       # frame_number -> list of 33 landmark dicts
        self.rep_results = []        # predict_rep() results, in rep order, for the aggregator
        self.last_rep_id = None
        self.last_result = None      # last predict_rep() result, or None if skipped/none yet
        self.last_skipped_reason = None

    # --- run_live_session hooks ---------------------------------------

    def on_frame(self, frame_number, landmarks):
        self.frame_buffer[frame_number] = [
            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in landmarks
        ]
        horizon = frame_number - FRAME_BUFFER_HORIZON
        for fn in list(self.frame_buffer.keys()):
            if fn < horizon:
                del self.frame_buffer[fn]

    def on_rep_event(self, rep_event):
        if not rep_event.get("completed"):
            return  # aborted reps aren't scored -- not a real attempt

        rep_frames = [
            self.frame_buffer[fn]
            for fn in range(rep_event["start_frame"], rep_event["end_frame"] + 1)
            if fn in self.frame_buffer
        ]
        self.last_rep_id = rep_event["rep_id"]

        try:
            result = self.classifier.predict_rep(rep_frames)
        except ValueError as e:
            self.last_result = None
            self.last_skipped_reason = str(e)
            print(f"  [Model] rep {rep_event['rep_id']}: skipped -- {e}")
            return

        self.last_result = result
        self.last_skipped_reason = None
        self.rep_results.append(result)
        f = result["fault_probabilities"]
        print(f"  [Model] rep {rep_event['rep_id']}: quality={result['rep_quality_percent']}%  "
              f"knee-bend likelihood={result['fault_percentages']['knee_bent']}%  "
              f"shallow likelihood={result['fault_percentages']['shallow']}%  "
              f"predicted_clean={result['predicted_clean']}  (p_knee={f['knee_bent']} p_shallow={f['shallow']})")

    def draw_overlay(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (PANEL_X0, PANEL_Y0), (PANEL_X1, PANEL_Y1), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
        cv2.rectangle(frame, (PANEL_X0, PANEL_Y0), (PANEL_X1, PANEL_Y1), (90, 90, 90), 1)

        x = PANEL_X0 + 10
        y = PANEL_Y0 + 20
        cv2.putText(frame, "MODEL (last rep)", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 22

        if self.last_result is None and self.last_skipped_reason is None:
            cv2.putText(frame, "No reps scored yet", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        elif self.last_result is None:
            cv2.putText(frame, f"Rep {self.last_rep_id}: skipped (degraded)", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)
        else:
            r = self.last_result
            dot_color = (0, 200, 0) if r["predicted_clean"] else (0, 100, 255)
            cv2.circle(frame, (x + 4, y - 4), 5, dot_color, -1)
            cv2.putText(frame, f"Rep {self.last_rep_id} quality: {r['rep_quality_percent']}%",
                        (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            y += 20
            f = r["fault_percentages"]
            cv2.putText(frame, f"knee-bend: {f['knee_bent']}%  shallow: {f['shallow']}%",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        y += 26
        cv2.line(frame, (x, y - 13), (PANEL_X1 - 10, y - 13), (80, 80, 80), 1)
        if self.rep_results:
            avg_q = sum(r["rep_quality_percent"] for r in self.rep_results) / len(self.rep_results)
            cv2.putText(frame, f"Session avg quality: {avg_q:.1f}% ({len(self.rep_results)} scored)",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Session avg quality: --", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def on_session_end(self):
        summary = aggregate_session(self.rep_results)
        print("\n--- Model session summary ---")
        print(f"  Total reps scored:        {summary['total_reps']}")
        print(f"  Avg rep quality:          {summary['avg_rep_quality_percent']}%")
        print(f"  Avg knee-bend likelihood: {summary['avg_fault_percent']['knee_bent']}%")
        print(f"  Avg shallow likelihood:   {summary['avg_fault_percent']['shallow']}%")
        print(f"  Flagged (knee_bent):      {summary['flagged_reps']['knee_bent']}")
        print(f"  Flagged (shallow):        {summary['flagged_reps']['shallow']}")
        print(f"  Predicted clean reps:     {summary['clean_reps']}")
        print("------------------------------\n")


def main():
    print("Loading leg-raise model...")
    classifier = LegRaiseClassifier()
    print(f"Model loaded. label_codes={classifier.label_codes}  "
          f"thresholds: knee={classifier.threshold_knee} shallow={classifier.threshold_shallow}\n")

    exercise, exercise_name, target_time = show_exercise_menu_and_select()
    subject_id, camera_view, form_intent = prompt_session_metadata(exercise_name)

    if exercise_name != "Side Leg Raise":
        print(f"\nNote: the trained model only covers Side Leg Raise. Running "
              f"{exercise_name} with Phase 1 only -- no model scoring this session.\n")
        extensions = None
    else:
        extensions = ModelScoringExtensions(classifier)

    run_live_session(
        exercise, exercise_name, target_time, subject_id, camera_view, form_intent,
        extensions=extensions,
    )


if __name__ == "__main__":
    main()
