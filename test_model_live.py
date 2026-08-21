"""
================================================================================
NeuralStride AI - Leg Raise Model Live Test Harness
================================================================================

Standalone script proving the trained leg-raise CNN-LSTM (packaged in
legraise_classifier/) runs end-to-end on live webcam data and produces sane
per-rep quality/fault numbers. This is a verification checkpoint, not a
product feature — the console prints BOTH the rule engine's verdict (exact,
threshold-based, what main.py already ships) and the model's result
(probabilistic, quality-scored) side by side so a human can compare them
rep by rep before the package is handed off.

REUSES, does not reimplement:
    - detectors.legraise.LegRaiseExercise  — state machine, rep counting,
      the exact KNEE_BENT/SHALLOW rule-engine logic
    - rep_window.RepWindowTracker          — rep boundary/frame-window
      segmentation (start_frame/end_frame per completed rep)
    - session_utils.SessionTracker         — rule-engine session summary
    - pose_utils.create_pose_detector      — MediaPipe Pose setup
    - legraise_classifier.LegRaiseClassifier / aggregate_session — the model

The only new logic here is a small per-frame landmark buffer so a completed
rep's frame window (as reported by RepWindowTracker) can be handed to
LegRaiseClassifier.predict_rep() — that's harness glue, not segmentation.

ENVIRONMENT:
    Needs MediaPipe (webcam capture) AND TensorFlow (the model) in the SAME
    process/venv. The project's two existing venvs don't have both (venv has
    mediapipe, not tensorflow; venv-train has tensorflow, not mediapipe) —
    this script is meant to run in a combined venv (venv-live in this repo)
    that has mediapipe==0.10.14, opencv-python, and tensorflow==2.18.1
    installed together.

RUNNING:
    venv-live\\Scripts\\python test_model_live.py
    Press 'q' in the video window to end the session and print summaries.
================================================================================
"""

import time

import cv2
import mediapipe as mp

from pose_utils import create_pose_detector
from detectors.legraise import LegRaiseExercise
from rep_window import RepWindowTracker
from session_utils import SessionTracker
from legraise_classifier import LegRaiseClassifier, aggregate_session

FRAME_BUFFER_HORIZON = 300  # keep this many recent frames -- generous vs. any single rep's span


def landmarks_to_dicts(landmarks):
    """Snapshot MediaPipe's landmark list into plain dicts immediately —
    MediaPipe reuses/overwrites its internal landmark objects next frame, so
    anything held across frames (our buffer) must be a copy, not a reference."""
    return [
        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
        for lm in landmarks
    ]


def print_rule_engine_verdict(rep_event):
    status = "CLEAN" if rep_event["clean"] else f"ISSUES: {', '.join(rep_event['issues'])}"
    print(f"  [Rule engine] rep {rep_event['rep_id']}: {status} "
          f"(frames {rep_event['start_frame']}-{rep_event['end_frame']}, "
          f"duration {rep_event['duration']:.2f}s)")


def print_model_result(rep_id, result):
    f = result["fault_percentages"]
    print(f"  [Model]       rep {rep_id}: quality={result['rep_quality_percent']}%  "
          f"knee-bend likelihood={f['knee_bent']}%  shallow likelihood={f['shallow']}%  "
          f"predicted_clean={result['predicted_clean']}")


def main():
    print("Loading leg-raise model...")
    classifier = LegRaiseClassifier()
    print(f"Model loaded. label_codes={classifier.label_codes}  "
          f"thresholds: knee={classifier.threshold_knee} shallow={classifier.threshold_shallow}\n")

    pose, mp_pose = create_pose_detector()
    mp_drawing = mp.solutions.drawing_utils

    exercise = LegRaiseExercise()
    rep_tracker = RepWindowTracker(idle_state=exercise.get_idle_state())
    session = SessionTracker(exercise_name="Side Leg Raise")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "NeuralStride AI - Leg Raise Model Live Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_buffer = {}   # frame_number -> list of 33 landmark dicts
    frame_number = 0
    model_rep_results = []  # for the aggregator, in rep order

    print("Live test started. Do Side Leg Raise reps in front of the camera. Press 'q' to end.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        now = time.time()
        frame_number += 1
        rep_event = None
        exercise_state = "INVALID"
        rep_count = exercise.get_rep_count()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            frame_buffer[frame_number] = landmarks_to_dicts(landmarks)
            for fn in list(frame_buffer.keys()):
                if fn < frame_number - FRAME_BUFFER_HORIZON:
                    del frame_buffer[fn]

            timestamp = now - session.start_time
            ready, not_ready_msg = exercise.check_ready(landmarks, mp_pose)
            already_tracking = exercise.is_tracking()

            if ready or already_tracking:
                result = exercise.update(landmarks, mp_pose, w, h, now)
                exercise_state = result.state
                rep_count = result.rep_count

                if result.rep_completed:
                    session.log_rep(
                        rep_count,
                        had_form_issues=result.rep_had_issues,
                        issues=result.rep_issues,
                        avg_knee_angle=result.rep_avg_angle,
                    )

                rep_event = rep_tracker.update(frame_number, timestamp, result)
            else:
                exercise.reset()
                rep_event = rep_tracker.abort(frame_number, timestamp)
        else:
            exercise_state = "INVALID"
            rep_event = rep_tracker.abort(frame_number, now - session.start_time)

        if rep_event is not None:
            if rep_event["completed"]:
                print_rule_engine_verdict(rep_event)
                rep_frames = [
                    frame_buffer[fn]
                    for fn in range(rep_event["start_frame"], rep_event["end_frame"] + 1)
                    if fn in frame_buffer
                ]
                try:
                    model_result = classifier.predict_rep(rep_frames)
                    model_rep_results.append(model_result)
                    print_model_result(rep_event["rep_id"], model_result)
                except ValueError as e:
                    print(f"  [Model]       rep {rep_event['rep_id']}: skipped -- {e}")
                print()
            else:
                print(f"  [Rule engine] rep {rep_event['rep_id']}: ABORTED (not scored by model)\n")

        cv2.putText(frame, f"State: {exercise_state}  Reps: {rep_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to end session", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    session.end_session()
    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    rule_summary = session.get_summary()
    model_summary = aggregate_session(model_rep_results)

    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print("\n--- Rule engine (Phase 1) ---")
    print(f"  Total reps:       {rule_summary['total_reps']}")
    print(f"  Clean reps:       {rule_summary['clean_reps']}")
    print(f"  Form issue reps:  {rule_summary['form_issue_reps']}")
    print(f"  Form accuracy:    {rule_summary['clean_percentage']}%")
    print(f"  Form issues:      {rule_summary['form_issues']}")

    print("\n--- Model (this package) ---")
    print(f"  Total reps scored:     {model_summary['total_reps']}")
    print(f"  Avg rep quality:       {model_summary['avg_rep_quality_percent']}%")
    print(f"  Avg knee-bend likelihood: {model_summary['avg_fault_percent']['knee_bent']}%")
    print(f"  Avg shallow likelihood:   {model_summary['avg_fault_percent']['shallow']}%")
    print(f"  Flagged (knee_bent):   {model_summary['flagged_reps']['knee_bent']}")
    print(f"  Flagged (shallow):     {model_summary['flagged_reps']['shallow']}")
    print(f"  Predicted clean reps:  {model_summary['clean_reps']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
