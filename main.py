"""
================================================================================
NeuralStride AI - Main Entry Point
================================================================================

This is the main file that runs the NeuralStride AI exercise analysis system.
It handles the exercise selection menu, camera setup, main processing loop,
workout summary display, CSV/JSONL data export, and WebSocket live stream.

RUNNING:
    1. Activate virtual environment:
        .\\venv\\Scripts\\activate
    2. Run:
        python main.py
    3. Select an exercise from the menu (e.g., 1 for Squat, 2 for Push-Up)
    4. Follow the on-screen and voice prompts
    5. (Optional) 3JS person opens browser to ws://localhost:8765 for live 3D view

MAIN LOOP FLOW:
    1. Read frame from webcam, resize and mirror
    2. Process frame with MediaPipe Pose (33 body landmarks)
    3. Draw skeleton overlay on video feed
    4. Export landmarks to JSONL file AND broadcast over WebSocket
    5. Check if person is in correct starting position (exercise.check_ready)
    6. If ready, run exercise logic (exercise.update)
        - Returns: state, feedback, form labels, voice message, rep data
    7. Log completed reps to SessionTracker
    8. Draw HUD (angles, state, rep count, feedback, timer, FPS)
    9. Display frame, check for quit ('q' key)
    10. On quit: show workout summary, save CSV, close WebSocket, close

FILES:
    main.py              -- This file (entry point)
    pose_utils.py        -- MediaPipe setup, standing posture check
    angle_utils.py       -- Angle calculation, smoothing buffers
    voice_utils.py       -- Windows SAPI text-to-speech engine
    display_utils.py     -- Feedback display, workout summary, HUD drawing
    session_utils.py     -- Session tracking, CSV export
    landmark_exporter.py -- Landmark coordinate export (JSONL for 3D viewer)
    video_landmark_extractor.py -- Extract landmarks from uploaded video files
    websocket_server.py  -- WebSocket live stream for real-time 3D viewer
    detectors/
        base.py          -- BaseExercise class (interface for all exercises)
        squat.py         -- Squat exercise logic
        pushup.py        -- Push-Up exercise logic
        plank.py         -- Plank exercise logic
        lunge.py         -- Lunge exercise logic
        __init__.py      -- Exercise registry (add new exercises here)

DEPENDENCIES:
    pip install opencv-python mediapipe==0.10.14 pywin32 numpy websockets

    IMPORTANT: MediaPipe must be 0.10.14 (newer versions removed mp.solutions.pose)
    protobuf must be 4.25.3
================================================================================
"""

import cv2
import mediapipe as mp
import ctypes
import time

from datetime import datetime

from pose_utils import create_pose_detector
from voice_utils import VoiceEngine
from display_utils import FeedbackDisplay, draw_workout_summary, draw_hud
from session_utils import SessionTracker
from landmark_exporter import LandmarkExporter
from feature_utils import extract_features
from rep_window import RepWindowTracker
from websocket_server import WebSocketBroadcaster
from detectors import show_exercise_menu, get_exercise

DIAGONAL_VIEW_EXERCISES = ("Push-Up", "Lunge")
VALID_CAMERA_VIEWS = ("side", "front", "diagonal")
VALID_FORM_INTENTS = ("good", "bad", "mixed")


def show_exercise_menu_and_select():
    show_exercise_menu()
    while True:
        try:
            choice = int(input("\nEnter exercise number: "))
            target_time = None
            if choice == 3:
                try:
                    target_time = int(input("Enter target hold time in seconds (e.g. 30): "))
                    if target_time < 5:
                        print("Minimum 5 seconds. Using 30.")
                        target_time = 30
                except ValueError:
                    print("Invalid input. Using 30 seconds.")
                    target_time = 30
            exercise, name = get_exercise(choice, target_time=target_time)
            if exercise:
                return exercise, name, target_time
            print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")


def prompt_session_metadata(exercise_name):
    """
    Prompt for the training-data session metadata (Phase 1 export layer).
    Every prompt accepts a blank Enter to take its default, so recording
    sessions stay fast.

    Returns:
        Tuple of (subject_id, camera_view, form_intent).
    """
    subject_id = input("Subject ID (blank for 'unknown'): ").strip() or "unknown"

    default_view = "diagonal" if exercise_name in DIAGONAL_VIEW_EXERCISES else "side"
    camera_view = input(
        f"Camera view [side/front/diagonal] (default: {default_view}): "
    ).strip().lower() or default_view
    if camera_view not in VALID_CAMERA_VIEWS:
        print(f"Unrecognised view. Using '{default_view}'.")
        camera_view = default_view

    form_intent = input(
        "Form intent [good/bad/mixed] (default: mixed): "
    ).strip().lower() or "mixed"
    if form_intent not in VALID_FORM_INTENTS:
        print("Unrecognised intent. Using 'mixed'.")
        form_intent = "mixed"

    return subject_id, camera_view, form_intent


def main():
    exercise, exercise_name, target_time = show_exercise_menu_and_select()
    subject_id, camera_view, form_intent = prompt_session_metadata(exercise_name)

    pose, mp_pose = create_pose_detector()

    voice = VoiceEngine()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        voice.stop()
        return

    print(f"\n{exercise_name} tracking started. Press 'q' to quit.", flush=True)

    window_name = f"NeuralStride AI - {exercise_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    cv2.resizeWindow(window_name, screen_width - 100, screen_height - 120)

    mp_drawing = mp.solutions.drawing_utils

    feedback_display = FeedbackDisplay()
    session = SessionTracker()
    landmark_exporter = LandmarkExporter(
        output_dir="Landmarks",
        exercise=exercise_name,
        subject_id=subject_id,
        camera_view=camera_view,
        form_intent=form_intent,
        target_time=target_time,
        mirrored=True,
        source="webcam",
    )
    rep_tracker = RepWindowTracker(idle_state=exercise.get_idle_state())
    ws_broadcaster = WebSocketBroadcaster(host="localhost", port=8765)

    position_reminder_spoken = False

    frame_count = 0
    fps_display = 0
    fps_timer = time.time()

    is_walking = False
    is_ready = False

    completed_rep_events = 0
    aborted_rep_events = 0

    def log_rep_event_if_any(rep_event):
        nonlocal completed_rep_events, aborted_rep_events
        if rep_event is None:
            return
        rep_event["exercise"] = exercise_name
        landmark_exporter.log_rep_event(rep_event)
        if rep_event["completed"]:
            completed_rep_events += 1
        else:
            aborted_rep_events += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        displayed_feedback = "None"
        exercise_state = "INVALID"
        posture_labels = []
        issue_codes = []
        body_angle_smoothed = 0
        rep_count = 0
        left_angle = 0
        right_angle = 0
        avg_angle = 0

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_display = frame_count
            frame_count = 0
            fps_timer = now

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Computed once per frame regardless of ready/tracking state —
            # this is the only extract_features() call per frame (30fps budget).
            features, feature_quality = extract_features(landmarks, mp_pose)

            frame_number = landmark_exporter.frame_count + 1
            frame_timestamp = now - session.start_time
            rep_event = None

            try:
                ready, not_ready_msg = exercise.check_ready(landmarks, mp_pose)
                already_tracking = exercise.is_tracking()

                if ready or already_tracking:
                    is_ready = True

                    result = exercise.update(landmarks, mp_pose, w, h, now)

                    exercise_state = result.state
                    posture_labels = result.form_labels
                    issue_codes = result.form_issue_codes
                    body_angle_smoothed = result.back_angle
                    rep_count = result.rep_count
                    left_angle = result.left_angle
                    right_angle = result.right_angle
                    avg_angle = result.avg_angle
                    is_walking = result.is_walking

                    if result.voice_message:
                        voice.speak(result.voice_message)

                    if result.rep_completed:
                        session.log_rep(
                            rep_count,
                            had_form_issues=result.rep_had_issues,
                            issues=result.rep_issues,
                            avg_knee_angle=result.rep_avg_angle
                        )

                    displayed_feedback = feedback_display.update(result.feedback, now)

                    rep_event = rep_tracker.update(frame_number, frame_timestamp, result)

                else:
                    is_ready = False
                    is_walking = False
                    exercise.reset()
                    rep_event = rep_tracker.abort(frame_number, frame_timestamp)

                    displayed_feedback = feedback_display.update(not_ready_msg, now)

                    if not position_reminder_spoken:
                        voice.speak(not_ready_msg)
                        position_reminder_spoken = True

            except Exception as e:
                print("Error:", e, flush=True)
                displayed_feedback = feedback_display.update("Error", now)

            log_rep_event_if_any(rep_event)

            if rep_tracker.current_rep is not None:
                rep_id_for_frame = rep_tracker.current_rep["rep_id"]
            elif rep_event is not None:
                rep_id_for_frame = rep_event["rep_id"]
            else:
                rep_id_for_frame = None

            json_string = landmark_exporter.log_frame(
                landmarks, frame_timestamp,
                state=exercise_state,
                rep_id=rep_id_for_frame,
                active_issues=issue_codes,
                is_walking=is_walking,
                features=features,
                feature_quality=feature_quality,
            )
            if json_string:
                ws_broadcaster.update(json_string)

        else:
            exercise_state = "INVALID"
            is_walking = False
            is_ready = False
            displayed_feedback = feedback_display.update("None", now)
            log_rep_event_if_any(rep_tracker.abort(landmark_exporter.frame_count, now - session.start_time))

        draw_hud(
            frame, rep_count, exercise_state, displayed_feedback,
            left_angle, right_angle, avg_angle,
            body_angle_smoothed,
            is_walking, is_ready,
            posture_labels,
            left_angle, right_angle,
            fps_display, session.get_elapsed_str(),
            exercise_name=exercise_name
        )

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    session.end_session()
    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    voice.stop()

    log_rep_event_if_any(rep_tracker.abort(landmark_exporter.frame_count, time.time() - session.start_time))

    lm_path, lm_frames = landmark_exporter.close()
    if lm_path:
        print(f"Landmark data saved: {lm_path} ({lm_frames} frames)", flush=True)
        print(
            f"Rep events written: {completed_rep_events} completed, "
            f"{aborted_rep_events} aborted",
            flush=True
        )

    ws_broadcaster.close()

    summary = session.get_summary()
    print("\n=== WORKOUT SUMMARY ===", flush=True)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}", flush=True)
    print(f"Duration: {summary['duration']}", flush=True)
    print(f"Total Reps: {summary['total_reps']}", flush=True)
    print(f"Clean Reps: {summary['clean_reps']}", flush=True)
    print(f"Form Issue Reps: {summary['form_issue_reps']}", flush=True)
    print(f"Form Accuracy: {summary['clean_percentage']}%", flush=True)
    print(f"Form Issues: {summary['form_issues']}", flush=True)
    print("======================\n", flush=True)

    csv_path = session.save_to_csv(output_dir="sessions")
    print(f"Session saved to: {csv_path}", flush=True)

    summary_window = "NeuralStride AI - Session Summary"
    cv2.namedWindow(summary_window, cv2.WINDOW_NORMAL)
    summary_img = draw_workout_summary(summary, datetime.now().strftime('%Y-%m-%d %I:%M %p'))
    cv2.imshow(summary_window, summary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
