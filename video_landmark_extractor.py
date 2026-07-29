"""
================================================================================
NeuralStride AI - Video Landmark Extractor
================================================================================

Extracts MediaPipe Pose landmarks from an uploaded video file and saves them
to a JSONL file in the same format as the live workout landmark export.

PURPOSE:
    Allows a user to upload a pre-recorded exercise video and extract raw
    pose landmark data from every frame. The resulting JSONL file can be fed
    into a machine learning model for supervised learning / training.

MIRROR CONSISTENCY (IMPORTANT — a real data-consistency bug, fixed here):
    main.py applies cv2.flip(frame, 1) before running MediaPipe on live
    webcam frames; this script historically did not. A landmark labelled
    "left_knee" therefore referred to physically OPPOSITE legs in the two
    data sources — mixing them into one training set silently corrupts every
    left/right and symmetry feature. This script now prompts to mirror
    frames (default: on) so video-extracted handedness matches live
    sessions, and always records the actual choice in metadata.mirrored.

UNLABELLED DATA:
    The rule-engine detectors do not run in this script — there is no
    exercise state machine here, only MediaPipe. Every frame is written with
    state: null, rep_id: null, active_issues: [], and NO rep events are ever
    written. session.source is always "video" and session.form_intent is
    always "unlabelled" — this file is only usable for pretraining or manual
    labelling, never for supervised rep classification as-is.

HOW IT WORKS:
    1. User runs this script
    2. A file dialog opens — user selects a video file (.mp4, .avi, .mov, .mkv)
    3. User is prompted for subject_id, camera_view, and whether to mirror
    4. Script reads the video frame-by-frame with OpenCV, mirroring if chosen
    5. Each frame is processed by MediaPipe Pose (same config as live tracking)
    6. Frames where a person is detected have their 33 landmarks, plus the
       schema v2 feature vector, written to JSONL
    7. Console shows real-time progress (frame count, percentage, ETA)
    8. On completion, JSONL is saved to Uploaded Video Landmarks/ folder
    9. Output filename matches the uploaded video name (e.g. squat_video.mp4 -> squat_video.jsonl)

OUTPUT FORMAT:
    Same JSONL schema v2 format as LandmarkExporter (see landmark_exporter.py
    docstring) — metadata header + frame lines, but with state/rep_id/
    active_issues always null/null/[] and no rep lines (see UNLABELLED DATA).

    Added metadata fields for uploaded videos (alongside the standard
    schema v2 "session" block):
        "source_video": "squat_video.mp4"
        "source_fps": 30.0
        "source_resolution": [1920, 1080]

RUNNING:
    python video_landmark_extractor.py

DEPENDENCIES:
    pip install opencv-python mediapipe==0.10.14 numpy
================================================================================
"""

import os
import json
import time
from datetime import datetime

import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

from landmark_exporter import LANDMARK_NAMES, POSE_CONNECTIONS
from feature_utils import extract_features, FEATURE_NAMES, FEATURE_COUNT


def prompt_mirror_choice():
    """Ask whether to mirror frames left-right. Defaults to mirroring ON
    so video-extracted handedness matches live webcam sessions (see the
    MIRROR CONSISTENCY note in the module docstring)."""
    answer = input("Mirror frames left-right to match live webcam sessions? [Y/n]: ").strip().lower()
    return answer not in ("n", "no")


def prompt_video_session_metadata():
    """Prompt for the subject_id / camera_view session fields. Blank Enter
    takes the default, matching main.py's recording-session prompts."""
    subject_id = input("Subject ID (blank for 'unknown'): ").strip() or "unknown"
    camera_view = input("Camera view [side/front/diagonal] (default: side): ").strip().lower() or "side"
    if camera_view not in ("side", "front", "diagonal"):
        print("Unrecognised view. Using 'side'.")
        camera_view = "side"
    return subject_id, camera_view


def pick_video_file():
    """Open a file dialog and return the selected video path, or None if cancelled."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    filepath = filedialog.askopenfilename(
        title="Select Exercise Video",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()
    return filepath if filepath else None


def create_jsonl_writer(output_dir, video_filename, fps, resolution, mirrored, subject_id, camera_view):
    """Create the output directory, open the JSONL file, and write the schema v2 metadata header."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    output_path = os.path.join(output_dir, f"{base_name}.jsonl")
    file_handle = open(output_path, "w", encoding="utf-8")

    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{subject_id}"

    header = {
        "type": "metadata",
        "format": "NeuralStride Landmark Export v2",
        "schema_version": 2,
        "landmark_count": 33,
        "landmark_names": LANDMARK_NAMES,
        "connections": POSE_CONNECTIONS,
        "coordinate_system": {
            "x": "normalized 0-1, left to right",
            "y": "normalized 0-1, top to bottom",
            "z": "depth, negative = closer to camera",
            "visibility": "0-1 confidence",
        },
        "feature_names": FEATURE_NAMES,
        "feature_count": FEATURE_COUNT,
        "mirrored": mirrored,
        "session": {
            "session_id": session_id,
            "exercise": base_name,
            "subject_id": subject_id,
            "camera_view": camera_view,
            "form_intent": "unlabelled",
            "target_time": None,
            "source": "video",
        },
        "source_video": os.path.basename(video_filename),
        "source_fps": round(fps, 2),
        "source_resolution": list(resolution),
        "created": datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"),
    }

    file_handle.write(json.dumps(header) + "\n")
    file_handle.flush()

    return file_handle, output_path


def format_eta(seconds):
    """Format seconds into M:SS string."""
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}:{secs:02d}"


def extract_landmarks_from_video(video_path, mirror, subject_id, camera_view):
    """Process a video file and extract landmarks to JSONL."""
    print(f"\nOpening video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    print(f"Video info: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    duration_seconds = total_frames / fps
    print(f"Duration: ~{format_eta(duration_seconds)}")

    writer, output_path = create_jsonl_writer(
        output_dir="Uploaded Video Landmarks",
        video_filename=video_path,
        fps=fps,
        resolution=(width, height),
        mirrored=mirror,
        subject_id=subject_id,
        camera_view=camera_view,
    )

    print(f"Output will be saved to: {output_path}\n")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_index = 0
    saved_frames = 0
    frame_buffer = []
    flush_interval = 50
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        if mirror:
            frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            timestamp = frame_index / fps

            features, feature_quality, feature_confidence = extract_features(landmarks, mp_pose, camera_view=camera_view)

            frame_data = {
                "type": "frame",
                "frame": frame_index,
                "timestamp": round(timestamp, 4),
                "landmarks": {},
                "state": None,
                "rep_id": None,
                "active_issues": [],
                "is_walking": False,
                "features": [round(f, 4) for f in features],
                "feature_quality": round(feature_quality, 4),
                "feature_confidence": [round(c, 4) for c in feature_confidence],
            }

            for i, lm in enumerate(landmarks):
                frame_data["landmarks"][str(i)] = {
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6),
                    "visibility": round(lm.visibility, 4),
                }

            frame_buffer.append(json.dumps(frame_data) + "\n")
            saved_frames += 1

            if len(frame_buffer) >= flush_interval:
                writer.write("".join(frame_buffer))
                writer.flush()
                frame_buffer.clear()

        if frame_index % 10 == 0 or frame_index == total_frames:
            elapsed = time.time() - start_time
            progress = frame_index / total_frames * 100
            fps_actual = frame_index / elapsed if elapsed > 0 else 0

            if frame_index < total_frames and fps_actual > 0:
                eta_seconds = (total_frames - frame_index) / fps_actual
                eta_str = format_eta(eta_seconds)
            else:
                eta_str = "done"

            print(
                f"\rFrame {frame_index}/{total_frames} "
                f"({progress:5.1f}%) | "
                f"Speed: {fps_actual:.1f} fps | "
                f"ETA: {eta_str} | "
                f"Saved: {saved_frames}",
                end="",
                flush=True,
            )

    if frame_buffer:
        writer.write("".join(frame_buffer))
        writer.flush()
        frame_buffer.clear()

    pose.close()
    cap.release()
    writer.close()

    elapsed = time.time() - start_time

    print(f"\n\n{'='*50}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"Video:       {os.path.basename(video_path)}")
    print(f"Frames:      {total_frames} total, {saved_frames} with person detected")
    print(f"Time taken:  {format_eta(elapsed)}")
    print(f"Output:      {output_path}")
    print(f"{'='*50}\n")


def main():
    print("=" * 50)
    print("NeuralStride AI - Video Landmark Extractor")
    print("=" * 50)
    print("Select a video file to extract pose landmarks.\n")

    video_path = pick_video_file()

    if not video_path:
        print("No file selected. Exiting.")
        return

    if not os.path.isfile(video_path):
        print(f"Error: File not found: {video_path}")
        return

    mirror = prompt_mirror_choice()
    subject_id, camera_view = prompt_video_session_metadata()

    extract_landmarks_from_video(video_path, mirror, subject_id, camera_view)


if __name__ == "__main__":
    main()
