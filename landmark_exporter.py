"""
================================================================================
NeuralStride AI - Landmark Coordinate Exporter (Schema v4 - Training Export)
================================================================================

Exports MediaPipe Pose landmark data to a JSONL file in near real-time. Each
line in the file is a complete JSON object — one line per frame — containing
all 33 body landmarks with their x, y, z coordinates and visibility.

Schema v2 added machine-learning-ready labelled data alongside the original
visualisation payload: every frame tagged with the exercise state, the id of
the rep in progress, active rule-engine issue codes and a per-frame
biomechanical feature vector; every rep additionally emitted as its own "rep"
line once its final label is known.

Schema v3 changes only where "active_issues" comes from. v2 reconstructed it
by reverse-mapping ExerciseResult.form_labels display strings back to codes —
a workaround that silently produced [] if a display string ever changed or
collided. v3 reads ExerciseResult.form_issue_codes directly (a field the
detectors now populate as raw codes, parallel to form_labels) and validates
every code against session_utils.VALID_ISSUE_CODES, raising immediately on
anything unrecognised instead of dropping it silently. Files written before
this change are schema_version 2 and their active_issues should be treated as
lower-confidence (see inspect_jsonl.py's v2/v3 handling).

PURPOSE:
    v1: captured raw pose landmark data for a Three.js 3D skeleton viewer.
    v2/v3: the above, PLUS everything a CNN-LSTM training pipeline needs to
    join frames to reps and reps to labels, without changing anything the
    v1 consumer (the Three.js client / WebSocket viewer) already relies on.
    Raw landmarks are never removed — new fields are only ever added.

HOW IT WORKS:
    1. When a workout starts, a new .jsonl file is created in the Landmarks/ folder
    2. The first line written is a metadata header (landmark names, connections,
       feature names, and the session block — exercise/subject/camera/intent)
    3. On every frame where MediaPipe detects a person, a frame line is written
    4. Data is flushed to disk every 15 frames (~0.5 seconds) for live readability
    5. Rep lines are written separately, AFTER the frames they describe — a
       rep's final label (clean/issues/aborted) is only known once it closes.
       Consumers must do a two-pass read: collect frame lines first, then use
       each rep line's start_frame/end_frame to slice the corresponding frames.
    6. When the workout ends, the file is flushed one final time and closed
    7. The file can be read by another program WHILE the workout is still running

WHY JSONL (not CSV):
    - CSV would need 130+ columns — unreadable
    - JSONL keeps each frame/rep as one structured object — easy to parse
    - Each line is atomic (complete JSON) so partial reads won't corrupt data
    - Still plain text — can be opened in any text editor to inspect

METADATA LINE FORMAT (line 1):
    {
        "type": "metadata",
        "format": "NeuralStride Landmark Export v4",
        "schema_version": 4,
        "landmark_count": 33,
        "landmark_names": { "0": "nose", ... },
        "connections": [ [0,1], ... ],
        "coordinate_system": { ... },
        "feature_names": ["left_knee_angle", ...],
        "feature_count": 22,
        "mirrored": true,
        "session": {
            "session_id": "20260724_143000_s01",
            "exercise": "Squat",
            "subject_id": "s01",
            "camera_view": "side",
            "form_intent": "mixed",
            "target_time": null,
            "source": "webcam"
        },
        "created": "2026-07-24 02:30:00 PM"
    }

FRAME LINE FORMAT:
    {
        "type": "frame",
        "frame": 1,
        "timestamp": 0.0334,
        "landmarks": { "0": {"x":..,"y":..,"z":..,"visibility":..}, ... },
        "state": "DOWN",
        "rep_id": 3,
        "active_issues": ["BACK_LEAN"],
        "is_walking": false,
        "features": [172.4, 171.9, ...],
        "feature_quality": 0.94
    }
    - "rep_id" is the id of the rep currently in progress, or null between reps.
    - "active_issues" holds raw issue codes (BACK_LEAN, KNEE_FORWARD, ...),
      never the human-readable display strings — the model trains on codes.
      Sourced directly from ExerciseResult.form_issue_codes (a field parallel
      to form_labels, populated by the detectors themselves). Every code is
      validated against session_utils.VALID_ISSUE_CODES in log_frame() —
      an unrecognised code raises immediately rather than being dropped.

REP LINE FORMAT (written after the frames it describes; NEVER broadcast):
    {
        "type": "rep",
        "rep_id": 3,
        "start_frame": 410,
        "end_frame": 468,
        "start_timestamp": 13.6,
        "end_timestamp": 15.6,
        "duration": 2.0,
        "completed": true,
        "aborted": false,
        "clean": false,
        "issues": ["BACK_LEAN", "SHALLOW"],
        "avg_angle": 78.4,
        "exercise": "Squat"
    }

LANDMARK INDICES (MediaPipe Pose, 0-32):
    0  = nose               16 = right_wrist       27 = left_ankle
    1  = left_eye_inner     17 = left_pinky         28 = right_ankle
    2  = left_eye           18 = right_pinky        29 = left_heel
    3  = left_eye_outer     19 = left_index          30 = right_heel
    4  = right_eye_inner    20 = right_index         31 = left_foot_index
    5  = right_eye          21 = left_thumb         32 = right_foot_index
    6  = right_eye_outer    22 = right_thumb
    7  = left_ear           23 = left_hip
    8  = right_ear          24 = right_hip
    9  = mouth_left         25 = left_knee
    10 = mouth_right        26 = right_knee
    11 = left_shoulder
    12 = right_shoulder
    13 = left_elbow
    14 = right_elbow
    15 = left_wrist

SKELETON CONNECTIONS:
    Draw a line (or cylinder in 3D) between each pair to form the skeleton:
    Face:        0-1, 1-2, 2-3, 3-7, 0-4, 4-5, 5-6, 6-8, 9-10
    Torso:       11-12 (shoulders), 11-23, 12-24 (shoulder to hip), 23-24 (hips)
    Left arm:    11-13 (shoulder-elbow), 13-15 (elbow-wrist)
    Right arm:   12-14 (shoulder-elbow), 14-16 (elbow-wrist)
    Left leg:    23-25 (hip-knee), 25-27 (knee-ankle), 27-29 (ankle-heel), 29-31 (heel-toe)
    Right leg:   24-26 (hip-knee), 26-28 (knee-ankle), 28-30 (ankle-heel), 30-32 (heel-toe)
    Hands:       15-17, 15-19, 15-21 (left), 16-18, 16-20, 16-22 (right)

THREE.JS / WEBSOCKET CONSUMERS (unchanged from v1):
    - Read line 1 -> landmark_names / connections / coordinate_system as before.
    - Frame lines gained new fields but keep "frame", "timestamp", "landmarks"
      byte-for-byte — existing viewer code that only reads those three keeps
      working unmodified.
    - Rep lines are a new "type": "rep" — the WebSocket broadcaster in
      main.py never receives them, only the JSONL file does.
    - ALWAYS filter by visibility > 0.1 before drawing a landmark.

OUTPUT:
    Files are saved to: Landmarks/landmarks_{exercise}_{subject_id}_{DD-MM-YYYY_HH-MM-SSAM}.jsonl

INTEGRATION:
    Called from main.py:
        - LandmarkExporter created before the main loop starts, with the
          session metadata prompted from the user
        - log_frame() called once per frame when person is detected
        - log_rep_event() called whenever rep_window.RepWindowTracker
          returns a completed/aborted rep event
        - close() called after workout ends

DEPENDENCIES:
    json, os, re, time, datetime — all Python standard library
    feature_utils — for FEATURE_NAMES / FEATURE_COUNT in the metadata header
================================================================================
"""

import json
import os
import re
import time
from datetime import datetime

from feature_utils import FEATURE_NAMES, FEATURE_COUNT
from session_utils import VALID_ISSUE_CODES


# ---------------------------------------------------------------------------
# LANDMARK NAMES
# ---------------------------------------------------------------------------
# Standard MediaPipe Pose landmark names indexed by landmark ID (0-32).
# The Three.js person uses these to label the 3D skeleton joints.
# Keys are strings because JSON object keys must be strings.
LANDMARK_NAMES = {
    "0": "nose",
    "1": "left_eye_inner",
    "2": "left_eye",
    "3": "left_eye_outer",
    "4": "right_eye_inner",
    "5": "right_eye",
    "6": "right_eye_outer",
    "7": "left_ear",
    "8": "right_ear",
    "9": "mouth_left",
    "10": "mouth_right",
    "11": "left_shoulder",
    "12": "right_shoulder",
    "13": "left_elbow",
    "14": "right_elbow",
    "15": "left_wrist",
    "16": "right_wrist",
    "17": "left_pinky",
    "18": "right_pinky",
    "19": "left_index",
    "20": "right_index",
    "21": "left_thumb",
    "22": "right_thumb",
    "23": "left_hip",
    "24": "right_hip",
    "25": "left_knee",
    "26": "right_knee",
    "27": "left_ankle",
    "28": "right_ankle",
    "29": "left_heel",
    "30": "right_heel",
    "31": "left_foot_index",
    "32": "right_foot_index"
}

# ---------------------------------------------------------------------------
# SKELETON CONNECTIONS
# ---------------------------------------------------------------------------
# Pairs of landmark indices that form the skeleton "bones".
# Draw a line (or cylinder in 3D) between each [start, end] pair.
# Same as MediaPipe's POSE_CONNECTIONS — included here so the Three.js person
# doesn't need to import MediaPipe or look up the connection list.
POSE_CONNECTIONS = [
    # Face
    [0, 1], [1, 2], [2, 3], [3, 7],
    [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10],
    # Shoulders (connects left and right shoulder)
    [11, 12],
    # Left arm: shoulder -> elbow -> wrist
    [11, 13], [13, 15],
    # Right arm: shoulder -> elbow -> wrist
    [12, 14], [14, 16],
    # Torso: shoulders down to hips
    [11, 23], [12, 24],
    # Hips (connects left and right hip)
    [23, 24],
    # Left leg: hip -> knee -> ankle
    [23, 25], [25, 27],
    # Right leg: hip -> knee -> ankle
    [24, 26], [26, 28],
    # Left foot: ankle -> heel -> toe
    [27, 29], [29, 31],
    # Right foot: ankle -> heel -> toe
    [28, 30], [30, 32],
    # Left hand: wrist -> pinky, index, thumb
    [15, 17], [15, 19], [15, 21],
    # Right hand: wrist -> pinky, index, thumb
    [16, 18], [16, 20], [16, 22]
]


def _sanitize_for_filename(value):
    """Replace filesystem-unsafe characters so subject_id/exercise can't break paths."""
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(value)).strip('_') or "unknown"


class LandmarkExporter:
    """
    Captures MediaPipe pose landmarks frame-by-frame and writes them to a
    JSONL file, tagged with rule-engine labels and biomechanical features for
    downstream CNN-LSTM training. The file updates on disk in near real-time
    so another program (e.g., a Three.js 3D viewer) can read it live.

    Usage (from main.py):
        exporter = LandmarkExporter(
            output_dir="Landmarks", exercise="Squat", subject_id="s01",
            camera_view="side", form_intent="mixed", target_time=None,
            mirrored=True, source="webcam",
        )
        # ... inside the main loop ...
        exporter.log_frame(landmarks, timestamp, state=..., rep_id=..., ...)
        # ... whenever RepWindowTracker closes a rep ...
        exporter.log_rep_event(rep_event)
        # ... after workout ends ...
        filepath, frame_count = exporter.close()
    """

    def __init__(
        self,
        output_dir="Landmarks",
        flush_interval=15,
        exercise="Unknown",
        subject_id="unknown",
        camera_view="side",
        form_intent="mixed",
        target_time=None,
        mirrored=True,
        source="webcam",
    ):
        """
        Creates the output directory (if needed) and opens a new .jsonl file.
        Writes a metadata header as the first line containing landmark names,
        skeleton connections, feature names, and the session label block.

        Args:
            output_dir:      Folder where the .jsonl file will be saved.
            flush_interval:  How many frames to buffer before flushing to disk.
            exercise:        Exercise display name (e.g. "Squat") — goes into
                             the filename and session.exercise.
            subject_id:      Free-text subject identifier for the session block
                             and filename. Sanitised for filesystem safety.
            camera_view:     "side" / "front" / "diagonal".
            form_intent:     "good" / "bad" / "mixed" — what the subject was
                             told to do, a coarse sanity check on auto labels.
            target_time:     Plank-only target hold time in seconds, else None.
            mirrored:        True if the source frames are left-right flipped
                             (live webcam sessions always are — main.py calls
                             cv2.flip before MediaPipe runs).
            source:          "webcam" or "video".
        """
        self.output_dir = output_dir
        self.flush_interval = flush_interval
        self.frame_count = 0
        self.start_time = time.time()
        self.file = None
        self.filepath = None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        safe_exercise = _sanitize_for_filename(exercise)
        safe_subject = _sanitize_for_filename(subject_id)

        # Example: landmarks_Squat_s01_15-01-2026_02-30-00PM.jsonl
        timestamp_str = datetime.now().strftime("%d-%m-%Y_%I-%M-%S%p")
        filename = f"landmarks_{safe_exercise}_{safe_subject}_{timestamp_str}.jsonl"
        self.filepath = os.path.join(output_dir, filename)
        self.file = open(self.filepath, "w")

        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_subject}"

        # --- Write metadata header as line 1 ---
        header = {
            "type": "metadata",
            "format": "NeuralStride Landmark Export v4",
            "schema_version": 4,
            "landmark_count": 33,
            "landmark_names": LANDMARK_NAMES,
            "connections": POSE_CONNECTIONS,
            "coordinate_system": {
                "x": "normalized 0-1, left to right",
                "y": "normalized 0-1, top to bottom",
                "z": "depth, negative = closer to camera",
                "visibility": "0-1 confidence"
            },
            "feature_names": FEATURE_NAMES,
            "feature_count": FEATURE_COUNT,
            "mirrored": mirrored,
            "session": {
                "session_id": session_id,
                "exercise": exercise,
                "subject_id": subject_id,
                "camera_view": camera_view,
                "form_intent": form_intent,
                "target_time": target_time,
                "source": source,
            },
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.file.write(json.dumps(header) + "\n")
        self.file.flush()

        print(f"Landmark export started: {self.filepath}", flush=True)

    def log_frame(
        self,
        landmarks,
        timestamp=None,
        state=None,
        rep_id=None,
        active_issues=None,
        is_walking=False,
        features=None,
        feature_quality=None,
        feature_confidence=None,
    ):
        """
        Writes one frame's landmark + label + feature data as a single JSON line.

        Called from main.py once per frame when MediaPipe detects a person.

        Args:
            landmarks:      MediaPipe pose_landmarks.landmark list (33 items).
            timestamp:      Relative time in seconds from workout start.
            state:          Current exercise state string, or None/"INVALID".
            rep_id:         Id of the rep currently in progress, or None.
            active_issues:  List of raw issue codes active this frame, straight
                             from ExerciseResult.form_issue_codes. Every code
                             is validated against VALID_ISSUE_CODES — an
                             unrecognised code raises ValueError immediately
                             (label corruption must fail fast, not train a
                             model on silently-dropped or garbled labels).
            is_walking:     True if WalkingDetector flagged this frame.
            features:       22-value feature vector from extract_features(),
                             or None if it could not be computed this frame.
            feature_quality: Quality score from extract_features(), or None.
            feature_confidence: 22-value confidence array from extract_features(),
                             or None.

        Raises:
            ValueError: if active_issues contains a code not in
                        session_utils.VALID_ISSUE_CODES.
        """
        if self.file is None:
            return

        if timestamp is None:
            timestamp = time.time() - self.start_time

        active_issues = active_issues if active_issues else []
        for code in active_issues:
            if code not in VALID_ISSUE_CODES:
                raise ValueError(
                    f"Unknown issue code {code!r} in active_issues - a detector is "
                    f"emitting a code missing from session_utils.VALID_ISSUE_CODES. "
                    f"Fix the vocabulary before recording more data."
                )

        self.frame_count += 1

        frame_data = {
            "type": "frame",
            "frame": self.frame_count,
            "timestamp": round(timestamp, 4),
            "landmarks": {},
            "state": state,
            "rep_id": rep_id,
            "active_issues": active_issues,
            "is_walking": bool(is_walking),
            "features": [round(f, 4) for f in features] if features is not None else [],
            "feature_quality": round(feature_quality, 4) if feature_quality is not None else 0.0,
        }
        if feature_confidence is not None:
            frame_data["feature_confidence"] = [round(c, 4) for c in feature_confidence]

        for i, lm in enumerate(landmarks):
            frame_data["landmarks"][str(i)] = {
                "x": round(lm.x, 6),
                "y": round(lm.y, 6),
                "z": round(lm.z, 6),
                "visibility": round(lm.visibility, 4)
            }

        json_string = json.dumps(frame_data)
        self.file.write(json_string + "\n")

        if self.frame_count % self.flush_interval == 0:
            self.file.flush()

        # Return the JSON string so main.py can pass it to the WebSocket
        # broadcaster without having to serialize the data a second time.
        return json_string

    def log_rep_event(self, rep_event):
        """
        Writes one completed or aborted rep as a "type": "rep" JSON line.

        Called from main.py whenever rep_window.RepWindowTracker.update()/
        abort() returns a non-None event. Rep lines are written AFTER the
        frames they span — consumers must collect frames first, then use
        each rep's start_frame/end_frame to slice them (see module docstring).

        This method's output must NEVER be passed to the WebSocket
        broadcaster — only frame lines are broadcast live.

        Args:
            rep_event: dict as produced by RepWindowTracker, with an
                       "exercise" key added by main.py before calling this.
        """
        if self.file is None:
            return

        line = {"type": "rep"}
        line.update(rep_event)
        self.file.write(json.dumps(line) + "\n")
        self.file.flush()

    def close(self):
        """
        Flushes any remaining buffered data to disk and closes the file.

        Returns:
            Tuple of (filepath, frame_count).
        """
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None
            return self.filepath, self.frame_count
        return None, self.frame_count
