"""
================================================================================
NeuralStride AI - Squat Exercise Detector
================================================================================

Detects and analyzes squat exercises using MediaPipe Pose landmarks.
Extends BaseExercise from detectors/base.py.

ANGLE TRACKING:
    - Knee angle (hip -> knee -> ankle) using Law of Cosines
    - Average of left and right knee angles
    - 5-frame rolling average for smoothing (AngleSmoother)

STATE DETECTION:
    UP   : knee angle > 160 degrees (standing)
    MID  : knee angle 85-160 degrees (partial squat)
    DOWN : knee angle < 85 degrees (deep squat)
    State debouncing: new state must hold for 5 frames before transitioning

REP COUNTING:
    Counts a full rep: UP -> DOWN (hold 20 frames) -> UP
    Each rep logs: rep number, duration, knee angle, form issues, clean/not

FORM CHECKS (only during DOWN phase):
    - Back lean detection: warning at 30 degrees, severe at 45 degrees
      (shoulder-hip angle from vertical)
    - Knee-over-toe: knee x-offset vs foot_index landmark (threshold: 0.05)
      Uses x-offset only (not z-depth, which is unreliable from MediaPipe)
    - Hip drop: left/right hip y-position difference (threshold: 0.04)
    - Shallow squat: stuck in MID without reaching DOWN state

ANTI FALSE-POSITIVE:
    - WalkingDetector: ankle alternation + hip shift over 10 frames
    - Knee symmetry: both knees must bend within 40 degrees of each other
    - AngleSmoother: 5-frame rolling average to kill single-frame spikes
    - StateDebouncer: prevents rapid UP/MID/DOWN flickering

VOICE FEEDBACK:
    - "Good squat" on clean rep completion
    - "Rep counted, fix your [issue]" on rep with form issues
    - "Fix your [issue]" during DOWN phase if form problem detected
    - "Go lower" if stuck in MID state
    - "Walking detected, please stand still" if walking in frame
    - "Stand upright" if not in standing position
    - Cooldowns: general 2s, posture 3s, standing 3s

CONSTANTS:
    ASYMMETRY_THRESHOLD        = 40     degrees max diff between knees
    WALKING_ANKLE_CHANGE       = 0.03   normalized ankle movement
    WALKING_HIP_CHANGE         = 0.02   normalized hip movement
    WALKING_FRAMES_REQUIRED    = 10     frames to confirm walking
    MIN_DOWN_FRAMES            = 20     frames to confirm DOWN state
    TRANSITION_DEBOUNCE_FRAMES = 5      frames before state changes
    BACK_ANGLE_THRESHOLD       = 30     degrees for back lean warning
    BACK_ANGLE_WARNING         = 45     degrees for severe back lean
    KNEE_OVER_TOE_THRESHOLD    = 0.05   normalized x-offset
    HIP_DROP_THRESHOLD         = 0.04   normalized y-diff between hips
================================================================================
"""

import numpy as np
import time
import collections
import mediapipe as mp

from angle_utils import calculate_angle, AngleSmoother, SingleSmoother
from pose_utils import is_standing_posture
from detectors.base import BaseExercise, ExerciseResult

ASYMMETRY_THRESHOLD = 40
WALKING_ANKLE_CHANGE_THRESHOLD = 0.03
WALKING_HIP_CHANGE_THRESHOLD = 0.02
WALKING_FRAMES_REQUIRED = 10
MIN_DOWN_FRAMES = 20
MIN_MID_FRAMES = 10
TRANSITION_DEBOUNCE_FRAMES = 5
MID_HOLD_THRESHOLD = 2.0

BACK_ANGLE_THRESHOLD = 30
BACK_ANGLE_WARNING = 45
KNEE_OVER_TOE_THRESHOLD = 0.03
HIP_DROP_THRESHOLD = 0.04

GENERAL_COOLDOWN = 2.0
POSTURE_COOLDOWN = 3.0


class WalkingDetector:
    def __init__(self, frames_required=WALKING_FRAMES_REQUIRED):
        self.prev_left_ankle_y = None
        self.prev_right_ankle_y = None
        self.prev_avg_hip_y = None
        self.walk_vote_count = 0
        self.frames_required = frames_required

    def update(self, landmarks, frame_height):
        is_walking = False

        left_ankle_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y
        right_ankle_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y
        left_hip_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y
        avg_hip_y = (left_hip_y + right_hip_y) / 2

        if self.prev_left_ankle_y is not None:
            left_ankle_change = abs(left_ankle_y - self.prev_left_ankle_y)
            right_ankle_change = abs(right_ankle_y - self.prev_right_ankle_y)
            hip_change = abs(avg_hip_y - self.prev_avg_hip_y)

            ankle_alternating = (
                (left_ankle_change > WALKING_ANKLE_CHANGE_THRESHOLD and right_ankle_change < WALKING_ANKLE_CHANGE_THRESHOLD / 2)
                or
                (right_ankle_change > WALKING_ANKLE_CHANGE_THRESHOLD and left_ankle_change < WALKING_ANKLE_CHANGE_THRESHOLD / 2)
            )

            significant_body_shift = hip_change > WALKING_HIP_CHANGE_THRESHOLD

            if (ankle_alternating or significant_body_shift) and avg_hip_y > 0.45:
                self.walk_vote_count += 1
            else:
                self.walk_vote_count = max(0, self.walk_vote_count - 1)

            if self.walk_vote_count >= self.frames_required:
                is_walking = True

        self.prev_left_ankle_y = left_ankle_y
        self.prev_right_ankle_y = right_ankle_y
        self.prev_avg_hip_y = avg_hip_y

        return is_walking

    def reset(self):
        self.prev_left_ankle_y = None
        self.prev_right_ankle_y = None
        self.prev_avg_hip_y = None
        self.walk_vote_count = 0


class StateDebouncer:
    def __init__(self, debounce_frames=TRANSITION_DEBOUNCE_FRAMES):
        self.debounce_frames = debounce_frames
        self.candidate_state = None
        self.candidate_count = 0
        self.current_state = "UP"

    def update(self, raw_state):
        if raw_state == self.current_state:
            self.candidate_state = None
            self.candidate_count = 0
            return self.current_state

        if raw_state != self.candidate_state:
            self.candidate_state = raw_state
            self.candidate_count = 1
        else:
            self.candidate_count += 1

        if self.candidate_count >= self.debounce_frames:
            self.current_state = raw_state
            self.candidate_state = None
            self.candidate_count = 0

        return self.current_state

    def reset(self):
        self.candidate_state = None
        self.candidate_count = 0
        self.current_state = "UP"


def get_squat_state(avg_knee_angle):
    if avg_knee_angle > 160:
        return "UP"
    elif avg_knee_angle < 85:
        return "DOWN"
    else:
        return "MID"


def check_knee_symmetry(left_angle, right_angle):
    if ASYMMETRY_THRESHOLD == 0:
        return True
    return abs(left_angle - right_angle) < ASYMMETRY_THRESHOLD


def calculate_back_angle(shoulder_point, hip_point, vertical_ref=None):
    shoulder = np.array(shoulder_point)
    hip = np.array(hip_point)

    if vertical_ref is None:
        vertical = np.array([hip[0], hip[1] - 100])
    else:
        vertical = np.array(vertical_ref)

    torso_vec = shoulder - hip
    vertical_vec = vertical - hip

    denominator = np.linalg.norm(torso_vec) * np.linalg.norm(vertical_vec)
    if denominator == 0:
        return 0

    cosine_angle = np.dot(torso_vec, vertical_vec) / denominator
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def check_knee_over_toe(knee_landmark, toe_landmark):
    offset = knee_landmark.x - toe_landmark.x
    return abs(offset) > KNEE_OVER_TOE_THRESHOLD


def check_hip_drop(left_hip, right_hip):
    return abs(left_hip.y - right_hip.y) > HIP_DROP_THRESHOLD


def run_posture_checks(landmarks, mp_pose, w, h, squat_state):
    issues = []

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    avg_shoulder = [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2]
    avg_hip = [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]

    shoulder_px = [avg_shoulder[0] * w, avg_shoulder[1] * h]
    hip_px = [avg_hip[0] * w, avg_hip[1] * h]

    back_angle = calculate_back_angle(shoulder_px, hip_px)

    if back_angle > BACK_ANGLE_WARNING:
        issues.append("SEVERE_LEAN")
    elif back_angle > BACK_ANGLE_THRESHOLD:
        issues.append("BACK_LEAN")

    left_knee_over = check_knee_over_toe(left_knee, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])
    right_knee_over = check_knee_over_toe(right_knee, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])

    if left_knee_over or right_knee_over:
        issues.append("KNEE_FORWARD")

    hip_dropped = check_hip_drop(left_hip, right_hip)
    if hip_dropped:
        issues.append("HIP_DROP")

    if squat_state == "MID" and back_angle > BACK_ANGLE_THRESHOLD:
        pass
    elif squat_state == "MID":
        issues.append("SHALLOW")

    return issues, back_angle


def posture_issues_to_feedback(issues):
    if not issues:
        return None
    if "SEVERE_LEAN" in issues:
        return "Straighten your back"
    elif "BACK_LEAN" in issues:
        return "Keep your back straight"
    elif "KNEE_FORWARD" in issues:
        return "Don't let knees go past toes"
    elif "HIP_DROP" in issues:
        return "Keep your hips level"
    elif "SHALLOW" in issues:
        return "Go deeper"
    return None


def posture_issues_to_voice(issues):
    if not issues:
        return None
    if "SEVERE_LEAN" in issues:
        return "Straighten your back"
    elif "BACK_LEAN" in issues:
        return "Keep your back straight"
    elif "KNEE_FORWARD" in issues:
        return "Don't let your knees go past your toes"
    elif "HIP_DROP" in issues:
        return "Keep your hips level"
    elif "SHALLOW" in issues:
        return "Go deeper"
    return None


def posture_issues_to_display(issues):
    labels = []
    for issue in issues:
        if issue == "SEVERE_LEAN":
            labels.append("Back lean (severe)")
        elif issue == "BACK_LEAN":
            labels.append("Back leaning")
        elif issue == "KNEE_FORWARD":
            labels.append("Knees past toes")
        elif issue == "HIP_DROP":
            labels.append("Hips uneven")
        elif issue == "SHALLOW":
            labels.append("Too shallow")
    return labels


def posture_issue_to_fix_phrase(issues):
    if not issues:
        return None
    if "SEVERE_LEAN" in issues:
        return "straighten your back"
    if "BACK_LEAN" in issues:
        return "lean back slightly"
    if "KNEE_FORWARD" in issues:
        return "push knees back behind toes"
    if "HIP_DROP" in issues:
        return "keep your hips level"
    if "SHALLOW" in issues:
        return "squat deeper"
    return "form"


class SquatExercise(BaseExercise):
    def __init__(self):
        super().__init__()
        self.get_name = lambda: "Squat"

        self.angle_smoother = AngleSmoother()
        self.back_angle_smoother = SingleSmoother()
        self.walking_detector = WalkingDetector()
        self.state_debouncer = StateDebouncer()

        self.squat_started = False
        self.down_frame_count = 0
        self.mid_frame_count = 0
        self.mid_start_time = None
        self.rep_posture_issues = set()

        self.last_voice_time = 0
        self.walking_spoken = False
        self.go_lower_spoken = False
        self.last_posture_voice = None
        self.last_posture_voice_time = 0

        self.left_knee_angle = 0
        self.right_knee_angle = 0
        self.current_back_angle = 0
        self.smooth_avg = 0

    def check_ready(self, landmarks, mp_pose):
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        if is_standing_posture(ls, rs, lh, rh, lk, rk, la, ra):
            return True, ""
        return False, "Stand upright"

        self.last_voice_time = 0
        self.walking_spoken = False
        self.go_lower_spoken = False
        self.last_posture_voice = None
        self.last_posture_voice_time = 0

        self.left_knee_angle = 0
        self.right_knee_angle = 0
        self.current_back_angle = 0
        self.smooth_avg = 0

    def reset(self):
        super().reset()
        self.squat_started = False
        self.down_frame_count = 0
        self.mid_frame_count = 0
        self.mid_start_time = None
        self.rep_posture_issues = set()
        self.walking_spoken = False
        self.go_lower_spoken = False
        self.last_posture_voice = None
        self.last_posture_voice_time = 0
        self.left_knee_angle = 0
        self.right_knee_angle = 0
        self.current_back_angle = 0
        self.smooth_avg = 0
        self.angle_smoother.reset()
        self.back_angle_smoother.reset()
        self.walking_detector.reset()
        self.state_debouncer.reset()

    def update(self, landmarks, mp_pose, w, h, now):
        is_walking = self.walking_detector.update(landmarks, h)

        if is_walking:
            self._reset_tracking()
            voice_msg = None
            if (not self.walking_spoken) and (now - self.last_voice_time >= GENERAL_COOLDOWN):
                voice_msg = "Walking detected, please stand still"
                self.last_voice_time = now
                self.walking_spoken = True
            return ExerciseResult(
                state="INVALID",
                feedback="Walking detected",
                is_walking=True,
                voice_message=voice_msg,
                rep_count=self.rep_count,
                back_angle=self.current_back_angle,
                left_angle=self.left_knee_angle,
                right_angle=self.right_knee_angle,
                avg_angle=self.smooth_avg
            )

        self.walking_spoken = False

        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee_lm = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        left_hip_point = [left_hip.x * w, left_hip.y * h]
        left_knee_point = [left_knee_lm.x * w, left_knee_lm.y * h]
        left_ankle_point = [left_ankle.x * w, left_ankle.y * h]

        self.left_knee_angle = calculate_angle(left_hip_point, left_knee_point, left_ankle_point)

        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee_lm = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        right_hip_point = [right_hip.x * w, right_hip.y * h]
        right_knee_point = [right_knee_lm.x * w, right_knee_lm.y * h]
        right_ankle_point = [right_ankle.x * w, right_ankle.y * h]

        self.right_knee_angle = calculate_angle(right_hip_point, right_knee_point, right_ankle_point)

        if not check_knee_symmetry(self.left_knee_angle, self.right_knee_angle):
            self._reset_tracking()
            return ExerciseResult(
                state="INVALID",
                feedback="Keep both legs even",
                form_labels=["ASYMMETRIC"],
                form_issue_codes=["ASYMMETRIC"],
                rep_count=self.rep_count,
                back_angle=self.current_back_angle,
                left_angle=self.left_knee_angle,
                right_angle=self.right_knee_angle,
                avg_angle=self.smooth_avg
            )

        self.angle_smoother.update(self.left_knee_angle, self.right_knee_angle)
        self.smooth_avg = self.angle_smoother.get_smoothed_avg()

        raw_state = get_squat_state(self.smooth_avg)
        squat_state = self.state_debouncer.update(raw_state)

        posture_issues, self.current_back_angle = run_posture_checks(
            landmarks, mp_pose, w, h, squat_state
        )
        self.back_angle_smoother.update(self.current_back_angle)
        posture_labels = posture_issues_to_display(posture_issues)

        if squat_state == "DOWN":
            self.down_frame_count += 1
            self.mid_frame_count = 0
        elif squat_state == "MID":
            self.mid_frame_count += 1
            if self.down_frame_count < MIN_DOWN_FRAMES:
                self.down_frame_count = max(0, self.down_frame_count - 1)
        else:
            self.down_frame_count = max(0, self.down_frame_count - 1)
            self.mid_frame_count = 0

        feedback = "None"
        voice_msg = None
        rep_completed = False
        rep_had_issues = False
        rep_issues = []
        rep_avg_angle = 0

        if self.down_frame_count >= MIN_DOWN_FRAMES:
            self.squat_started = True

            if not posture_issues:
                feedback = "Good squat"
            else:
                fix_phrase = posture_issue_to_fix_phrase(posture_issues)
                feedback = f"Fix your {fix_phrase}"
                self.go_lower_spoken = False

            if squat_state == "DOWN" and posture_issues:
                self.rep_posture_issues.update(posture_issues)

        elif squat_state == "MID" and self.down_frame_count < MIN_DOWN_FRAMES:
            posture_fb = posture_issues_to_feedback(posture_issues)

            if posture_fb:
                feedback = posture_fb
            else:
                feedback = "Go lower"

            posture_voice = posture_issues_to_voice(posture_issues)

            if self.mid_start_time is None:
                self.mid_start_time = now

            if now - self.mid_start_time >= MID_HOLD_THRESHOLD:
                if posture_voice and posture_voice != self.last_posture_voice:
                    if now - self.last_posture_voice_time >= POSTURE_COOLDOWN:
                        voice_msg = posture_voice
                        self.last_voice_time = now
                        self.last_posture_voice = posture_voice
                        self.last_posture_voice_time = now
                elif not posture_voice:
                    if (not self.go_lower_spoken) and (now - self.last_voice_time >= GENERAL_COOLDOWN):
                        voice_msg = "Go lower"
                        self.last_voice_time = now
                        self.go_lower_spoken = True

            self.last_posture_voice = None
        else:
            self.mid_start_time = None
            self.go_lower_spoken = False
            self.last_posture_voice = None

        if self.squat_started and squat_state == "UP":
            self.rep_count += 1
            self.squat_started = False
            self.down_frame_count = 0
            self.mid_frame_count = 0
            self.mid_start_time = None
            self.go_lower_spoken = False
            self.last_posture_voice = None

            rep_completed = True
            rep_avg_angle = self.smooth_avg

            if self.rep_posture_issues:
                rep_had_issues = True
                rep_issues = list(self.rep_posture_issues)
                fix_phrase = posture_issue_to_fix_phrase(list(self.rep_posture_issues))
                feedback = "Rep counted"
                voice_msg = f"Rep counted, {fix_phrase}"
            else:
                feedback = "Good squat"
                voice_msg = "Good squat"

            self.rep_posture_issues = set()

        if squat_state == "UP" and not self.squat_started:
            feedback = "None"
            self.mid_start_time = None
            self.go_lower_spoken = False
            self.last_posture_voice = None

        return ExerciseResult(
            state=squat_state,
            feedback=feedback,
            form_labels=posture_labels,
            form_issue_codes=posture_issues,
            voice_message=voice_msg,
            rep_count=self.rep_count,
            back_angle=self.back_angle_smoother.get_smoothed(),
            rep_completed=rep_completed,
            rep_had_issues=rep_had_issues,
            rep_issues=rep_issues,
            rep_avg_angle=rep_avg_angle,
            left_angle=self.left_knee_angle,
            right_angle=self.right_knee_angle,
            avg_angle=self.smooth_avg
        )

    def _reset_tracking(self):
        self.down_frame_count = 0
        self.mid_frame_count = 0
        self.squat_started = False
        self.mid_start_time = None
        self.go_lower_spoken = False
        self.state_debouncer.reset()
        self.angle_smoother.reset()
        self.back_angle_smoother.reset()
        self.last_posture_voice = None
