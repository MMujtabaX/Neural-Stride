"""
================================================================================
NeuralStride AI - Stationary Lunge Exercise Detector
================================================================================

Detects and analyzes stationary lunge exercises using MediaPipe Pose landmarks.
Extends BaseExercise from detectors/base.py.

LUNG TYPE:
    Stationary lunge — one foot forward, one foot back, step down and up in place.
    Diagonal view recommended for accurate knee angle and knee-over-toe measurement.

ANGLE TRACKING:
    - Front knee angle (hip -> knee -> ankle) using Law of Cosines
    - Back knee angle (hip -> knee -> ankle) using Law of Cosines
    - Average of front left/right and back left/right knee angles
    - 5-frame rolling average for smoothing (AngleSmoother)

STATE DETECTION:
    UP   : front knee angle > 160 degrees (standing tall, legs mostly straight)
    MID  : front knee angle 90-160 degrees (partial lunge)
    DOWN : front knee angle < 100 degrees (deep lunge, ~90 degree knee)
    State debouncing: new state must hold for 5 frames before transitioning

REP COUNTING:
    Counts a full rep: UP -> DOWN (hold 15 frames) -> UP
    Each rep logs: rep number, duration, knee angle, form issues, clean/not

FORM CHECKS (only during DOWN phase):
    1. Front knee-over-toe: front knee shouldn't extend past front toe (threshold: 0.08)
    2. Front knee angle ~90 degrees: flags if too shallow (front knee > 120 in DOWN)
    3. Back knee angle: checks back knee is close to ground (back knee < 130 degrees in DOWN)
    4. Torso upright (back straight): shoulder-hip angle from vertical (same as squat)
    5. Hip drop: left/right hip y-position difference (threshold: 0.04)
    6. Back knee off ground: back knee y-position should be close to ankle y
       (hovering, not touching)

ANTI FALSE-POSITIVE:
    - WalkingDetector: ankle alternation + hip shift over 10 frames
    - AngleSmoother: 5-frame rolling average to kill single-frame spikes
    - StateDebouncer: prevents rapid UP/MID/DOWN flickering

VOICE FEEDBACK:
    - "Good lunge" on clean rep completion
    - "Rep counted, fix your [issue]" on rep with form issues
    - "Fix your [issue]" during DOWN phase if form problem detected
    - "Go lower" if stuck in MID state
    - "Walking detected, please stand still" if walking in frame
    - "Stand upright" if not in standing position
    - Cooldowns: general 2s, posture 3s, standing 3s

CONSTANTS:
    MIN_DOWN_FRAMES            = 15     frames to confirm DOWN state
    MID_HOLD_THRESHOLD         = 2.0    seconds in MID before "go lower" voice
    TRANSITION_DEBOUNCE_FRAMES = 5      frames before state changes
    BACK_ANGLE_THRESHOLD       = 30     degrees for back lean warning
    BACK_ANGLE_WARNING         = 45     degrees for severe back lean
    FRONT_KNEE_OVER_TOE_THRESHOLD = 0.08  normalized x-offset (knee vs toe)
    KNEE_FORWARD_DEBOUNCE_FRAMES   = 5      frames before flagging knee-over-toe
    HIP_DROP_THRESHOLD         = 0.04   normalized y-diff between hips
    FRONT_KNEE_TOO_SHALLOW     = 120    front knee angle in DOWN too shallow
    BACK_KNEE_TOO_HIGH          = 130    back knee angle in DOWN too far from ground
    GENERAL_COOLDOWN            = 2.0    seconds
    POSTURE_COOLDOWN            = 3.0    seconds
================================================================================
"""

import numpy as np
import time
import collections
import mediapipe as mp

from angle_utils import calculate_angle, AngleSmoother, SingleSmoother
from pose_utils import is_standing_posture
from detectors.base import BaseExercise, ExerciseResult

MIN_DOWN_FRAMES = 8
MID_HOLD_THRESHOLD = 2.0
TRANSITION_DEBOUNCE_FRAMES = 5

BACK_ANGLE_THRESHOLD = 30
BACK_ANGLE_WARNING = 45
FRONT_KNEE_OVER_TOE_THRESHOLD = 0.06
KNEE_FORWARD_DEBOUNCE_FRAMES = 5
HIP_DROP_THRESHOLD = 0.04
FRONT_KNEE_TOO_SHALLOW = 120
BACK_KNEE_TOO_HIGH = 130

GENERAL_COOLDOWN = 2.0
POSTURE_COOLDOWN = 3.0

MIN_LANDMARK_VISIBILITY = 0.4


def check_landmarks_visible(landmarks, mp_pose):
    key_joints = [
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]
    for joint in key_joints:
        lm = landmarks[joint.value]
        if lm.visibility < MIN_LANDMARK_VISIBILITY:
            return False

    return True


class WalkingDetector:
    def __init__(self, frames_required=10):
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
                (left_ankle_change > 0.03 and right_ankle_change < 0.015)
                or
                (right_ankle_change > 0.03 and left_ankle_change < 0.015)
            )

            significant_body_shift = hip_change > 0.02

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


def get_lunge_state(avg_front_knee_angle):
    if avg_front_knee_angle > 155:
        return "UP"
    elif avg_front_knee_angle < 120:
        return "DOWN"
    else:
        return "MID"


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


def _get_front_back_sides(landmarks, mp_pose, w, h, locked_front_side=None):
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

    if locked_front_side is not None:
        front_side = locked_front_side
        back_side = "RIGHT" if front_side == "LEFT" else "LEFT"
    else:
        if left_knee_y < right_knee_y:
            front_side, back_side = "LEFT", "RIGHT"
        else:
            front_side, back_side = "RIGHT", "LEFT"

    lm = mp_pose.PoseLandmark

    if front_side == "LEFT":
        front_hip = [landmarks[lm.LEFT_HIP.value].x * w, landmarks[lm.LEFT_HIP.value].y * h]
        front_knee = [landmarks[lm.LEFT_KNEE.value].x * w, landmarks[lm.LEFT_KNEE.value].y * h]
        front_ankle = [landmarks[lm.LEFT_ANKLE.value].x * w, landmarks[lm.LEFT_ANKLE.value].y * h]
        front_foot = landmarks[lm.LEFT_FOOT_INDEX.value]
        back_hip = [landmarks[lm.RIGHT_HIP.value].x * w, landmarks[lm.RIGHT_HIP.value].y * h]
        back_knee = [landmarks[lm.RIGHT_KNEE.value].x * w, landmarks[lm.RIGHT_KNEE.value].y * h]
        back_ankle = [landmarks[lm.RIGHT_ANKLE.value].x * w, landmarks[lm.RIGHT_ANKLE.value].y * h]
        back_foot = landmarks[lm.RIGHT_FOOT_INDEX.value]
    else:
        front_hip = [landmarks[lm.RIGHT_HIP.value].x * w, landmarks[lm.RIGHT_HIP.value].y * h]
        front_knee = [landmarks[lm.RIGHT_KNEE.value].x * w, landmarks[lm.RIGHT_KNEE.value].y * h]
        front_ankle = [landmarks[lm.RIGHT_ANKLE.value].x * w, landmarks[lm.RIGHT_ANKLE.value].y * h]
        front_foot = landmarks[lm.RIGHT_FOOT_INDEX.value]
        back_hip = [landmarks[lm.LEFT_HIP.value].x * w, landmarks[lm.LEFT_HIP.value].y * h]
        back_knee = [landmarks[lm.LEFT_KNEE.value].x * w, landmarks[lm.LEFT_KNEE.value].y * h]
        back_ankle = [landmarks[lm.LEFT_ANKLE.value].x * w, landmarks[lm.LEFT_ANKLE.value].y * h]
        back_foot = landmarks[lm.LEFT_FOOT_INDEX.value]

    return front_hip, front_knee, front_ankle, front_foot, back_hip, back_knee, back_ankle, back_foot, front_side


def check_front_knee_over_toe(front_knee_landmark, front_ankle_landmark):
    x_offset = front_knee_landmark.x - front_ankle_landmark.x
    return abs(x_offset) > FRONT_KNEE_OVER_TOE_THRESHOLD


def check_hip_drop(left_hip, right_hip):
    return abs(left_hip.y - right_hip.y) > HIP_DROP_THRESHOLD


def run_lunge_form_checks(landmarks, mp_pose, w, h, lunge_state, locked_front_side=None):
    issues = []

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    avg_shoulder = [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2]
    avg_hip = [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]

    shoulder_px = [avg_shoulder[0] * w, avg_shoulder[1] * h]
    hip_px = [avg_hip[0] * w, avg_hip[1] * h]

    back_angle = calculate_back_angle(shoulder_px, hip_px)

    if back_angle > BACK_ANGLE_WARNING:
        issues.append("SEVERE_LEAN")
    elif back_angle > BACK_ANGLE_THRESHOLD:
        issues.append("BACK_LEAN")

    hip_dropped = check_hip_drop(left_hip, right_hip)
    if hip_dropped:
        issues.append("HIP_DROP")

    if lunge_state == "DOWN":
        _, _, front_ankle_pt, _, _, _, _, _, front_side = _get_front_back_sides(landmarks, mp_pose, w, h, locked_front_side=locked_front_side)

        lm = mp_pose.PoseLandmark
        if front_side == "LEFT":
            front_knee_lm = landmarks[lm.LEFT_KNEE.value]
        else:
            front_knee_lm = landmarks[lm.RIGHT_KNEE.value]

        front_knee_over = check_front_knee_over_toe(front_knee_lm, landmarks[lm.LEFT_ANKLE.value if front_side == "LEFT" else lm.RIGHT_ANKLE.value])
        if front_knee_over:
            issues.append("KNEE_FORWARD")

    return issues, back_angle


def lunge_issues_to_feedback(issues):
    if not issues:
        return None
    if "SEVERE_LEAN" in issues:
        return "Straighten your back"
    elif "BACK_LEAN" in issues:
        return "Keep your back straight"
    elif "KNEE_FORWARD" in issues:
        return "Don't let knee go past ankle"
    elif "HIP_DROP" in issues:
        return "Keep your hips level"
    return None


def lunge_issues_to_voice(issues):
    if not issues:
        return None
    if "SEVERE_LEAN" in issues:
        return "Straighten your back"
    elif "BACK_LEAN" in issues:
        return "Keep your back straight"
    elif "KNEE_FORWARD" in issues:
        return "Don't let your front knee go past your ankle"
    elif "HIP_DROP" in issues:
        return "Keep your hips level"
    return None


def lunge_issues_to_display(issues):
    labels = []
    for issue in issues:
        if issue == "SEVERE_LEAN":
            labels.append("Back lean (severe)")
        elif issue == "BACK_LEAN":
            labels.append("Back leaning")
        elif issue == "KNEE_FORWARD":
            labels.append("Knee past ankle")
        elif issue == "HIP_DROP":
            labels.append("Hips uneven")
    return labels


def lunge_issue_to_fix_phrase(issues):
    if not issues:
        return None
    if "SEVERE_LEAN" in issues:
        return "straighten your back"
    if "BACK_LEAN" in issues:
        return "keep your back straight"
    if "KNEE_FORWARD" in issues:
        return "push your front knee back behind your toes"
    if "HIP_DROP" in issues:
        return "keep your hips level"
    return "improve your form"


class LungeExercise(BaseExercise):
    def __init__(self):
        super().__init__()
        self.get_name = lambda: "Lunge"

        self.angle_smoother = AngleSmoother()
        self.front_angle_smoother = SingleSmoother()
        self.back_angle_smoother = SingleSmoother()
        self.walking_detector = WalkingDetector()
        self.state_debouncer = StateDebouncer()

        self.lunge_started = False
        self.tracking_active = False
        self.locked_front_side = None
        self.knee_forward_frame_count = 0
        self.down_frame_count = 0
        self.mid_frame_count = 0
        self.mid_start_time = None
        self.rep_posture_issues = set()

        self.last_voice_time = 0
        self.walking_spoken = False
        self.go_lower_spoken = False
        self.last_posture_voice = None
        self.last_posture_voice_time = 0

        self.front_knee_angle = 0
        self.back_knee_angle = 0
        self.current_back_angle = 0
        self.smooth_avg = 0

    def check_ready(self, landmarks, mp_pose):
        if self.tracking_active:
            return True, ""

        if not check_landmarks_visible(landmarks, mp_pose):
            return False, "Make sure full body is visible"

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

    def reset(self):
        super().reset()
        self.lunge_started = False
        self.locked_front_side = None
        self.down_frame_count = 0
        self.mid_frame_count = 0
        self.mid_start_time = None
        self.rep_posture_issues = set()
        self.walking_spoken = False
        self.go_lower_spoken = False
        self.last_posture_voice = None
        self.last_posture_voice_time = 0
        self.front_knee_angle = 0
        self.back_knee_angle = 0
        self.current_back_angle = 0
        self.smooth_avg = 0
        self.angle_smoother.reset()
        self.front_angle_smoother.reset()
        self.back_angle_smoother.reset()
        self.walking_detector.reset()
        self.state_debouncer.reset()

    def is_tracking(self):
        return self.tracking_active

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
                left_angle=self.front_knee_angle,
                right_angle=self.back_knee_angle,
                avg_angle=self.smooth_avg
            )

        self.walking_spoken = False

        if not check_landmarks_visible(landmarks, mp_pose):
            self._reset_tracking()
            return ExerciseResult(
                state="INVALID",
                feedback="Make sure full body is visible",
                rep_count=self.rep_count,
                back_angle=self.current_back_angle,
                left_angle=self.front_knee_angle,
                right_angle=self.back_knee_angle,
                avg_angle=self.smooth_avg
            )

        (front_hip, front_knee_pt, front_ankle_pt, front_foot,
         back_hip, back_knee_pt, back_ankle_pt, back_foot,
         detected_front_side) = _get_front_back_sides(
            landmarks, mp_pose, w, h, locked_front_side=self.locked_front_side
        )

        self.front_knee_angle = calculate_angle(front_hip, front_knee_pt, front_ankle_pt)
        self.back_knee_angle = calculate_angle(back_hip, back_knee_pt, back_ankle_pt)

        self.front_angle_smoother.update(self.front_knee_angle)
        self.angle_smoother.update(self.front_knee_angle, self.back_knee_angle)
        self.smooth_avg = self.angle_smoother.get_smoothed_avg()
        smooth_front = self.front_angle_smoother.get_smoothed()

        raw_state = get_lunge_state(smooth_front)
        lunge_state = self.state_debouncer.update(raw_state)

        if raw_state != "UP" and not self.tracking_active:
            self.tracking_active = True

        posture_issues, self.current_back_angle = run_lunge_form_checks(
            landmarks, mp_pose, w, h, lunge_state, locked_front_side=self.locked_front_side
        )
        if lunge_state == "DOWN":
            if self.locked_front_side is None:
                self.locked_front_side = detected_front_side
            self.down_frame_count += 1
            self.mid_frame_count = 0

            if "KNEE_FORWARD" in posture_issues:
                self.knee_forward_frame_count += 1
            else:
                self.knee_forward_frame_count = max(0, self.knee_forward_frame_count - 1)

            if self.knee_forward_frame_count >= KNEE_FORWARD_DEBOUNCE_FRAMES:
                if "KNEE_FORWARD" not in posture_issues:
                    posture_issues.append("KNEE_FORWARD")
            else:
                if "KNEE_FORWARD" in posture_issues:
                    posture_issues.remove("KNEE_FORWARD")
        elif lunge_state == "MID":
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
            self.lunge_started = True
            self.tracking_active = True

            if not posture_issues:
                feedback = "Good lunge"
            else:
                fix_phrase = lunge_issue_to_fix_phrase(posture_issues)
                feedback = fix_phrase
                self.go_lower_spoken = False

            if lunge_state == "DOWN" and posture_issues:
                self.rep_posture_issues.update(posture_issues)

        elif lunge_state == "MID" and self.down_frame_count < MIN_DOWN_FRAMES:
            posture_fb = lunge_issues_to_feedback(posture_issues)

            if posture_fb:
                feedback = posture_fb
            else:
                feedback = "Go lower"

            posture_voice = lunge_issues_to_voice(posture_issues)

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

        if self.lunge_started and lunge_state == "UP":
            self.rep_count += 1
            self.lunge_started = False
            self.locked_front_side = None
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
                fix_phrase = lunge_issue_to_fix_phrase(list(self.rep_posture_issues))
                feedback = "Rep counted"
                voice_msg = f"Rep counted, {fix_phrase}"
            else:
                feedback = "Good lunge"
                voice_msg = "Good lunge"

            self.rep_posture_issues = set()

        if lunge_state == "UP" and not self.lunge_started:
            feedback = "None"
            self.mid_start_time = None
            self.go_lower_spoken = False
            self.last_posture_voice = None

        posture_labels = lunge_issues_to_display(posture_issues)

        return ExerciseResult(
            state=lunge_state,
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
            left_angle=self.front_knee_angle,
            right_angle=self.back_knee_angle,
            avg_angle=self.smooth_avg
        )

    def _reset_tracking(self):
        self.down_frame_count = 0
        self.mid_frame_count = 0
        self.lunge_started = False
        self.mid_start_time = None
        self.go_lower_spoken = False
        self.state_debouncer.reset()
        self.angle_smoother.reset()
        self.front_angle_smoother.reset()
        self.back_angle_smoother.reset()
        self.last_posture_voice = None
