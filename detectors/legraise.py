"""
================================================================================
NeuralStride AI - Side Leg Raise Exercise Detector (Front View)
================================================================================

Detects standing side leg raises from a FRONT-FACING camera.

EXERCISE:
    Person stands facing the camera, raises one leg out to the side (left or right).
    The detector auto-detects which leg is being raised based on ankle displacement.

STATE MACHINE:
    UP    : Standing, both feet on ground (idle between reps)
    MID   : Leg is being raised but not yet at threshold
    DOWN  : Leg raised to >= 45 degrees (peak of rep)
    A rep = leg leaves UP, reaches DOWN (>= 45 deg), returns to UP

FORM CHECKS:
    1. Knee must stay straight (> 160 degrees) on the raised leg
    2. Rep speed must be average (not too fast, not too slow)
    3. Leg must reach >= 45 degrees at peak to count as a valid rep

AUTO-DETECTION:
    Compares left vs right ankle vertical displacement from hip.
    The side with more displacement is the active (raised) leg.

CONSTANTS:
    RAISE_ANGLE_THRESHOLD     = 45    degrees to count as valid rep peak
    KNEE_STRAIGHT_THRESHOLD  = 160   degrees - below this = bent knee error
    MIN_DOWN_FRAMES           = 5     minimum frames in DOWN before counting
    MIN_UP_FRAMES             = 5     minimum frames in UP before new rep
    FAST_REP_THRESHOLD        = 0.4  seconds - below this = too fast
    SLOW_REP_THRESHOLD        = 4.0  seconds - above this = too slow
================================================================================
"""

import time
import math
import mediapipe as mp

from angle_utils import calculate_angle, SingleSmoother
from detectors.base import BaseExercise, ExerciseResult

RAISE_ANGLE_THRESHOLD = 45
KNEE_STRAIGHT_THRESHOLD = 168
MIN_DOWN_FRAMES = 5
MIN_UP_FRAMES = 5
FAST_REP_THRESHOLD = 1.3
SLOW_REP_THRESHOLD = 5.0
GENERAL_COOLDOWN = 1.0
POSTURE_COOLDOWN = 1.5


def calculate_abduction_angle(hip_lm, ankle_lm):
    """
    Calculate hip abduction angle (leg raise angle from vertical).

    Measures the angle between the vertical line from hip and the hip-to-ankle
    line. When standing, ankle is directly below hip so angle is ~0. When leg
    is raised horizontally, angle is ~90.

    Returns:
        Abduction angle in degrees. 0 = leg vertical (standing), 90 = leg horizontal.
    """
    dx = ankle_lm.x - hip_lm.x
    dy = ankle_lm.y - hip_lm.y
    if dy < 1e-6:
        return 90.0
    return abs(math.degrees(math.atan2(abs(dx), dy)))


def detect_active_side(left_hip, right_hip, left_ankle, right_ankle):
    """
    Detect which leg is being raised by comparing abduction angles.

    Returns:
        "LEFT", "RIGHT", or None (if neither leg is clearly raised).
    """
    left_angle = calculate_abduction_angle(left_hip, left_ankle)
    right_angle = calculate_abduction_angle(right_hip, right_ankle)

    diff = abs(left_angle - right_angle)
    if diff < 5:
        return None
    return "LEFT" if left_angle > right_angle else "RIGHT"


def check_knee_straight(hip_lm, knee_lm, ankle_lm):
    """
    Check if the knee is straight (not bent).

    Returns:
        True if knee angle >= KNEE_STRAIGHT_THRESHOLD (straight).
    """
    hip = (hip_lm.x * 640, hip_lm.y * 480)
    knee = (knee_lm.x * 640, knee_lm.y * 480)
    ankle = (ankle_lm.x * 640, ankle_lm.y * 480)
    angle = calculate_angle(hip, knee, ankle)
    return angle >= KNEE_STRAIGHT_THRESHOLD, angle


class StateDebouncer:
    """Prevents state flickering by requiring N consecutive frames."""

    def __init__(self, frames=3):
        self.count = 0
        self.required = frames
        self.last_state = "UP"

    def update(self, raw_state):
        if raw_state == self.last_state:
            self.count = 0
            return self.last_state
        self.count += 1
        if self.count >= self.required:
            self.last_state = raw_state
            self.count = 0
        return self.last_state

    def reset(self):
        self.count = 0
        self.last_state = "UP"


class LegRaiseExercise(BaseExercise):
    def __init__(self):
        super().__init__()
        self.get_name = lambda: "Side Leg Raise"

        self.state_debouncer = StateDebouncer(frames=3)
        self.active_side = None

        self.left_knee_smoother = SingleSmoother()
        self.right_knee_smoother = SingleSmoother()
        self.angle_smoother = SingleSmoother(buffer_size=7)

        self.tracking_active = False
        self.rep_started = False
        self.reached_peak = False
        self.rep_knee_bent = False
        self.down_frame_count = 0
        self.up_frame_count = 0
        self.rep_start_time = None

        self.last_voice_time = 0
        self.last_posture_voice = None
        self.last_posture_voice_time = 0
        self.good_rep_spoken = False

        self.current_abduction_angle = 0
        self.current_knee_angle = 0

    def check_ready(self, landmarks, mp_pose):
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        shoulder_y_diff = abs(ls.y - rs.y)
        hip_y_diff = abs(lh.y - rh.y)

        upright = shoulder_y_diff < 0.06 and hip_y_diff < 0.06
        if upright:
            return True, ""
        return False, "Stand facing the camera with feet together"

    def reset(self):
        super().reset()
        self.state_debouncer.reset()
        self.active_side = None
        self.left_knee_smoother.reset()
        self.right_knee_smoother.reset()
        self.angle_smoother.reset()
        self.tracking_active = False
        self.rep_started = False
        self.reached_peak = False
        self.rep_knee_bent = False
        self.down_frame_count = 0
        self.up_frame_count = 0
        self.rep_start_time = None
        self.last_voice_time = 0
        self.last_posture_voice = None
        self.last_posture_voice_time = 0
        self.good_rep_spoken = False
        self.current_abduction_angle = 0
        self.current_knee_angle = 0

    def is_tracking(self):
        return self.tracking_active

    def get_idle_state(self):
        return "UP"

    def update(self, landmarks, mp_pose, w, h, now):
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        left_abd = calculate_abduction_angle(lh, la)
        right_abd = calculate_abduction_angle(rh, ra)

        self.active_side = detect_active_side(lh, rh, la, ra)
        if self.active_side is None:
            active_abd = (left_abd + right_abd) / 2
        elif self.active_side == "LEFT":
            active_abd = left_abd
        else:
            active_abd = right_abd

        self.angle_smoother.update(active_abd)
        smoothed_angle = self.angle_smoother.get_smoothed()
        self.current_abduction_angle = smoothed_angle

        if smoothed_angle >= RAISE_ANGLE_THRESHOLD:
            raw_state = "DOWN"
        elif smoothed_angle >= 15:
            raw_state = "MID"
        else:
            raw_state = "UP"

        state = self.state_debouncer.update(raw_state)

        print(f"[DEBUG] L_abd={left_abd:.1f} R_abd={right_abd:.1f} smooth={smoothed_angle:.1f} side={self.active_side} raw={raw_state} state={state} reps={self.rep_count}", flush=True)

        self.tracking_active = True

        posture_issues = []
        voice_msg = None
        rep_completed = False
        rep_had_issues = False
        rep_issues = []

        if self.active_side == "LEFT":
            knee_straight, knee_angle = check_knee_straight(lh, lk, la)
        elif self.active_side == "RIGHT":
            knee_straight, knee_angle = check_knee_straight(rh, rk, ra)
        else:
            knee_straight = True
            knee_angle = 180.0

        self.current_knee_angle = knee_angle

        if not knee_straight and smoothed_angle >= 15:
            posture_issues.append("KNEE_BENT")
            if self.rep_started:
                self.rep_knee_bent = True

        if not self.rep_started and state in ["MID", "DOWN"]:
            self.rep_started = True
            self.rep_start_time = now
            self.reached_peak = False
            self.rep_knee_bent = False

        if state == "DOWN":
            self.down_frame_count += 1
            self.up_frame_count = 0
            self.reached_peak = True

            if self.down_frame_count == MIN_DOWN_FRAMES and not self.good_rep_spoken:
                self.good_rep_spoken = True
        else:
            self.down_frame_count = 0

        if state == "UP":
            self.up_frame_count += 1

            if self.rep_started and self.up_frame_count >= MIN_UP_FRAMES and self.reached_peak:
                rep_completed = True
                self.rep_count += 1
                self.rep_started = False
                self.good_rep_spoken = False
                self.reached_peak = False

                if self.rep_knee_bent:
                    rep_had_issues = True
                    rep_issues = ["KNEE_BENT"]
                    voice_msg = "Rep counted, straighten your knee"
                else:
                    voice_msg = "Good leg raise"

                self.rep_start_time = None
        else:
            self.up_frame_count = 0

        if not rep_completed:
            if posture_issues:
                if "KNEE_BENT" in posture_issues:
                    feedback = "Keep your knee straight"
                    posture_voice = "Straighten your knee"
                else:
                    feedback = "Fix your form"
                    posture_voice = None

                if posture_voice and posture_voice != self.last_posture_voice:
                    if now - self.last_posture_voice_time >= POSTURE_COOLDOWN:
                        voice_msg = posture_voice
                        self.last_voice_time = now
                        self.last_posture_voice = posture_voice
                        self.last_posture_voice_time = now
        else:
            self.last_posture_voice = None

        if self.rep_started and not self.reached_peak and state == "UP":
            if now - self.last_posture_voice_time >= GENERAL_COOLDOWN:
                voice_msg = "Raise your leg higher"
                self.last_voice_time = now
                self.last_posture_voice_time = now
                self.rep_started = False
                self.rep_start_time = None

        if self.active_side:
            side_label = self.active_side.capitalize()
            angle_display = f"{side_label} leg: {smoothed_angle:.0f} deg"
        else:
            angle_display = f"{smoothed_angle:.0f} deg"

        return ExerciseResult(
            state=state,
            feedback=feedback if posture_issues else angle_display,
            form_labels=[angle_display] + posture_issues,
            form_issue_codes=posture_issues,
            voice_message=voice_msg,
            rep_count=self.rep_count,
            back_angle=smoothed_angle,
            rep_completed=rep_completed,
            rep_had_issues=rep_had_issues,
            rep_issues=rep_issues,
            rep_avg_angle=smoothed_angle,
            left_angle=left_abd,
            right_angle=right_abd,
            avg_angle=smoothed_angle,
        )
