"""
================================================================================
NeuralStride AI - Push-Up Exercise Detector
================================================================================

Detects and analyzes push-up exercises using MediaPipe Pose landmarks.
Extends BaseExercise from detectors/base.py.

CAMERA VIEW:
    Side/diagonal view recommended. Person lies horizontal (plank position)
    with camera at an angle to see both elbow depth and hip alignment.

ANGLE TRACKING:
    - Elbow angle (shoulder -> elbow -> wrist) using Law of Cosines
    - Average of left and right elbow angles
    - 5-frame rolling average for smoothing (AngleSmoother)

STATE DETECTION:
    UP   : elbow angle > 160 degrees (arms straight, plank position)
    MID  : elbow angle 90-160 degrees (partial push-up)
    DOWN : elbow angle < 90 degrees (chest near ground)
    State debouncing: new state must hold for 5 frames before transitioning

REP COUNTING:
    Counts a full rep: UP -> DOWN (hold 15 frames) -> UP
    Each rep logs: rep number, duration, elbow angle, form issues, clean/not

FORM CHECKS:
    - Hip sag: hips dropping below the shoulder-ankle midpoint (threshold: 0.06)
    - Pike: hips too high above the shoulder-ankle midpoint (threshold: 0.06)
    - Shallow push-up: stuck in MID without reaching DOWN state
    - Arm symmetry: both elbows must bend within 40 degrees of each other
    - Hand spacing: wrist-to-wrist x-distance (narrow < 0.08, wide > 0.35)
    - Elbow flare: elbow x-offset from shoulder relative to shoulder span (> 0.10)

STARTING POSITION (check_ready):
    - Body must be roughly horizontal (shoulder-hip y-diff < 0.25)
    - Body must be extended (hip-ankle y-diff < 0.30)
    - Once tracking starts (after first DOWN reached), ready check is skipped
      so the exercise isn't interrupted during push-up movement

VOICE FEEDBACK:
    - "Good push-up" on clean rep completion
    - "Rep counted, lift your hips up" on hip sag rep (specific actionable phrases)
    - "Rep counted, lower your hips down" on pike rep
    - "Rep counted, push up deeper" on shallow rep
    - "Rep counted, widen/narrow your hand spacing" for hand spacing issues
    - "Rep counted, tuck your elbows in" for elbow flare
    - "Go lower" if stuck in MID state
    - "Get into push-up position" (spoken once) if not in plank position
    - Cooldowns: general 2s, posture 3s

CONSTANTS:
    ARM_ASYMMETRY_THRESHOLD    = 40     degrees max diff between elbows
    WALKING_ANKLE_CHANGE       = 0.03   normalized ankle movement
    WALKING_HIP_CHANGE         = 0.02   normalized hip movement
    WALKING_FRAMES_REQUIRED    = 10     frames to confirm walking
    MIN_DOWN_FRAMES            = 15     frames to confirm DOWN state
    TRANSITION_DEBOUNCE_FRAMES = 5      frames before state changes
    HIP_SAG_THRESHOLD          = 0.06   normalized diff from expected hip position
    PIKE_THRESHOLD             = 0.06   normalized diff from expected hip position
    PLANK_READY_THRESHOLD      = 0.25   shoulder-hip y-diff for plank detection
    HAND_SPACING_NARROW_RATIO = 0.6   wrist_distance / shoulder_span (too close)
    HAND_SPACING_WIDE_RATIO   = 2.8   wrist_distance / shoulder_span (too far)
    ELBOW_FLARE_THRESHOLD     = 0.10  elbow offset / shoulder span ratio

================================================================================
"""

import time
import mediapipe as mp

from angle_utils import calculate_angle, AngleSmoother
from detectors.base import BaseExercise, ExerciseResult

WALKING_ANKLE_CHANGE_THRESHOLD = 0.03
WALKING_HIP_CHANGE_THRESHOLD = 0.02
WALKING_FRAMES_REQUIRED = 10
MIN_DOWN_FRAMES = 10
TRANSITION_DEBOUNCE_FRAMES = 5
MID_HOLD_THRESHOLD = 2.0

ARM_ASYMMETRY_THRESHOLD = 40
HIP_SAG_THRESHOLD = 0.06
PIKE_THRESHOLD = 0.06

GENERAL_COOLDOWN = 2.0
POSTURE_COOLDOWN = 3.0
PLANK_READY_THRESHOLD = 0.25

HAND_SPACING_NARROW_RATIO = 0.6
HAND_SPACING_WIDE_RATIO = 2.8
ELBOW_FLARE_THRESHOLD = 0.10


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


def get_pushup_state(avg_elbow_angle):
    if avg_elbow_angle > 160:
        return "UP"
    elif avg_elbow_angle < 90:
        return "DOWN"
    else:
        return "MID"


def check_arm_symmetry(left_angle, right_angle):
    if ARM_ASYMMETRY_THRESHOLD == 0:
        return True
    return abs(left_angle - right_angle) < ARM_ASYMMETRY_THRESHOLD


def check_hand_spacing(landmarks, mp_pose):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    wrist_distance = abs(left_wrist.x - right_wrist.x)
    shoulder_span = abs(left_shoulder.x - right_shoulder.x)

    if shoulder_span < 0.01:
        return None

    ratio = wrist_distance / shoulder_span

    if ratio < HAND_SPACING_NARROW_RATIO:
        return "HANDS_NARROW"
    elif ratio > HAND_SPACING_WIDE_RATIO:
        return "HANDS_WIDE"
    return None


def check_elbow_flare(landmarks, mp_pose):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    shoulder_span = abs(left_shoulder.x - right_shoulder.x)
    if shoulder_span < 0.01:
        return None

    left_flare = (left_elbow.x - left_shoulder.x) / shoulder_span
    right_flare = (right_shoulder.x - right_elbow.x) / shoulder_span

    avg_flare = (left_flare + right_flare) / 2
    if avg_flare > ELBOW_FLARE_THRESHOLD:
        return "ELBOW_FLARE"
    return None


def check_hip_alignment(landmarks, mp_pose):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    avg_hip_y = (left_hip.y + right_hip.y) / 2
    avg_ankle_y = (left_ankle.y + right_ankle.y) / 2

    expected_hip_y = (avg_shoulder_y + avg_ankle_y) / 2
    diff = avg_hip_y - expected_hip_y

    if diff > HIP_SAG_THRESHOLD:
        return "HIP_SAG"
    elif diff < -PIKE_THRESHOLD:
        return "PIKE"
    return None


def run_pushup_form_checks(landmarks, mp_pose, w, h, pushup_state):
    issues = []

    hip_issue = check_hip_alignment(landmarks, mp_pose)
    if hip_issue:
        issues.append(hip_issue)

    spacing_issue = check_hand_spacing(landmarks, mp_pose)
    if spacing_issue:
        issues.append(spacing_issue)

    if pushup_state == "MID":
        issues.append("SHALLOW")

    return issues, 0


def pushup_issues_to_feedback(issues):
    if not issues:
        return None
    if "HIP_SAG" in issues:
        return "Lift your hips"
    elif "PIKE" in issues:
        return "Lower your hips"
    elif "SHALLOW" in issues:
        return "Go lower"
    elif "HANDS_NARROW" in issues:
        return "Widen hand spacing"
    elif "HANDS_WIDE" in issues:
        return "Narrow hand spacing"
    elif "ELBOW_FLARE" in issues:
        return "Tuck elbows in"
    return None


def pushup_issues_to_voice(issues):
    if not issues:
        return None
    if "HIP_SAG" in issues:
        return "Lift your hips up"
    elif "PIKE" in issues:
        return "Lower your hips down"
    elif "SHALLOW" in issues:
        return "Go lower"
    elif "HANDS_NARROW" in issues:
        return "Widen your hand spacing"
    elif "HANDS_WIDE" in issues:
        return "Narrow your hand spacing"
    elif "ELBOW_FLARE" in issues:
        return "Tuck your elbows in"
    return None


def pushup_issues_to_display(issues):
    labels = []
    for issue in issues:
        if issue == "HIP_SAG":
            labels.append("Hips sagging")
        elif issue == "PIKE":
            labels.append("Hips too high")
        elif issue == "SHALLOW":
            labels.append("Too shallow")
        elif issue == "HANDS_NARROW":
            labels.append("Hands too narrow")
        elif issue == "HANDS_WIDE":
            labels.append("Hands too wide")
        elif issue == "ELBOW_FLARE":
            labels.append("Elbows flaring")
    return labels


def pushup_issue_to_fix_phrase(issues):
    if not issues:
        return None
    if "HIP_SAG" in issues:
        return "lift your hips up"
    if "PIKE" in issues:
        return "lower your hips down"
    if "SHALLOW" in issues:
        return "push up deeper"
    if "HANDS_NARROW" in issues:
        return "widen your hand spacing"
    if "HANDS_WIDE" in issues:
        return "narrow your hand spacing"
    if "ELBOW_FLARE" in issues:
        return "tuck your elbows in"
    return "improve your form"


class PushUpExercise(BaseExercise):
    def __init__(self):
        super().__init__()
        self.get_name = lambda: "Push-Up"

        self.angle_smoother = AngleSmoother()
        self.walking_detector = WalkingDetector()
        self.state_debouncer = StateDebouncer()

        self.pushup_started = False
        self.down_frame_count = 0
        self.mid_frame_count = 0
        self.mid_start_time = None
        self.rep_posture_issues = set()

        self.last_voice_time = 0
        self.walking_spoken = False
        self.go_lower_spoken = False
        self.last_posture_voice = None
        self.last_posture_voice_time = 0

        self.left_elbow_angle = 0
        self.right_elbow_angle = 0
        self.head_angle = 0
        self.smooth_avg = 0
        self.tracking_active = False

    def check_ready(self, landmarks, mp_pose):
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        avg_shoulder_y = (ls.y + rs.y) / 2
        avg_hip_y = (lh.y + rh.y) / 2
        avg_ankle_y = (la.y + ra.y) / 2

        shoulder_hip_diff = abs(avg_shoulder_y - avg_hip_y)
        hip_ankle_diff = abs(avg_hip_y - avg_ankle_y)

        body_horizontal = shoulder_hip_diff < PLANK_READY_THRESHOLD
        body_extended = hip_ankle_diff < PLANK_READY_THRESHOLD + 0.05

        if body_horizontal and body_extended:
            return True, ""
        return False, "Get into push-up position"

    def reset(self):
        super().reset()
        self.pushup_started = False
        self.down_frame_count = 0
        self.mid_frame_count = 0
        self.mid_start_time = None
        self.rep_posture_issues = set()
        self.walking_spoken = False
        self.go_lower_spoken = False
        self.last_posture_voice = None
        self.last_posture_voice_time = 0
        self.left_elbow_angle = 0
        self.right_elbow_angle = 0
        self.head_angle = 0
        self.smooth_avg = 0
        self.tracking_active = False
        self.angle_smoother.reset()
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
                back_angle=self.head_angle,
                left_angle=self.left_elbow_angle,
                right_angle=self.right_elbow_angle,
                avg_angle=self.smooth_avg
            )

        self.walking_spoken = False

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        left_shoulder_pt = [left_shoulder.x * w, left_shoulder.y * h]
        left_elbow_pt = [left_elbow.x * w, left_elbow.y * h]
        left_wrist_pt = [left_wrist.x * w, left_wrist.y * h]

        self.left_elbow_angle = calculate_angle(left_shoulder_pt, left_elbow_pt, left_wrist_pt)

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        right_shoulder_pt = [right_shoulder.x * w, right_shoulder.y * h]
        right_elbow_pt = [right_elbow.x * w, right_elbow.y * h]
        right_wrist_pt = [right_wrist.x * w, right_wrist.y * h]

        self.right_elbow_angle = calculate_angle(right_shoulder_pt, right_elbow_pt, right_wrist_pt)

        if not check_arm_symmetry(self.left_elbow_angle, self.right_elbow_angle):
            self._reset_tracking()
            return ExerciseResult(
                state="INVALID",
                feedback="Keep both arms even",
                form_labels=["ASYMMETRIC"],
                rep_count=self.rep_count,
                back_angle=self.head_angle,
                left_angle=self.left_elbow_angle,
                right_angle=self.right_elbow_angle,
                avg_angle=self.smooth_avg
            )

        self.angle_smoother.update(self.left_elbow_angle, self.right_elbow_angle)
        self.smooth_avg = self.angle_smoother.get_smoothed_avg()

        raw_state = get_pushup_state(self.smooth_avg)
        pushup_state = self.state_debouncer.update(raw_state)

        posture_issues, self.head_angle = run_pushup_form_checks(
            landmarks, mp_pose, w, h, pushup_state
        )
        posture_labels = pushup_issues_to_display(posture_issues)

        if pushup_state == "DOWN":
            self.down_frame_count += 1
            self.mid_frame_count = 0
        elif pushup_state == "MID":
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
            self.pushup_started = True
            self.tracking_active = True

            if not posture_issues:
                feedback = "Good push-up"
            else:
                fix_phrase = pushup_issue_to_fix_phrase(posture_issues)
                feedback = fix_phrase
                self.go_lower_spoken = False

            if pushup_state == "DOWN" and posture_issues:
                self.rep_posture_issues.update(posture_issues)
            elif pushup_state != "DOWN":
                spacing_only = [i for i in posture_issues if i in ("HANDS_NARROW", "HANDS_WIDE", "ELBOW_FLARE")]
                self.rep_posture_issues.update(spacing_only)

        elif pushup_state == "MID" and self.down_frame_count < MIN_DOWN_FRAMES:
            posture_fb = pushup_issues_to_feedback(posture_issues)

            if posture_fb:
                feedback = posture_fb
            else:
                feedback = "Go lower"

            posture_voice = pushup_issues_to_voice(posture_issues)

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

        if self.pushup_started and pushup_state == "UP":
            self.rep_count += 1
            self.pushup_started = False
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
                fix_phrase = pushup_issue_to_fix_phrase(list(self.rep_posture_issues))
                feedback = "Rep counted"
                voice_msg = f"Rep counted, {fix_phrase}"
            else:
                feedback = "Good push-up"
                voice_msg = "Good push-up"

            self.rep_posture_issues = set()

        if pushup_state == "UP" and not self.pushup_started:
            feedback = "None"
            self.mid_start_time = None
            self.go_lower_spoken = False
            self.last_posture_voice = None

        return ExerciseResult(
            state=pushup_state,
            feedback=feedback,
            form_labels=posture_labels,
            voice_message=voice_msg,
            rep_count=self.rep_count,
            back_angle=self.head_angle,
            rep_completed=rep_completed,
            rep_had_issues=rep_had_issues,
            rep_issues=rep_issues,
            rep_avg_angle=rep_avg_angle,
            left_angle=self.left_elbow_angle,
            right_angle=self.right_elbow_angle,
            avg_angle=self.smooth_avg
        )

    def _reset_tracking(self):
        self.down_frame_count = 0
        self.mid_frame_count = 0
        self.pushup_started = False
        self.mid_start_time = None
        self.go_lower_spoken = False
        self.state_debouncer.reset()
        self.angle_smoother.reset()
        self.last_posture_voice = None
