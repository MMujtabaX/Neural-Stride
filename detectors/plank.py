"""
================================================================================
NeuralStride AI - Plank Exercise Detector
================================================================================

Detects and analyzes low plank (forearm) holds using MediaPipe Pose landmarks.
Extends BaseExercise from detectors/base.py.

PLANK TYPE:
    Low plank (forearms on ground, elbows bent at ~90 degrees).
    Side view recommended. Camera perpendicular to body to see alignment.

HOLD TRACKING:
    - Detects when person is in correct plank position (check_ready)
    - Starts a timer once tracking begins
    - Counts down to target time (configurable per session)
    - When target time reached, counts 1 rep and starts a new hold
    - If person drops out of plank before target, hold resets (reps preserved)

STATE DETECTION:
    HOLDING  : body horizontal, elbows bent, maintaining position
    BROKEN   : body position lost (standing up, collapsing, etc.)

FORM CHECKS:
    - Hip sag: hips dropping below shoulder-ankle midpoint (threshold: 0.06)
    - Pike: hips too high above shoulder-ankle midpoint (threshold: 0.06)
    - Shoulder alignment: shoulder-hip y-diff must stay small

REP COUNTING:
    1 rep = reaching the target hold time with acceptable form
    If form issues occur during hold, the rep is counted but flagged

VOICE FEEDBACK:
    - Countdown voice at key milestones (50%, 75%, done)
    - Form corrections during hold (hip sag, pike)
    - "Hold complete" on successful rep
    - "Hold broken" if plank position lost before target
    - Cooldowns: general 2s, posture 3s

CONSTANTS:
    HIP_SAG_THRESHOLD          = 0.04
    PIKE_THRESHOLD             = 0.02
    PLANK_READY_THRESHOLD      = 0.25  shoulder-hip y-diff for plank detection
    PLANK_EXTENDED_THRESHOLD   = 0.30  hip-ankle y-diff for plank detection
    GENERAL_COOLDOWN            = 0.5
    POSTURE_COOLDOWN            = 0.8

================================================================================
"""

import time
import mediapipe as mp

from angle_utils import calculate_angle
from detectors.base import BaseExercise, ExerciseResult

HIP_SAG_THRESHOLD = 0.035
PIKE_THRESHOLD = 0.02
PLANK_READY_THRESHOLD = 0.25
PLANK_EXTENDED_THRESHOLD = 0.30

GENERAL_COOLDOWN = 0.5
POSTURE_COOLDOWN = 0.8

DEFAULT_TARGET_TIME = 30


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


def plank_issues_to_feedback(issues):
    if not issues:
        return None
    if "HIP_SAG" in issues:
        return "Lift your hips"
    elif "PIKE" in issues:
        return "Lower your hips"
    return None


def plank_issues_to_voice(issues):
    if not issues:
        return None
    if "HIP_SAG" in issues:
        return "Lift your hips up"
    elif "PIKE" in issues:
        return "Lower your hips down"
    return None


def plank_issues_to_display(issues):
    labels = []
    for issue in issues:
        if issue == "HIP_SAG":
            labels.append("Hips sagging")
        elif issue == "PIKE":
            labels.append("Hips too high")
    return labels


def plank_issue_to_fix_phrase(issues):
    if not issues:
        return None
    if "HIP_SAG" in issues:
        return "hip position"
    if "PIKE" in issues:
        return "hip position"
    return "form"


class PlankExercise(BaseExercise):
    def __init__(self, target_time=DEFAULT_TARGET_TIME):
        super().__init__()
        self.get_name = lambda: "Plank"

        self.target_time = target_time
        self.hold_start_time = None
        self.hold_active = False
        self.tracking_active = False
        self.hold_posture_issues = set()

        self.last_voice_time = 0
        self.last_posture_voice = None
        self.last_posture_voice_time = 0
        self.countdown_50_spoken = False
        self.countdown_75_spoken = False
        self.countdown_done_spoken = False
        self.hold_broken_spoken = False

        self.elbow_angle = 0
        self.hold_elapsed = 0

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
        body_extended = hip_ankle_diff < PLANK_EXTENDED_THRESHOLD

        if body_horizontal and body_extended:
            return True, ""
        return False, "Get into plank position"

    def reset(self):
        super().reset()
        self.hold_start_time = None
        self.hold_active = False
        self.tracking_active = False
        self.hold_posture_issues = set()
        self.last_voice_time = 0
        self.last_posture_voice = None
        self.last_posture_voice_time = 0
        self.countdown_50_spoken = False
        self.countdown_75_spoken = False
        self.countdown_done_spoken = False
        self.hold_broken_spoken = False
        self.elbow_angle = 0
        self.hold_elapsed = 0

    def is_tracking(self):
        return self.tracking_active

    def get_idle_state(self):
        return "BROKEN"

    def update(self, landmarks, mp_pose, w, h, now):
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        left_shoulder_pt = [left_shoulder.x * w, left_shoulder.y * h]
        left_elbow_pt = [left_elbow.x * w, left_elbow.y * h]
        left_wrist_pt = [left_wrist.x * w, left_wrist.y * h]

        self.elbow_angle = calculate_angle(left_shoulder_pt, left_elbow_pt, left_wrist_pt)

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

        still_in_plank = (
            shoulder_hip_diff < PLANK_READY_THRESHOLD and
            hip_ankle_diff < PLANK_EXTENDED_THRESHOLD
        )

        if not still_in_plank:
            if self.hold_active:
                self.hold_active = False
                self.hold_start_time = None
                self.hold_posture_issues = set()
                self.countdown_50_spoken = False
                self.countdown_75_spoken = False
                self.countdown_done_spoken = False
                if not self.hold_broken_spoken and (now - self.last_voice_time >= GENERAL_COOLDOWN):
                    self.hold_broken_spoken = True
                    self.last_voice_time = now
                    return ExerciseResult(
                        state="BROKEN",
                        feedback="Hold broken",
                        form_labels=[],
                        voice_message="Hold broken",
                        rep_count=self.rep_count,
                        back_angle=self.elbow_angle,
                        left_angle=self.elbow_angle,
                        right_angle=self.elbow_angle,
                        avg_angle=self.elbow_angle
                    )
            self.hold_broken_spoken = False
            self.hold_elapsed = 0
            return ExerciseResult(
                state="BROKEN",
                feedback="Get into plank position",
                rep_count=self.rep_count,
                back_angle=self.elbow_angle,
                left_angle=self.elbow_angle,
                right_angle=self.elbow_angle,
                avg_angle=self.elbow_angle
            )

        self.hold_broken_spoken = False
        self.tracking_active = True

        if not self.hold_active:
            self.hold_active = True
            self.hold_start_time = now
            self.hold_posture_issues = set()
            self.countdown_50_spoken = False
            self.countdown_75_spoken = False
            self.countdown_done_spoken = False

        self.hold_elapsed = now - self.hold_start_time
        progress = self.hold_elapsed / self.target_time

        posture_issues = []
        hip_issue = check_hip_alignment(landmarks, mp_pose)
        if hip_issue:
            posture_issues.append(hip_issue)

        posture_labels = plank_issues_to_display(posture_issues)

        feedback = f"Holding... {self.hold_elapsed:.1f}s / {self.target_time}s"
        voice_msg = None
        rep_completed = False
        rep_had_issues = False
        rep_issues = []

        if progress >= 1.0:
            self.rep_count += 1
            rep_completed = True
            rep_avg_angle = self.hold_elapsed

            snapshot_issues = list(self.hold_posture_issues) if self.hold_posture_issues else []

            if snapshot_issues:
                rep_had_issues = True
                rep_issues = snapshot_issues

                if "HIP_SAG" in snapshot_issues and "PIKE" in snapshot_issues:
                    voice_msg = "Hold complete. Fix your hip position"
                elif "HIP_SAG" in snapshot_issues:
                    voice_msg = "Hold complete. Lift your hips up"
                elif "PIKE" in snapshot_issues:
                    voice_msg = "Hold complete. Lower your hips down"
                else:
                    voice_msg = "Hold complete"
                feedback = f"Hold complete! Rep {self.rep_count}"
            else:
                feedback = f"Hold complete! Rep {self.rep_count}"
                voice_msg = "Hold complete"

            self.hold_active = False
            self.hold_start_time = None
            self.hold_posture_issues = set()
            self.countdown_50_spoken = False
            self.countdown_75_spoken = False
            self.countdown_done_spoken = False
        else:
            if progress >= 0.75 and not self.countdown_75_spoken:
                self.countdown_75_spoken = True
                remaining = max(1, int(self.target_time - self.hold_elapsed))
                voice_msg = f"{remaining} seconds remaining"
                self.last_voice_time = now
            elif progress >= 0.50 and not self.countdown_50_spoken:
                self.countdown_50_spoken = True
                remaining = max(1, int(self.target_time - self.hold_elapsed))
                voice_msg = f"{remaining} seconds remaining"
                self.last_voice_time = now

            if posture_issues:
                posture_voice = plank_issues_to_voice(posture_issues)
                if posture_voice and posture_voice != self.last_posture_voice:
                    if now - self.last_posture_voice_time >= POSTURE_COOLDOWN:
                        voice_msg = posture_voice
                        self.last_voice_time = now
                        self.last_posture_voice = posture_voice
                        self.last_posture_voice_time = now

            if not posture_issues:
                self.last_posture_voice = None

        return ExerciseResult(
            state="HOLDING",
            feedback=feedback,
            form_labels=posture_labels,
            form_issue_codes=posture_issues,
            voice_message=voice_msg,
            rep_count=self.rep_count,
            back_angle=self.elbow_angle,
            rep_completed=rep_completed,
            rep_had_issues=rep_had_issues,
            rep_issues=rep_issues,
            rep_avg_angle=self.hold_elapsed,
            left_angle=self.elbow_angle,
            right_angle=self.elbow_angle,
            avg_angle=self.elbow_angle
        )
