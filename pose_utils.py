"""
================================================================================
NeuralStride AI - Pose Detection Utilities
================================================================================

Provides MediaPipe Pose detector setup and standing posture validation.

CONTENTS:
    is_standing_posture()  — Checks if the person is in a standing position
    create_pose_detector() — Creates and configures the MediaPipe Pose detector

STANDING POSTURE CHECK:
    Used by exercise detectors (squat, lunge) to verify the person is standing
    upright before starting rep tracking. Prevents false rep detection when
    the person is sitting, lying down, or in an awkward position.

    How it works:
        - Compares average shoulder Y vs average hip Y
        - In a standing person, shoulders are ABOVE hips (lower Y value in
          image coordinates, where Y=0 is the top of the frame)
        - If shoulder Y >= hip Y, the person is upside-down or not standing

MEDIAPIPE POSE DETECTOR:
    Creates the Pose object that processes video frames and returns 33 body
    landmarks. Configuration:
        - static_image_mode=False: Treats input as video (faster, uses tracking)
        - model_complexity=1: Balanced accuracy/speed (0=lite, 1=full, 2=heavy)
        - smooth_landmarks=True: Reduces jitter between frames
        - min_detection_confidence=0.5: Minimum confidence to detect a person
        - min_tracking_confidence=0.5: Minimum confidence to track between frames

    Returns both the pose object and mp_pose module reference (needed for
    accessing PoseLandmark constants like LEFT_KNEE, RIGHT_HIP, etc.)

DEPENDENCIES:
    pip install mediapipe==0.10.14
    IMPORTANT: Must be 0.10.14 (newer versions removed mp.solutions.pose)
================================================================================
"""

import mediapipe as mp


def is_standing_posture(
    left_shoulder, right_shoulder,
    left_hip, right_hip,
    left_knee, right_knee,
    left_ankle, right_ankle
):
    """
    Check if the person is in a standing upright position.

    Used by exercise.check_ready() to verify the person is standing before
    allowing rep tracking to begin. Prevents false reps when the person is
    sitting, lying on the ground, or in a non-standing position.

    How it works:
        In image coordinates, Y increases downward (0 = top, 1 = bottom).
        A standing person has shoulders above their hips, so:
            avg_shoulder_y < avg_hip_y (shoulders have a LOWER y value)

    Args:
        left_shoulder:  Left shoulder landmark (NormalizedLandmark object)
        right_shoulder: Right shoulder landmark
        left_hip:       Left hip landmark
        right_hip:      Right hip landmark
        left_knee:      Left knee landmark (accepted but not used in check)
        right_knee:     Right knee landmark (accepted but not used in check)
        left_ankle:     Left ankle landmark (accepted but not used in check)
        right_ankle:    Right ankle landmark (accepted but not used in check)

    Returns:
        True if person appears to be standing (shoulders above hips).
        False if not standing (lying down, sitting, etc.)

    Note:
        Knee and ankle landmarks are accepted as parameters for API consistency
        with the exercise detectors (so all landmarks can be passed uniformly),
        but are not currently used in the standing check itself.
        Some exercises (push-up, plank) override check_ready() entirely.
    """
    # Average Y position of both shoulders and both hips
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    avg_hip_y = (left_hip.y + right_hip.y) / 2

    # If shoulders are at or below hip level, person is NOT standing
    # (in a standing position, shoulders must be above hips)
    if avg_shoulder_y >= avg_hip_y:
        return False

    return True


def create_pose_detector():
    """
    Create and configure the MediaPipe Pose detector.

    The Pose object processes RGB video frames and detects 33 body landmarks.
    It returns pose_landmarks containing normalized x, y, z, visibility for each
    landmark, plus POSE_CONNECTIONS for drawing the skeleton.

    Configuration:
        static_image_mode=False:
            Treats frames as a video stream (not independent images).
            Enables tracking mode — faster processing after initial detection.

        model_complexity=1:
            Model accuracy level. 0=lightning, 1=full (default), 2=heavy.
            1 is the best balance of speed and accuracy for real-time use.

        smooth_landmarks=True:
            Applies internal smoothing to reduce landmark jitter.
            Works alongside our AngleSmoother for extra stability.

        enable_segmentation=False:
            Disables body segmentation mask (we don't need it).
            Saves processing time.

        min_detection_confidence=0.5:
            Minimum confidence (0-1) to detect a person in the frame.
            0.5 = medium sensitivity — avoids false detections on background objects.

        min_tracking_confidence=0.5:
            Minimum confidence (0-1) to track the detected person between frames.
            0.5 = medium — allows tracking through mild occlusion.

    Returns:
        Tuple of (pose, mp_pose):
            pose:    Configured Pose object — call pose.process(rgb_frame)
            mp_pose: mp.solutions.pose module — needed for PoseLandmark constants
                     (e.g., mp_pose.PoseLandmark.LEFT_KNEE.value = 25)
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose
