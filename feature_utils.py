"""
================================================================================
NeuralStride AI - Biomechanical Feature Extraction (Training-Data Layer)
================================================================================

Computes a fixed-length, fixed-order, scale-invariant feature vector from a
single frame of MediaPipe Pose landmarks. The same 22-value vector is produced
for every exercise (Squat, Push-Up, Plank, Lunge) so a single CNN-LSTM
training pipeline can consume all of them without per-exercise branching.

PURPOSE:
    Phase 1 (this module) only computes and exposes features. No model code,
    no windowing, no dataset building lives here — see landmark_exporter.py
    for how these vectors are written alongside the raw 33 landmarks.

WHY NOT RAW COORDINATES:
    Raw (x, y, z) landmark coordinates are camera-position and scale dependent
    — the same squat performed closer to the camera or off-center produces a
    different raw vector. Joint angles (degrees) are already scale invariant.
    The remaining geometry features (hip drop, torso length, etc.) are divided
    by hip_width so they stay comparable across body sizes and camera distances.

FEATURE VECTOR LAYOUT (FEATURE_NAMES, in order):
    0-9   : Joint angles (degrees, 0-180) via calculate_angle() from angle_utils.
            Knees, hips, elbows, shoulders, ankles, left and right.
    10    : back_lean_angle — mid-shoulder to mid-hip vector vs vertical-up.
    11-13 : Symmetry — absolute left/right differences of knee/hip/elbow angles.
    14-21 : Scale-invariant geometry, every value divided by hip_width (or
            shoulder_width if hip_width is degenerate). Features 14 and 15 are
            SIGNED (left-right differences) — see MIRROR_PAIRS note below.

DEGENERATE INPUT:
    If hip_width < 1e-6, fall back to shoulder_width. If that is also
    degenerate, return a zero vector with quality = 0.0. calculate_angle()
    already returns 0 for degenerate geometry (left as-is). Every geometry
    feature is clamped to [-10, 10] so a single bad frame cannot introduce a
    training-breaking outlier. The vector is asserted finite before return.

QUALITY SCORE:
    quality = min(visibility) over CORE_LANDMARKS (shoulders, hips, knees,
    ankles). Returned separately from the feature vector — the training
    pipeline uses it to drop unreliable frames, it is not itself a feature.

MIRROR AUGMENTATION (MIRROR_PAIRS, mirror_features()):
    Training data benefits from left/right mirrored duplicates. MIRROR_PAIRS
    lists feature indices that must be SWAPPED when mirroring a sample
    (e.g. left_knee_angle <-> right_knee_angle). Features 14 (hip_drop) and 15
    (shoulder_tilt) are signed left-minus-right differences — mirroring must
    NEGATE these rather than swap them, since there is nothing on "the other
    side" to swap with. mirror_features() applies both rules correctly; do not
    hand-roll this elsewhere.

USAGE:
    from feature_utils import extract_features
    features, quality = extract_features(landmarks, mp_pose)

DEPENDENCIES:
    numpy (via angle_utils), math — no new pip installs required.
================================================================================
"""

import math

from angle_utils import calculate_angle

FEATURE_NAMES = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "back_lean_angle",
    "knee_angle_diff",
    "hip_angle_diff",
    "elbow_angle_diff",
    "hip_drop",
    "shoulder_tilt",
    "left_knee_toe_offset_x",
    "right_knee_toe_offset_x",
    "hip_height_over_ankle",
    "torso_length",
    "ankle_separation",
    "wrist_separation",
]

FEATURE_COUNT = len(FEATURE_NAMES)  # 22

# Feature-vector index pairs to swap when mirroring a sample left <-> right
# for data augmentation. Signed features 14 (hip_drop) and 15 (shoulder_tilt)
# are NOT in this list — they must be negated instead, see mirror_features().
MIRROR_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (16, 17)]

# Signed features that must be negated (not swapped) on mirror.
_MIRROR_NEGATE_INDICES = (14, 15)

# Landmark indices used for the per-frame quality score:
# shoulders, hips, knees, ankles.
CORE_LANDMARKS = [11, 12, 23, 24, 25, 26, 27, 28]

_GEOMETRY_CLAMP_MIN = -10.0
_GEOMETRY_CLAMP_MAX = 10.0
_DEGENERATE_EPS = 1e-6


def _pt(lm):
    """Normalized (x, y) tuple for a landmark — resolution independent."""
    return (lm.x, lm.y)


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp(value):
    if value < _GEOMETRY_CLAMP_MIN:
        return _GEOMETRY_CLAMP_MIN
    if value > _GEOMETRY_CLAMP_MAX:
        return _GEOMETRY_CLAMP_MAX
    return value


def extract_features(landmarks, mp_pose):
    """
    Compute the 22-value biomechanical feature vector for one frame.

    Args:
        landmarks: MediaPipe pose_landmarks.landmark list (33 NormalizedLandmark
                   objects), same object passed to every exercise detector.
        mp_pose:   mp.solutions.pose module (for PoseLandmark constants).

    Returns:
        Tuple of (feature_vector: list[float] of length FEATURE_COUNT,
                  quality_score: float in 0-1).
    """
    lm = mp_pose.PoseLandmark

    left_shoulder = _pt(landmarks[lm.LEFT_SHOULDER.value])
    right_shoulder = _pt(landmarks[lm.RIGHT_SHOULDER.value])
    left_elbow = _pt(landmarks[lm.LEFT_ELBOW.value])
    right_elbow = _pt(landmarks[lm.RIGHT_ELBOW.value])
    left_wrist = _pt(landmarks[lm.LEFT_WRIST.value])
    right_wrist = _pt(landmarks[lm.RIGHT_WRIST.value])
    left_hip = _pt(landmarks[lm.LEFT_HIP.value])
    right_hip = _pt(landmarks[lm.RIGHT_HIP.value])
    left_knee = _pt(landmarks[lm.LEFT_KNEE.value])
    right_knee = _pt(landmarks[lm.RIGHT_KNEE.value])
    left_ankle = _pt(landmarks[lm.LEFT_ANKLE.value])
    right_ankle = _pt(landmarks[lm.RIGHT_ANKLE.value])
    left_foot_index = _pt(landmarks[lm.LEFT_FOOT_INDEX.value])
    right_foot_index = _pt(landmarks[lm.RIGHT_FOOT_INDEX.value])

    hip_width = _euclidean(left_hip, right_hip)
    if hip_width < _DEGENERATE_EPS:
        hip_width = _euclidean(left_shoulder, right_shoulder)
        if hip_width < _DEGENERATE_EPS:
            return [0.0] * FEATURE_COUNT, 0.0

    # --- Joint angles (0-10) ---
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
    left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
    right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot_index)

    mid_hip = _midpoint(left_hip, right_hip)
    mid_shoulder = _midpoint(left_shoulder, right_shoulder)
    mid_ankle = _midpoint(left_ankle, right_ankle)

    vertical_ref = (mid_hip[0], mid_hip[1] - 1.0)
    back_lean_angle = calculate_angle(mid_shoulder, mid_hip, vertical_ref)

    # --- Symmetry (11-13) ---
    knee_angle_diff = abs(left_knee_angle - right_knee_angle)
    hip_angle_diff = abs(left_hip_angle - right_hip_angle)
    elbow_angle_diff = abs(left_elbow_angle - right_elbow_angle)

    # --- Scale-invariant geometry (14-21), all divided by hip_width ---
    hip_drop = _clamp((left_hip[1] - right_hip[1]) / hip_width)
    shoulder_tilt = _clamp((left_shoulder[1] - right_shoulder[1]) / hip_width)
    left_knee_toe_offset_x = _clamp((left_knee[0] - left_foot_index[0]) / hip_width)
    right_knee_toe_offset_x = _clamp((right_knee[0] - right_foot_index[0]) / hip_width)
    hip_height_over_ankle = _clamp((mid_ankle[1] - mid_hip[1]) / hip_width)
    torso_length = _clamp(_euclidean(mid_shoulder, mid_hip) / hip_width)
    ankle_separation = _clamp(abs(left_ankle[0] - right_ankle[0]) / hip_width)
    wrist_separation = _clamp(abs(left_wrist[0] - right_wrist[0]) / hip_width)

    features = [
        left_knee_angle,
        right_knee_angle,
        left_hip_angle,
        right_hip_angle,
        left_elbow_angle,
        right_elbow_angle,
        left_shoulder_angle,
        right_shoulder_angle,
        left_ankle_angle,
        right_ankle_angle,
        back_lean_angle,
        knee_angle_diff,
        hip_angle_diff,
        elbow_angle_diff,
        hip_drop,
        shoulder_tilt,
        left_knee_toe_offset_x,
        right_knee_toe_offset_x,
        hip_height_over_ankle,
        torso_length,
        ankle_separation,
        wrist_separation,
    ]

    # Defensive sanitation — never hand NaN/inf to a training pipeline.
    features = [float(v) if math.isfinite(v) else 0.0 for v in features]
    assert len(features) == FEATURE_COUNT
    assert all(math.isfinite(v) for v in features)

    quality = min(landmarks[i].visibility for i in CORE_LANDMARKS)

    return features, quality


def mirror_features(vec):
    """
    Return a left/right-mirrored copy of a feature vector for augmentation.

    Swaps every pair in MIRROR_PAIRS (e.g. left_knee_angle <-> right_knee_angle)
    and negates the signed left-right difference features (hip_drop,
    shoulder_tilt) instead of swapping them, since mirroring flips which side
    is "ahead" without there being a separate value to swap with.
    """
    mirrored = list(vec)
    for i, j in MIRROR_PAIRS:
        mirrored[i], mirrored[j] = vec[j], vec[i]
    for i in _MIRROR_NEGATE_INDICES:
        mirrored[i] = -vec[i]
    return mirrored
