"""
================================================================================
NeuralStride AI - Angle Calculation Utilities
================================================================================

Provides angle calculation and smoothing functions used by all exercise
detectors for measuring joint angles and filtering out noise.

CONTENTS:
    calculate_angle()  — Calculates the angle between three points using Law of Cosines
    AngleSmoother       — Rolling average smoother for paired joint angles (left/right)
    SingleSmoother      — Rolling average smoother for single values (back angle, etc.)

ANGLE CALCULATION (Law of Cosines):
    Given three points A, B, C where B is the vertex (joint):
        - Calculate vectors BA = A - B and BC = C - B
        - Find the angle between these vectors using dot product:
            cos(theta) = (BA . BC) / (|BA| * |BC|)
        - Return theta in degrees

    Example for knee angle:
        A = hip landmark, B = knee landmark, C = ankle landmark
        Returns the knee flexion angle (180 = straight leg, 90 = bent knee)

SMOOTHING (Rolling Average):
    MediaPipe landmark positions jitter slightly frame-to-frame, causing
    calculated angles to fluctuate by 5-15 degrees even when the person
    is perfectly still. The smoothers fix this by averaging over the last
    N frames (default: 5 frames).

    Why smoothing matters:
        Without smoothing: angle jumps between 85-100 degrees (random noise)
        With smoothing:    angle stays stable at ~92 degrees (true value)

    Buffer size of 5 at 30fps = 166ms window — responsive enough for
    real-time feedback but smooth enough to kill noise.

USAGE IN EXERCISE DETECTORS:
    squat.py:
        angle_smoother = AngleSmoother()
        angle_smoother.update(left_knee_angle, right_knee_angle)
        left, right = angle_smoother.get_smoothed()
        avg = angle_smoother.get_smoothed_avg()

    lunge.py (single smoother for back angle):
        back_smoother = SingleSmoother()
        back_smoother.update(back_angle)
        smoothed = back_smoother.get_smoothed()

DEPENDENCIES:
    pip install numpy
================================================================================
"""

import numpy as np
import collections

# Number of frames to average over for angle smoothing
# 5 frames at 30fps = ~167ms window — good balance of responsiveness vs stability
SMOOTHING_BUFFER_SIZE = 5


def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point B formed by points A, B, C.

    Uses the Law of Cosines:
        Given vectors BA and BC from the vertex B:
        cos(angle) = dot(BA, BC) / (|BA| * |BC|)

    This is the standard method for measuring joint angles from pose landmarks.

    Args:
        a: First point (e.g., hip) — tuple/list of (x, y) or (x, y, z)
        b: Vertex point (e.g., knee) — the joint where the angle is measured
        c: Third point (e.g., ankle) — tuple/list of (x, y) or (x, y, z)

    Returns:
        Angle in degrees (0-180). Returns 0 if points are degenerate
        (e.g., B coincides with A or C, making the angle undefined).

    Examples:
        # Knee angle (hip -> knee -> ankle)
        knee_angle = calculate_angle(hip, knee, ankle)  # ~90 when squatting

        # Elbow angle (shoulder -> elbow -> wrist)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)  # ~90 in push-up

        # Back angle (shoulder -> hip -> vertical reference)
        back_angle = calculate_angle(shoulder, hip, [hip_x, hip_y + 1])
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Create vectors from vertex B to points A and C
    ba = a - b   # Vector from B to A
    bc = c - b   # Vector from B to C

    # Calculate magnitudes of both vectors
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    # If either vector has zero length, the angle is undefined
    # (happens when landmarks overlap, e.g., person facing directly at camera)
    if denominator == 0:
        return 0

    # Calculate cosine of the angle using dot product
    cosine_angle = np.dot(ba, bc) / denominator

    # Clamp to [-1, 1] to prevent arccos domain errors from floating point noise
    # (sometimes cosine ends up as 1.0000001 or -1.0000001)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Convert from radians to degrees
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


class AngleSmoother:
    """
    Rolling average smoother for paired joint angles (left and right).

    Maintains two separate buffers (one for left side, one for right side).
    Each frame, the new angles are appended and the buffer automatically
    drops the oldest value (deque with maxlen). The smoothed value is the
    mean of all values currently in the buffer.

    Used by: squat.py, pushup.py, lunge.py for knee/elbow angle smoothing.

    Args:
        buffer_size: Number of frames to average over. Default: 5 (SMOOTHING_BUFFER_SIZE)
    """

    def __init__(self, buffer_size=SMOOTHING_BUFFER_SIZE):
        self.left_buffer = collections.deque(maxlen=buffer_size)    # Left side angle history
        self.right_buffer = collections.deque(maxlen=buffer_size)   # Right side angle history

    def update(self, left_angle, right_angle):
        """
        Add new angle readings to the buffers.

        Args:
            left_angle:  Current frame's left joint angle (degrees)
            right_angle: Current frame's right joint angle (degrees)
        """
        self.left_buffer.append(left_angle)
        self.right_buffer.append(right_angle)

    def get_smoothed(self):
        """
        Returns the smoothed (averaged) left and right angles.

        Returns:
            Tuple of (left_smoothed, right_smoothed) in degrees.
            Returns (0, 0) if no data has been added yet.
        """
        if len(self.left_buffer) == 0:
            return 0, 0
        return (np.mean(self.left_buffer), np.mean(self.right_buffer))

    def get_smoothed_avg(self):
        """
        Returns the smoothed average of left and right angles.

        Used by exercise detectors to get a single representative angle
        for state detection and form checking.

        Returns:
            Float: average of smoothed left and right angles in degrees.
        """
        left, right = self.get_smoothed()
        return (left + right) / 2

    def reset(self):
        """Clear all buffered data. Called when exercise is reset."""
        self.left_buffer.clear()
        self.right_buffer.clear()


class SingleSmoother:
    """
    Rolling average smoother for a single value.

    Same concept as AngleSmoother but for one value instead of a pair.
    Used for smoothing values like back angle, hip offset, etc.

    Args:
        buffer_size: Number of frames to average over. Default: 5 (SMOOTHING_BUFFER_SIZE)
    """

    def __init__(self, buffer_size=SMOOTHING_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=buffer_size)

    def update(self, value):
        """
        Add a new value to the buffer.

        Args:
            value: Current frame's value (e.g., back angle in degrees)
        """
        self.buffer.append(value)

    def get_smoothed(self):
        """
        Returns the smoothed (averaged) value.

        Returns:
            Float: mean of all values in the buffer, or 0 if empty.
        """
        if len(self.buffer) == 0:
            return 0
        return np.mean(self.buffer)

    def reset(self):
        """Clear all buffered data. Called when exercise is reset."""
        self.buffer.clear()
