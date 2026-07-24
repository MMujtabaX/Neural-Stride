"""
================================================================================
NeuralStride AI - Base Exercise Class & Exercise Result
================================================================================

Defines the interface (contract) that all exercise detectors must follow.
Every exercise (Squat, Push-Up, Plank, Lunge) extends BaseExercise.

CONTENTS:
    ExerciseResult  — Dataclass holding the result of one exercise.update() call
    BaseExercise    — Abstract base class with default method stubs

EXERCISE RESULT FIELDS:
    Every frame, exercise.update() returns an ExerciseResult with:
        state:           Current state string ("UP", "DOWN", "MID", "HOLDING", etc.)
        feedback:        Form feedback text to display (e.g., "Fix your back lean")
        form_labels:     List of active issue codes (e.g., ["BACK_LEAN", "HIP_DROP"])
        is_walking:      True if WalkingDetector detected walking in frame
        voice_message:   Text to speak via VoiceEngine (None = nothing to say)
        rep_count:       Total completed reps so far
        back_angle:      Current smoothed back lean angle (degrees from vertical)
        rep_completed:   True if a rep was just completed this frame
        rep_had_issues:  True if the just-completed rep had form issues
        rep_issues:      List of issue codes for the just-completed rep
        rep_avg_angle:   Average knee/elbow angle during the just-completed rep
        left_angle:      Left side angle (smoothed, degrees)
        right_angle:     Right side angle (smoothed, degrees)
        avg_angle:       Average of left and right angles (smoothed, degrees)

BASE EXERCISE METHODS:
    get_name()     — Returns exercise name string
    reset()        — Resets all state (rep count, buffers, etc.)
    check_ready()  — Checks if person is in correct starting position
    is_tracking()  — Returns True if already mid-exercise (skip ready check)
    update()       — Main per-frame logic: state detection, form checks, rep counting
    get_rep_count()— Returns current rep count

ADDING A NEW EXERCISE:
    1. Create detectors/new_exercise.py extending BaseExercise
    2. Implement check_ready(), update(), and override any defaults
    3. Register in detectors/__init__.py EXERCISES dict
    4. Add to main.py docstring file list
================================================================================
"""

from dataclasses import dataclass, field


@dataclass
class ExerciseResult:
    """
    Holds all output data from a single exercise.update() call.

    Returned by every exercise detector every frame. Contains the current
    state, feedback, voice message, rep data, and angles for the HUD display.

    All exercise detectors populate this and return it to main.py, which
    reads the fields to update the display, voice, and session tracker.

    Default values represent an "inactive" state (no detection, no feedback).
    Exercise detectors set only the fields relevant to their logic.
    """
    state: str = "INVALID"              # Current exercise state ("UP", "DOWN", "MID", "HOLDING", etc.)
    feedback: str = "None"               # Form feedback text for on-screen display
    form_labels: list = field(default_factory=list)  # Active form issue codes (e.g., ["BACK_LEAN"])
    is_walking: bool = False              # True if walking detected in this frame
    voice_message: str = None             # Text to speak (None = no speech this frame)
    rep_count: int = 0                    # Total completed reps so far
    back_angle: float = 0                 # Smoothed back lean angle from vertical (degrees)
    rep_completed: bool = False           # True if a rep finished THIS frame
    rep_had_issues: bool = False           # True if the just-finished rep had form issues
    rep_issues: list = field(default_factory=list)  # Issue codes for the just-finished rep
    rep_avg_angle: float = 0              # Average angle during the just-finished rep
    left_angle: float = 0                  # Left side joint angle (smoothed, degrees)
    right_angle: float = 0                 # Right side joint angle (smoothed, degrees)
    avg_angle: float = 0                  # Average of left and right (smoothed, degrees)


class BaseExercise:
    """
    Abstract base class that all exercise detectors must extend.

    Provides default method stubs. Each exercise (Squat, Push-Up, Plank, Lunge)
    overrides the methods it needs with its own implementation.

    The main loop in main.py calls these methods in order:
        1. check_ready()  — Is the person in starting position?
        2. update()       — Process this frame (angles, state, form, reps)
        3. reset()        — If person leaves frame or presses reset
        4. get_rep_count()— For display purposes
    """

    def __init__(self):
        """Initialize with zero rep count. Subclasses add their own state."""
        self.rep_count = 0

    def get_name(self):
        """Returns the exercise name string (e.g., "Squat", "Push-Up")."""
        return "Base"

    def reset(self):
        """
        Reset all exercise state to initial values.
        Called when the person leaves the frame or exercise is restarted.
        Subclasses should override to clear their own buffers and counters.
        """
        self.rep_count = 0

    def check_ready(self, landmarks, mp_pose):
        """
        Check if the person is in the correct starting position.

        Called every frame when the person is detected but not yet tracking.
        Returns (True, "") to start tracking, or (False, message) to wait.

        Args:
            landmarks: MediaPipe pose landmarks list (33 NormalizedLandmark objects)
            mp_pose:   mp.solutions.pose module (for PoseLandmark constants)

        Returns:
            Tuple of (is_ready: bool, message: str)
            is_ready=True: person is in position, start tracking
            is_ready=False: person not ready, message is spoken/displayed
        """
        return False, "Get into position"

    def is_tracking(self):
        """
        Returns True if already mid-exercise (rep tracking in progress).

        When True, main.py skips the check_ready() call so the exercise
        isn't interrupted during movement. Set by the exercise detector
        when the first rep state (usually DOWN) is first reached.
        """
        return False

    def update(self, landmarks, mp_pose, w, h, now):
        """
        Main per-frame processing logic. Called every frame while tracking.

        Responsible for:
            - Calculating joint angles
            - Detecting state transitions (UP/DOWN/MID)
            - Running form checks
            - Counting reps
            - Generating feedback and voice messages

        Args:
            landmarks: MediaPipe pose landmarks list (33 NormalizedLandmark objects)
            mp_pose:   mp.solutions.pose module
            w:         Frame width in pixels (e.g., 640)
            h:         Frame height in pixels (e.g., 480)
            now:       Current time.time() for cooldown timers

        Returns:
            ExerciseResult dataclass with all frame data
        """
        return ExerciseResult()

    def get_rep_count(self):
        """Returns the current number of completed reps."""
        return self.rep_count

    def get_idle_state(self):
        """
        State string meaning "between reps, not currently performing".

        Used by rep_window.RepWindowTracker to know when a rep window should
        open (state leaves this value) and close without completion (state
        returns to this value without rep_completed firing). Squat, Push-Up
        and Lunge all use the UP/MID/DOWN vocabulary and share this default;
        Plank overrides it to "BROKEN" (see detectors/plank.py).
        """
        return "UP"
