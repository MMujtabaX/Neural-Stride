"""
================================================================================
NeuralStride AI - Display Utilities
================================================================================

Handles all on-screen visual output drawn on the OpenCV video frame.

CONTENTS:
    FeedbackDisplay  — Manages timed text feedback (shows message for N seconds)
    draw_workout_summary() — Creates the end-of-workout summary image
    draw_hud()       — Draws the heads-up display overlay on each frame

FEEDBACK DISPLAY:
    Exercise logic generates feedback text (e.g., "Fix your back lean").
    The FeedbackDisplay shows this text for FEEDBACK_DISPLAY_DURATION seconds,
    then clears it. New messages replace old ones immediately and reset the timer.

WORKOUT SUMMARY:
    After pressing 'q', a 640x480 dark blue image is shown with:
    - Date and duration
    - Total, clean, and form-issue rep counts
    - Form accuracy percentage
    - Form issues breakdown (which issues and how many reps)
    Press any key to close the summary window.

HUD (Heads-Up Display):
    Drawn every frame on top of the camera feed. Shows:
    - Quit instruction ("Press q to quit")
    - Camera view recommendation (side/diagonal)
    - Knee/elbow angles (left, right, average)
    - Current exercise state (UP/DOWN/MID with color coding)
    - Rep count
    - Form feedback text (if active)
    - Back angle (hidden for Push-Up and Plank — not relevant)
    - Guard labels (walking detection, active form issues)
    - FPS counter
    - Elapsed time

COLOR CODING (HUD):
    State UP   = green   (0, 255, 0)
    State DOWN = red     (0, 0, 255)
    State MID  = yellow  (0, 255, 255)
    State other = pink   (255, 100, 100)

BACK ANGLE COLORS:
    Good (< 30 deg)    = green   (0, 200, 0)
    Warning (30-45 deg) = orange  (0, 165, 255)
    Severe (> 45 deg)   = red     (0, 0, 255)

DEPENDENCIES:
    pip install opencv-python numpy
================================================================================
"""

import cv2
import numpy as np
import time

# How long feedback text stays on screen before clearing (seconds)
FEEDBACK_DISPLAY_DURATION = 3.0

# Back angle thresholds for color coding the back angle display
BACK_ANGLE_THRESHOLD = 30   # Degrees — above this, show warning (orange)
BACK_ANGLE_WARNING = 45    # Degrees — above this, show severe (red)


class FeedbackDisplay:
    """
    Manages timed on-screen feedback text.

    When an exercise detects a form issue, it returns feedback text
    (e.g., "Back lean detected"). This class shows that text for
    FEEDBACK_DISPLAY_DURATION seconds, then auto-clears it.

    New messages replace old ones immediately and reset the timer.
    If the same message is already showing, it is NOT refreshed
    (prevents timer reset spam from repeated detections).
    """

    def __init__(self, duration=FEEDBACK_DISPLAY_DURATION):
        """
        Args:
            duration: How many seconds to show feedback before clearing.
                      Default: 3.0 seconds.
        """
        self.duration = duration
        self.current_text = "None"   # "None" means no feedback to display
        self.start_time = None       # When the current text was first set

    def update(self, new_text, now=None):
        """
        Called every frame from the main loop to update feedback display.

        Args:
            new_text: Feedback string from exercise logic.
                      "None" = no feedback to show.
            now: Current time.time(). If None, gets current time internally.

        Returns:
            The text that should currently be displayed on screen.
            Returns "None" if the duration has expired or no feedback is active.
        """
        if now is None:
            now = time.time()

        # If a NEW (different) non-None message arrives, show it and start timer
        if new_text != "None" and new_text != self.current_text:
            self.current_text = new_text
            self.start_time = now

        # No feedback active — return immediately
        if self.current_text == "None":
            return "None"

        # Check if the display duration has expired
        if self.start_time is not None and (now - self.start_time) >= self.duration:
            self.current_text = "None"
            self.start_time = None

        return self.current_text

    def reset(self):
        """
        Clears any active feedback immediately.
        Called when the exercise is reset (person leaves frame, etc.)
        """
        self.current_text = "None"
        self.start_time = None


def draw_workout_summary(summary, session_date):
    """
    Creates a 640x480 pixel summary image shown after workout ends.

    Draws on a dark blue background with white/colored text showing:
    - Session date
    - Workout duration
    - Total reps, clean reps, form issue reps
    - Form accuracy percentage
    - Form issues breakdown (which issues occurred and on how many reps)

    Args:
        summary:      Dictionary from SessionTracker.get_summary()
                      Contains total_reps, clean_reps, form_issues, etc.
        session_date:  String with date/time for display header

    Returns:
        numpy array (480, 640, 3) — the summary image to display with cv2.imshow()
    """
    # Create a blank dark blue image (640x480)
    summary_img = np.zeros((480, 640, 3), dtype=np.uint8)
    summary_img[:] = (30, 30, 50)

    # Header bar — slightly lighter blue rectangle at the top
    cv2.rectangle(summary_img, (0, 0), (640, 60), (50, 50, 90), -1)
    cv2.putText(summary_img, "WORKOUT SUMMARY", (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(summary_img, "NeuralStride AI", (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Start drawing content below the header
    y = 90           # Current vertical position (moves down as we add lines)
    line_h = 32      # Pixel spacing between lines

    # Session info
    cv2.putText(summary_img, f"Date: {session_date}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    y += line_h
    cv2.putText(summary_img, f"Duration: {summary['duration']}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    y += line_h

    # Horizontal divider line
    cv2.line(summary_img, (20, y), (620, y), (80, 80, 120), 1)
    y += 15

    # Rep counts section
    cv2.putText(summary_img, f"Total Reps: {summary['total_reps']}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    y += line_h + 5
    cv2.putText(summary_img, f"Clean Reps: {summary['clean_reps']}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    y += line_h
    cv2.putText(summary_img, f"Form Issue Reps: {summary['form_issue_reps']}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    y += line_h
    cv2.putText(summary_img, f"Form Accuracy: {summary['clean_percentage']}%", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2)
    y += line_h + 5

    # Divider before form issues
    cv2.line(summary_img, (20, y), (620, y), (80, 80, 120), 1)
    y += 15

    # Form issues breakdown
    cv2.putText(summary_img, "Form Issues Breakdown:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    y += line_h

    if summary['form_issues'] != "None":
        # Long issue strings may not fit in one line, so chunk into 60-char pieces
        issues_text = summary['form_issues']
        while len(issues_text) > 0:
            chunk = issues_text[:60]
            issues_text = issues_text[60:]
            cv2.putText(summary_img, f"  {chunk}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            y += line_h - 5
    else:
        # No form issues at all — perfect workout
        cv2.putText(summary_img, "  Perfect form on all reps!", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y += line_h - 5

    # Bottom divider and exit prompt
    y += 10
    cv2.line(summary_img, (20, y), (620, y), (80, 80, 120), 1)
    y += 20
    cv2.putText(summary_img, "Press any key to exit", (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return summary_img


def draw_hud(frame, rep_count, squat_state, displayed_feedback,
             left_angle, right_angle, avg_angle,
             back_angle_smoothed,
             is_walking, standing_ok, posture_labels,
             left_knee_angle, right_knee_angle,
             fps_display, elapsed_str,
             exercise_name="Squat"):
    """
    Draws the heads-up display overlay on each video frame.

    All text is drawn directly on the OpenCV frame using cv2.putText().
    Some elements use a two-pass technique (black outline + white text) for
    better visibility against varying background colors.

    Args:
        frame:              The OpenCV video frame (numpy array) to draw on.
                           Modified in-place — no return value needed.
        rep_count:          Current number of completed reps.
        squat_state:        Current exercise state string ("UP", "DOWN", "MID", etc.)
        displayed_feedback:  Current feedback text (from FeedbackDisplay), or "None"
        left_angle:         Left knee/elbow angle (smoothed, degrees)
        right_angle:        Right knee/elbow angle (smoothed, degrees)
        avg_angle:          Average of left and right angles (smoothed, degrees)
        back_angle_smoothed: Back lean angle from vertical (smoothed, degrees)
        is_walking:         True if WalkingDetector detected walking
        standing_ok:        True if person is in correct starting position
        posture_labels:     List of active form issue labels (e.g., ["BACK_LEAN"])
        left_knee_angle:    Left knee angle raw (for angle display)
        right_knee_angle:   Right knee angle raw (for angle display)
        fps_display:         Frames per second counter (updated every 1 second)
        elapsed_str:        Elapsed time string "MM:SS"
        exercise_name:      Name of current exercise ("Squat", "Push-Up", etc.)
                           Used to decide which HUD elements to show/hide.
    """

    # --- State color coding ---
    # Each exercise state gets a distinct color for quick visual recognition
    if squat_state == "UP":
        state_color = (0, 255, 0)       # Green — standing position
    elif squat_state == "DOWN":
        state_color = (0, 0, 255)       # Red — deepest point of exercise
    elif squat_state == "MID":
        state_color = (0, 255, 255)     # Yellow — transitional
    else:
        state_color = (255, 100, 100)   # Pink — invalid or other state

    # --- Top-left: Quit instruction ---
    cv2.putText(frame, "Press q to quit", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # --- Camera view recommendation ---
    # Push-Up and Lunge work best with diagonal view; others with side view
    if exercise_name in ("Push-Up", "Lunge"):
        view_text = "Use diagonal view for best accuracy"
    else:
        view_text = "Use side view for best accuracy"
    # Two-pass rendering: black outline (offset by 2px) then colored text on top
    cv2.putText(frame, view_text, (152, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, view_text, (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # --- Angle display (left knee, right knee, average) ---
    cv2.putText(frame, f"Left: {int(left_knee_angle)}  Right: {int(right_knee_angle)}  Avg: {int(avg_angle)}",
                (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Exercise state ---
    cv2.putText(frame, f"State: {squat_state}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

    # --- Rep counter ---
    cv2.putText(frame, f"Reps: {rep_count}", (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # --- Form feedback text (only shown when active, not "None") ---
    if displayed_feedback != "None":
        cv2.putText(frame, f"Feedback: {displayed_feedback}", (5, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # --- Back angle (hidden for Push-Up and Plank — not relevant for horizontal exercises) ---
    if exercise_name not in ("Push-Up", "Plank"):
        # Color based on severity: green = good, orange = warning, red = severe
        back_color = (0, 0, 255) if back_angle_smoothed > BACK_ANGLE_WARNING else (0, 165, 255) if back_angle_smoothed > BACK_ANGLE_THRESHOLD else (0, 200, 0)
        cv2.putText(frame, f"Back Angle: {int(back_angle_smoothed)} deg", (5, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, back_color, 1)

    # --- Guard labels (combines walking detection + active form issues) ---
    protection_labels = []
    if is_walking:
        protection_labels.append("WALKING")    # WalkingDetector flagged movement
    if posture_labels:
        protection_labels.extend(posture_labels)  # Form issues from exercise logic

    # Only show the guard line if there are active labels
    if protection_labels:
        guard_text = ", ".join(protection_labels)
        cv2.putText(frame, f"Guard: {guard_text}", (5, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)

    # --- Top-right: FPS counter ---
    cv2.putText(frame, f"FPS: {fps_display}", (560, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # --- Bottom-right: Elapsed time ---
    cv2.putText(frame, f"Time: {elapsed_str}", (490, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
