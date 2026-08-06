"""
================================================================================
NeuralStride AI - Session Tracker & CSV Export
================================================================================

Tracks workout session data (reps, form issues, timing) and exports
everything to a CSV file when the workout ends.

CONTENTS:
    ISSUE_NAMES         — Maps internal issue codes to human-readable names
    SessionTracker      — Main class that logs reps and saves CSV

ISSUE CODES:
    Internal form issue codes used by exercise detectors:
        BACK_LEAN    — Torso leaning too far forward (shoulder-hip angle from vertical)
        SEVERE_LEAN  — Extreme back lean (worse threshold)
        KNEE_FORWARD — Front knee extending past toes (squat/lunge)
        HIP_DROP     — Hips uneven (left/right y-difference too large)
        SHALLOW      — Squat/push-up not going deep enough
        HIP_SAG      — Push-up: hips dropping below body line
        PIKE         — Push-up: hips too high above body line
        HANDS_NARROW — Push-up: grip too close together
        HANDS_WIDE   — Push-up: grip too far apart
        ELBOW_FLARE  — Push-up: elbows pointing too far outward

SESSION FLOW:
    1. SessionTracker() created at workout start — records start_time
    2. session.log_rep() called by main.py each time a rep is completed
       - Logs: rep number, timestamp, clean/issue flag, issues list, knee angle
       - Increments total_reps, clean_reps, or form_issue_reps counter
       - Builds form_issue_breakdown dict for the summary
    3. session.end_session() called when 'q' is pressed — records end_time
    4. session.get_summary() called to build summary dict for display
    5. session.save_to_csv() called to write the CSV file

CSV FORMAT:
    Saved to: sessions/session_DD-MM-YYYY_HH-MM-SSAM.csv
    Top section: one row per rep with columns:
        rep, timestamp_seconds, timestamp_mm_ss, clean, form_issues, avg_knee_angle
    Bottom section: session summary rows (key-value pairs):
        Date, Duration, Total Reps, Clean Reps, Form Issue Reps, Form Accuracy,
        Form Issues Breakdown, CSV File

TIMESTAMPS:
    Relative to session start (time.time() - start_time).
    CSV uses both raw seconds and MM:SS format for readability.

DEPENDENCIES:
    time, csv, os, datetime — all Python standard library (no pip install needed)
================================================================================
"""

import time
import csv
import os
from datetime import datetime


# Maps internal form issue codes to human-readable display names
# Used in CSV export and console output
# All exercises share this single dictionary — exercise-specific issues
# (SHALLOW, HIP_SAG, etc.) simply never appear in sessions of other exercises
ISSUE_NAMES = {
    "BACK_LEAN": "Back lean",
    "SEVERE_LEAN": "Severe back lean",
    "KNEE_FORWARD": "Front knee over toe",
    "HIP_DROP": "Hip drop",
    "SHALLOW": "Shallow squat",
    "HIP_SAG": "Hip sag",
    "PIKE": "Pike",
    "HANDS_NARROW": "Hands too narrow",
    "HANDS_WIDE": "Hands too wide",
    "ELBOW_FLARE": "Elbow flare",
    "KNEE_BENT": "Knee bent during leg raise",
    "TOO_FAST": "Rep too fast",
    "TOO_SLOW": "Rep too slow",
}

# Every issue code any detector may emit. ISSUE_NAMES covers the ones that appear in
# rep summaries; this set is the full vocabulary including frame-level-only codes.
# Used by landmark_exporter.py to fail loudly on an unrecognised code rather than
# silently dropping or mislabelling training data.
VALID_ISSUE_CODES = frozenset({
    *ISSUE_NAMES.keys(),
    "ASYMMETRIC",   # squat/push-up: limbs bending unevenly — frame-level guard, never logged to a rep
})


class SessionTracker:
    """
    Records workout session data — reps, timing, form issues — and
    exports everything to a CSV file when the session ends.

    Called from main.py:
        session = SessionTracker()           # At workout start
        session.log_rep(...)                  # Each time a rep completes
        session.end_session()                # When 'q' is pressed
        summary = session.get_summary()       # Build summary dict
        path = session.save_to_csv()          # Write CSV file
    """

    def __init__(self):
        """Initialize a new session. Records the start time immediately."""
        self.start_time = time.time()       # Workout start timestamp (epoch seconds)
        self.end_time = None                # Set when end_session() is called
        self.reps = []                      # List of rep data dicts (one per completed rep)
        self.total_reps = 0                  # Counter: total completed reps
        self.clean_reps = 0                  # Counter: reps with no form issues
        self.form_issue_reps = 0             # Counter: reps with at least one form issue
        self.form_issue_breakdown = {}       # Dict: {issue_code: count} e.g. {"BACK_LEAN": 3}

    def log_rep(self, rep_number, had_form_issues, issues, avg_knee_angle):
        """
        Log a completed rep to the session.

        Called by main.py when exercise.update() returns rep_completed=True.
        Each rep is stored as a dict and added to self.reps list.

        Args:
            rep_number:      Sequential rep number (1, 2, 3, ...)
            had_form_issues:  True if any form issues were detected during this rep
            issues:           List of issue code strings (e.g., ["BACK_LEAN", "KNEE_FORWARD"])
                              Empty list if rep was clean
            avg_knee_angle:   Average knee/elbow angle during the rep (degrees)
        """
        self.total_reps += 1

        # Build the rep data dict with all info needed for CSV export
        rep_data = {
            "rep": rep_number,
            "timestamp": time.time() - self.start_time,  # Seconds since workout started
            "clean": not had_form_issues,                # True = perfect form
            "issues": issues.copy() if issues else [],   # Copy to avoid mutation
            "avg_knee_angle": round(avg_knee_angle, 1)   # Rounded to 1 decimal
        }
        self.reps.append(rep_data)

        # Update counters based on whether the rep had form issues
        if had_form_issues:
            self.form_issue_reps += 1
            # Count each issue type for the breakdown summary
            for issue in issues:
                if issue not in self.form_issue_breakdown:
                    self.form_issue_breakdown[issue] = 0
                self.form_issue_breakdown[issue] += 1
        else:
            self.clean_reps += 1

    def get_elapsed_seconds(self):
        """
        Returns elapsed session time in seconds.
        Before end_session(): returns live duration (current time - start).
        After end_session():  returns fixed duration (end - start).
        """
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def get_elapsed_str(self):
        """
        Returns elapsed session time as a formatted "MM:SS" string.
        Used in the HUD timer display and CSV export.
        """
        elapsed = self.get_elapsed_seconds()
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        return f"{minutes:02d}:{seconds:02d}"

    def end_session(self):
        """Record the session end time. Called when user presses 'q'."""
        self.end_time = time.time()

    def get_summary(self):
        """
        Build a summary dictionary with all session stats.
        Used by the console printout and the workout summary image.

        Returns:
            Dict with keys: total_reps, clean_reps, form_issue_reps,
            clean_percentage, duration, duration_seconds, form_issues, rep_details
        """
        elapsed = self.get_elapsed_seconds()
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60

        # Build the form issues breakdown string
        # e.g., "Back lean: 3, Knee over toe: 2"
        form_issues_str = ""
        if self.form_issue_breakdown:
            parts = []
            for issue, count in self.form_issue_breakdown.items():
                name = ISSUE_NAMES.get(issue, issue)  # Convert code to readable name
                parts.append(f"{name}: {count}")
            form_issues_str = ", ".join(parts)
        else:
            form_issues_str = "None"

        # Calculate form accuracy: (clean reps / total reps) * 100
        clean_pct = 0
        if self.total_reps > 0:
            clean_pct = round((self.clean_reps / self.total_reps) * 100, 1)

        return {
            "total_reps": self.total_reps,
            "clean_reps": self.clean_reps,
            "form_issue_reps": self.form_issue_reps,
            "clean_percentage": clean_pct,
            "duration": f"{minutes:02d}:{seconds:02d}",
            "duration_seconds": round(elapsed, 1),
            "form_issues": form_issues_str,
            "rep_details": self.reps
        }

    def save_to_csv(self, output_dir="sessions"):
        """
        Export all session data to a CSV file.

        File format:
            Top section: one row per rep with headers (rep, timestamp, clean, etc.)
            Blank row separator
            Bottom section: session summary as key-value rows

        Args:
            output_dir: Directory to save the CSV file. Default: "sessions/"

        Returns:
            Filepath string of the saved CSV (used by main.py to print path)
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate filename with 12-hour timestamp format
        # Example: session_15-01-2025_02-30-00PM.csv
        timestamp_str = datetime.now().strftime("%d-%m-%Y_%I-%M-%S%p")
        filename = f"session_{timestamp_str}.csv"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # --- Rep data section ---
            # Header row defining the columns
            writer.writerow([
                "rep",
                "timestamp_seconds",
                "timestamp_mm_ss",
                "clean",
                "form_issues",
                "avg_knee_angle"
            ])

            # One row per completed rep
            for rep in self.reps:
                # Convert issue codes to readable names, joined with "; "
                issues_str = "; ".join(
                    ISSUE_NAMES.get(i, i) for i in rep["issues"]
                ) if rep["issues"] else "None"

                # Format timestamp as both raw seconds and MM:SS
                ts = rep["timestamp"]
                mm = int(ts) // 60
                ss = int(ts) % 60
                ts_str = f"{mm:02d}:{ss:02d}"

                writer.writerow([
                    rep["rep"],
                    round(ts, 2),
                    ts_str,
                    "Yes" if rep["clean"] else "No",
                    issues_str,
                    rep["avg_knee_angle"]
                ])

            # Blank row separator between rep data and summary
            writer.writerow([])

            # --- Session summary section ---
            summary = self.get_summary()
            writer.writerow(["--- SESSION SUMMARY ---"])
            writer.writerow(["Date", datetime.now().strftime("%Y-%m-%d %I:%M %p")])
            writer.writerow(["Duration", summary["duration"]])
            writer.writerow(["Total Reps", summary["total_reps"]])
            writer.writerow(["Clean Reps", summary["clean_reps"]])
            writer.writerow(["Form Issue Reps", summary["form_issue_reps"]])
            writer.writerow(["Form Accuracy", f"{summary['clean_percentage']}%"])
            writer.writerow(["Form Issues Breakdown", summary["form_issues"]])
            writer.writerow(["CSV File", filename])

        return filepath
