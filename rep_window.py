"""
================================================================================
NeuralStride AI - Repetition Window Tracker
================================================================================

Tracks the frame boundaries of each repetition so training windows can align
to real reps instead of an arbitrary fixed-length clock. A squat rep spans
roughly 60-150 frames — slicing a fixed 30-frame window would cut reps
arbitrarily, so this module records an explicit start_frame -> end_frame span
per rep and hands it to landmark_exporter.log_rep_event() for the JSONL file.

WHY A SEPARATE MODULE (not inside the detectors):
    squat.py, pushup.py and lunge.py use UP/MID/DOWN states; plank.py uses
    HOLDING/BROKEN. Tracking rep boundaries here — driven only by the public
    ExerciseResult each exercise already returns — means zero detector logic
    has to change to get this data. The only detector-side change permitted
    is BaseExercise.get_idle_state() (see detectors/base.py) plus PlankExercise
    overriding it to "BROKEN".

REP LIFECYCLE:
    - OPEN:      no rep currently tracked, and result.state is neither the
                 exercise's idle_state nor "INVALID" -> a new rep starts.
    - COMPLETED: a rep is open and result.rep_completed is True -> closed with
                 completed=True, using the final labels straight off `result`
                 (rep_had_issues, rep_issues, rep_avg_angle).
    - ABORTED:   a rep is open, state has been back at idle_state for more
                 than ABORT_GRACE_FRAMES consecutive frames, and rep_completed
                 still hasn't fired -> closed with completed=False,
                 aborted=True, clean=None, issues=[]. Aborted reps are KEPT,
                 not discarded — partial/shallow attempts are valuable
                 negative training examples.
    - abort():   called by main.py when the person leaves the frame, check_ready
                 fails, or the session ends, so a rep window is never left
                 spanning a gap where nothing was tracked. This is an explicit,
                 immediate close — it does not wait out the grace window.

WHY THE GRACE WINDOW (ABORT_GRACE_FRAMES):
    squat.py / lunge.py / pushup.py fire rep_completed on the exact same frame
    their state flips back to idle_state, so update() sees rep_completed=True
    before it ever checks state==idle_state — the grace window is never
    consulted for them. legraise.py instead requires MIN_UP_FRAMES consecutive
    idle frames after the state flip before it commits to rep_completed=True
    (its debounce against flicker). Closing the window the instant state==
    idle_state, before that decision lands, aborted every legraise rep with no
    label and silently dropped the real rep_completed event a few frames
    later (current_rep was already None by then). The grace window gives a
    detector like that room to still finish deciding; if nothing arrives
    within ABORT_GRACE_FRAMES idle frames, the window closes as aborted same
    as before, just a few frames later than it used to for genuinely
    abandoned attempts.

update() returns None on any frame where no rep event occurred (the common
case — most frames are mid-rep or idle with nothing to report).
"""

ABORT_GRACE_FRAMES = 6


class RepWindowTracker:
    """
    Tracks one open repetition window at a time and emits rep event dicts
    at completion or abandonment. See module docstring for the lifecycle.

    Usage (from main.py):
        tracker = RepWindowTracker(idle_state=exercise.get_idle_state())
        # ... once per frame, after exercise.update() ...
        rep_event = tracker.update(frame_number, timestamp, result)
        if rep_event:
            rep_event["exercise"] = exercise_name
            landmark_exporter.log_rep_event(rep_event)
        # ... when the person is not ready / not detected / session ends ...
        rep_event = tracker.abort(frame_number, timestamp)
    """

    def __init__(self, idle_state):
        """
        Args:
            idle_state: State string meaning "between reps, not currently
                        performing" for this exercise (exercise.get_idle_state()).
        """
        self.idle_state = idle_state
        self.current_rep = None  # dict with rep_id/start_frame/start_timestamp, or None
        self._next_rep_id = 1
        self._idle_frame_count = 0  # consecutive idle frames seen while current_rep is open and not yet completed

    def update(self, frame_number, timestamp, result):
        """
        Process one frame's ExerciseResult. Call once per frame, after
        exercise.update() has run.

        Returns:
            A rep event dict (see landmark_exporter.log_rep_event docstring
            for the exact shape) if a rep just completed or aborted this
            frame, otherwise None.
        """
        if self.current_rep is None:
            if result.state != self.idle_state and result.state != "INVALID":
                self.current_rep = {
                    "rep_id": self._next_rep_id,
                    "start_frame": frame_number,
                    "start_timestamp": timestamp,
                }
                self._next_rep_id += 1
                self._idle_frame_count = 0
            return None

        if result.rep_completed:
            self._idle_frame_count = 0
            return self._close(
                frame_number, timestamp,
                completed=True, aborted=False,
                clean=not result.rep_had_issues,
                issues=list(result.rep_issues),
                avg_angle=result.rep_avg_angle,
            )

        if result.state == self.idle_state:
            self._idle_frame_count += 1
            if self._idle_frame_count >= ABORT_GRACE_FRAMES:
                self._idle_frame_count = 0
                return self._close(
                    frame_number, timestamp,
                    completed=False, aborted=True,
                    clean=None, issues=[], avg_angle=0,
                )
            return None

        self._idle_frame_count = 0
        return None

    def abort(self, frame_number, timestamp):
        """
        Force-close any open rep as aborted. Call when the person leaves the
        frame, check_ready() fails and exercise.reset() runs, or the session
        is ending — anywhere a rep window must not be left spanning a gap.

        Returns:
            The rep event dict if a rep was open and closed, otherwise None.
        """
        if self.current_rep is None:
            return None
        self._idle_frame_count = 0
        return self._close(
            frame_number, timestamp,
            completed=False, aborted=True,
            clean=None, issues=[], avg_angle=0,
        )

    def reset(self):
        """Discard any open rep and restart rep numbering from 1."""
        self.current_rep = None
        self._next_rep_id = 1
        self._idle_frame_count = 0

    def _close(self, frame_number, timestamp, completed, aborted, clean, issues, avg_angle):
        rep = self.current_rep
        self.current_rep = None
        return {
            "rep_id": rep["rep_id"],
            "start_frame": rep["start_frame"],
            "end_frame": frame_number,
            "start_timestamp": round(rep["start_timestamp"], 4),
            "end_timestamp": round(timestamp, 4),
            "duration": round(timestamp - rep["start_timestamp"], 4),
            "completed": completed,
            "aborted": aborted,
            "clean": clean,
            "issues": issues,
            "avg_angle": avg_angle,
        }
