"""
================================================================================
NeuralStride AI - Exercise Registry
================================================================================

Central registry for all available exercises. Main.py uses this to show
the exercise selection menu and instantiate the chosen exercise class.

CONTENTS:
    EXERCISES dict      — Maps menu number to (name, class) pairs
    show_exercise_menu() — Prints the selection menu to console
    get_exercise()       — Returns an exercise instance by menu number

REGISTERED EXERCISES:
    1. Squat            — SquatExercise      (detectors/squat.py)
    2. Push-Up          — PushUpExercise     (detectors/pushup.py)
    3. Plank            — PlankExercise      (detectors/plank.py)
    4. Lunge            — LungeExercise      (detectors/lunge.py)
    5. Side Leg Raise   — LegRaiseExercise   (detectors/legraise.py)

TO ADD A NEW EXERCISE:
    1. Create detectors/your_exercise.py extending BaseExercise
    2. Add import at the top of this file
    3. Add entry to EXERCISES dict with the next available number
    4. The menu and instantiation will work automatically
================================================================================
"""

# Import all exercise classes
from detectors.squat import SquatExercise
from detectors.pushup import PushUpExercise
from detectors.plank import PlankExercise
from detectors.lunge import LungeExercise
from detectors.legraise import LegRaiseExercise

# Exercise registry: menu_number -> (display_name, exercise_class)
# Main.py reads this to build the selection menu and create instances
EXERCISES = {
    1: ("Squat", SquatExercise),
    2: ("Push-Up", PushUpExercise),
    3: ("Plank", PlankExercise),
    4: ("Lunge", LungeExercise),
    5: ("Side Leg Raise", LegRaiseExercise),
}


def show_exercise_menu():
    """
    Prints the exercise selection menu to the console.
    Called by main.py before the user picks an exercise.
    """
    print("\n=== NeuralStride AI - Exercise Selection ===")
    for num, (name, _) in EXERCISES.items():
        print(f"  {num}. {name}")
    print("============================================")


def get_exercise(choice, target_time=None):
    """
    Create and return an exercise instance based on the user's menu choice.

    Args:
        choice:      Menu number (1-4) selected by the user
        target_time: For Plank only — target hold time in seconds.
                     Ignored for other exercises. Passed to PlankExercise
                     constructor so it knows how long each hold should be.

    Returns:
        Tuple of (exercise_instance, exercise_name) if choice is valid.
        (None, None) if choice is not in the registry.
    """
    if choice in EXERCISES:
        name, cls = EXERCISES[choice]

        # Plank requires a target_time argument for hold duration
        if choice == 3 and target_time is not None:
            return cls(target_time=target_time), name

        # All other exercises use default constructor
        return cls(), name

    return None, None
