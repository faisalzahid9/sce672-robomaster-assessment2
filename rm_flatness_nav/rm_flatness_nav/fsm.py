from enum import Enum


class State(Enum):
    SEARCHING = "SEARCHING"
    ALIGNING = "ALIGNING"
    CHASING = "CHASING"
    ARRIVED = "ARRIVED"
    SEARCH_FAILED = "SEARCH_FAILED"
    BACKING_UP = "BACKING_UP"
    RETURNING_HOME = "RETURNING_HOME"


def choose_state(current, seconds_since_seen, bearing_rad,
                 camera_distance_m, tof_distance_m):
    """Select a state without sending any robot commands."""

    # Terminal states: once stopped at the target (or after a
    # failed sweep), stay stopped.
    if current in (State.ARRIVED, State.SEARCH_FAILED,
                    State.RETURNING_HOME):
        return current

    # Stop when the camera sees the chosen tag at 0.20 m. ToF is
    # NOT used for "arrived": it measures the nearest obstacle,
    # not necessarily the chosen tag.
    if camera_distance_m is not None and camera_distance_m <= 0.20+1e-6:
        return State.ARRIVED

    # While backing up, stay in BACKING_UP until the planner's
    # timer transitions us out. Exit early only if the tag appears.
    if current == State.BACKING_UP and bearing_rad is None:
        return State.BACKING_UP

    # Search only after the tag has been missing for 0.45 seconds.
    if seconds_since_seen > 0.45:
        return State.SEARCHING

    # During a brief loss of the tag, keep the state. The later motion
    # controller will command zero forward speed until it sees T3 again.
    if bearing_rad is None:
        return current

    if abs(bearing_rad) > 0.18:
        return State.ALIGNING

    return State.CHASING

