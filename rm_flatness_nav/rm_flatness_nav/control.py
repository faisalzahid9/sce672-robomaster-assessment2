import math

from .fsm import State
from .trajectory import evaluate

MAX_TURN_RPS = 0.30
MAX_FORWARD_MPS = 0.16
STOP_M = 0.20
HARD_STOP_M = 0.06
CLOSE_RANGE_M = 0.35
SEARCH_BACKUP_V_MPS = 0.10


def desired_command(state, bearing, last_bearing, unseen_for,
                    tof_m, plan=None, plan_time_s=0.0,
                    camera_distance=None):
    """Return (forward_mps, yaw_rps); publish nothing."""
    # Terminal states: no motion.
    if state in (State.ARRIVED, State.SEARCH_FAILED):
        return 0.0, 0.0

    # Search: rotate the body to look for the chosen tag. This
    # does not need a ToF reading and is not blocked by nearby
    # walls, because finding the tag is a vision task.
    if state == State.SEARCHING:
        direction = -1.0 if last_bearing is not None and last_bearing > 0 else 1.0
        return 0.0, direction * MAX_TURN_RPS

    # Backup: reverse slowly so a hidden tag can come into view.
    if state == State.BACKING_UP:
        return -SEARCH_BACKUP_V_MPS, 0.0

    # Safety brake: never drive forward into an obstacle. ToF is
    # only a veto, never a target distance.
    if tof_m is not None and math.isfinite(tof_m) and tof_m <= HARD_STOP_M:
        return 0.0, 0.0

    # Close-range direct approach: when the camera sees the tag
    # between 0.20 and 0.35 m, skip the trajectory and drive
    # directly toward 0.20 m. This converges in a second or two
    # and survives brief tag flicker (which resets the plan).
    if (camera_distance is not None and bearing is not None
            and STOP_M + 1e-6 < camera_distance <= CLOSE_RANGE_M):
        yaw = max(-MAX_TURN_RPS, min(MAX_TURN_RPS, -1.5 * bearing))
        if abs(bearing) > 0.4:
            return 0.0, yaw
        # RoboMaster chassis has a minimum mobility dead-band
        # around 0.04-0.05 m/s; below that the wheels stall.
        # Use a floor plus a stronger gain so close-mode
        # actually converges to 0.20 m.
        v = max(0.05, min(0.10, 2.0 * (camera_distance - STOP_M)))
        return v, yaw

    # Hold still near the box if the tag is lost or alignment needs a turn.
    if (tof_m is not None and math.isfinite(tof_m)
            and tof_m <= 0.35
            and (state != State.CHASING or bearing is None)):
        return 0.0, 0.0

    error = bearing if bearing is not None else last_bearing
    if error is None or unseen_for > 0.45:
        return 0.0, 0.0

    if state == State.ALIGNING:
        yaw = max(-MAX_TURN_RPS, min(MAX_TURN_RPS, -1.5 * error))
        return 0.0, yaw

    if state == State.CHASING and plan is not None:
        T, alpha_x, alpha_y = plan
        if plan_time_s >= T:
            return 0.0, 0.0

        _, _, planned_v, planned_yaw = evaluate(
            T, alpha_x, alpha_y, plan_time_s
        )
        yaw = planned_yaw - 0.8 * error
        yaw = max(-MAX_TURN_RPS, min(MAX_TURN_RPS, yaw))
        forward = max(0.0, min(MAX_FORWARD_MPS, planned_v))
        return forward, yaw

    return 0.0, 0.0
