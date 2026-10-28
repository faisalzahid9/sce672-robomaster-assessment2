import numpy as np

WHEEL_RADIUS_M = 0.05
CHASSIS_RADIUS_M = 0.20
KV = 0.38
KA = 0.06


def wheel_rates(vx, wz):
    """Return FR, FL, RL, RR wheel speeds in rad/s."""
    front_right = (vx + CHASSIS_RADIUS_M * wz) / WHEEL_RADIUS_M
    front_left = (vx - CHASSIS_RADIUS_M * wz) / WHEEL_RADIUS_M
    return np.array([
        front_right, front_left, front_left, front_right
    ], dtype=float)


def estimate_voltages(vx, wz, previous_rates, dt):
    """Return current wheel rates and modeled V1, V2, V3, V4."""
    if dt <= 0:
        raise ValueError("dt must be positive")

    rates = wheel_rates(vx, wz)
    previous = np.asarray(previous_rates, dtype=float).reshape(4)
    acceleration = (rates - previous) / dt
    voltages = KV * rates + KA * acceleration
    return rates, voltages

