import numpy as np

STOP_DISTANCE_M = 0.20
MAX_SPEED_MPS = 0.16


def solve_flat_polynomial(z0, zdot0, zf, zdotf, T):
    """Solve the four cubic coefficients from start/end position and speed."""
    if T <= 0:
        raise ValueError("T must be positive")

    matrix = np.array([
        [1, 0,   0,       0],
        [0, 1,   0,       0],
        [1, T, T*T,   T*T*T],
        [0, 1, 2*T,   3*T*T],
    ], dtype=float)
    boundary = np.array([z0, zdot0, zf, zdotf], dtype=float)
    return np.linalg.solve(matrix, boundary)


def make_trajectory(d0):
    """Plan from the current distance d0 to a 0.20 m standoff."""
    if d0 <= STOP_DISTANCE_M:
        raise ValueError("Already at the stopping distance")

    travel = d0 - STOP_DISTANCE_M
    T = max(1.0, 1.6 * travel / MAX_SPEED_MPS)

    alpha_x = solve_flat_polynomial(0.0, 0.06, travel, 0.0, T)
    alpha_y = solve_flat_polynomial(0.0, 0.0, 0.0, 0.0, T)
    return T, alpha_x, alpha_y


def evaluate(T, alpha_x, alpha_y, t):
    """Return planned x, y, linear speed v, and turn rate omega."""
    t = min(max(float(t), 0.0), T)

    def position_velocity_acceleration(a):
        position = a[0] + a[1]*t + a[2]*t*t + a[3]*t*t*t
        velocity = a[1] + 2*a[2]*t + 3*a[3]*t*t
        acceleration = 2*a[2] + 6*a[3]*t
        return position, velocity, acceleration

    x, xdot, xddot = position_velocity_acceleration(alpha_x)
    y, ydot, yddot = position_velocity_acceleration(alpha_y)

    v = float(np.hypot(xdot, ydot))
    speed_squared = xdot*xdot + ydot*ydot
    omega = (
        float((xdot*yddot - ydot*xddot) / speed_squared)
        if speed_squared > 1e-12 else 0.0
    )

    if t >= T:
        v, omega = 0.0, 0.0
    return float(x), float(y), v, omega
