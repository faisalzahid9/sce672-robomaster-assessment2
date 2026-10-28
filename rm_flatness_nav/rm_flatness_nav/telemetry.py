import csv

import numpy as np

from .motor_model import estimate_voltages


class VoltageLogger:
    def __init__(self, path):
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "t_s", "V1_V", "V2_V", "V3_V", "V4_V",
            "state", "v_mps", "omega_rps", "tof_m"
        ])
        self.file.flush()
        self.previous_rates = np.zeros(4)
        self.previous_time = None

    def record(self, t, v, omega, state, tof=None):
        """Write one timestamped row and print its four modeled voltages."""
        dt = 1.0 if self.previous_time is None else t - self.previous_time
        rates, volts = estimate_voltages(
            v, omega, self.previous_rates, dt
        )
        self.previous_rates = rates
        self.previous_time = t

        label = state.value if hasattr(state, "value") else str(state)
        self.writer.writerow([
            f"{t:.3f}",
            *(f"{voltage:.3f}" for voltage in volts),
            label,
            f"{v:.3f}",
            f"{omega:.3f}",
            "" if tof is None else f"{tof:.3f}",
        ])
        self.file.flush()

        values = ", ".join(f"{voltage:.3f}" for voltage in volts)
        print(f"t={t:.1f}s | V1..V4=[{values}] V | {label}", flush=True)
        return volts

    def close(self):
        self.file.close()
