from __future__ import annotations

import numpy as np

from gp_model import TimeSliceGPCollection


def velocity_field(
    model: TimeSliceGPCollection,
    t: float,
    x: np.ndarray,
) -> np.ndarray:
    return model.predict(t, x)


def rk4_rollout(
    model: TimeSliceGPCollection,
    x_init: np.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x_init.ndim != 2 or x_init.shape[1] != 2:
        raise ValueError("x_init must have shape (n_samples, 2).")

    dt = (t1 - t0) / n_steps
    times = np.linspace(t0, t1, n_steps + 1)
    path = np.zeros((n_steps + 1, x_init.shape[0], 2), dtype=float)
    path[0] = x_init

    x = x_init.copy()
    for i in range(n_steps):
        t = times[i]
        k1 = velocity_field(model, t, x)
        k2 = velocity_field(model, t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = velocity_field(model, t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = velocity_field(model, t + dt, x + dt * k3)
        x = x + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4)
        path[i + 1] = x

    return times, path
