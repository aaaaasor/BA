from __future__ import annotations

import numpy as np


def build_training_data(
    x0: np.ndarray,
    x1: np.ndarray,
    n_time_slices: int,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_time_slices <= 0:
        raise ValueError("n_time_slices must be positive.")

    if x0.shape != x1.shape or x0.ndim != 2 or x0.shape[1] != 2:
        raise ValueError("x0 and x1 must both have shape (n_samples, 2).")

    dt = (t_max - t_min) / n_time_slices
    t_slices = t_min + (dt * np.arange(n_time_slices, dtype=float))
    xt_slices = ((1.0 - t_slices)[:, None, None] * x0[None, :, :]) + (t_slices[:, None, None] * x1[None, :, :])
    velocity_slices = np.broadcast_to((x1 - x0)[None, :, :], xt_slices.shape)

    time_column = np.repeat(t_slices, x0.shape[0])
    flat_inputs = np.column_stack([time_column, xt_slices.reshape(-1, 2)])
    flat_targets = velocity_slices.reshape(-1, 2)
    return t_slices, xt_slices, velocity_slices, flat_inputs, flat_targets
