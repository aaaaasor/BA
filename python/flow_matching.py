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

    dt = (t_max - t_min) / n_time_slices
    slice_times = t_min + (dt * np.arange(n_time_slices, dtype=float))
    xt_slices = ((1.0 - slice_times)[:, None] * x0[None, :]) + (slice_times[:, None] * x1[None, :])
    velocity_slices = np.broadcast_to((x1 - x0)[None, :], xt_slices.shape)

    time_column = np.repeat(slice_times, x0.size)
    flat_inputs = np.column_stack([time_column, xt_slices.reshape(-1)])
    flat_targets = velocity_slices.reshape(-1, 1)
    return slice_times, xt_slices, velocity_slices, flat_inputs, flat_targets
