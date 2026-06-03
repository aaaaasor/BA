from __future__ import annotations

import numpy as np

from config import VarianceConstraintConfig
from gp_model import TimeSliceGPCollection


def velocity_field(
    model: TimeSliceGPCollection,
    t: float,
    x: np.ndarray,
) -> np.ndarray:
    return model.predict(t, x)


def _variance_scalar(
    model: TimeSliceGPCollection,
    t: float,
    x: np.ndarray,
) -> np.ndarray:
    return model.predict_variance_scalar(t, x).reshape(-1)


def constrained_velocity_field(
    model: TimeSliceGPCollection,
    t: float,
    x: np.ndarray,
    t0: float,
    t1: float,
    constraint_config: VarianceConstraintConfig | None = None,
) -> np.ndarray:
    mu = velocity_field(model, t, x)
    if constraint_config is None or not constraint_config.enabled:
        return mu

    variance_now = _variance_scalar(model, t, x)
    grad_x = model.predict_variance_grad_x_scalar(t, x)

    normalized_t = (t - t0) / max(t1 - t0, constraint_config.time_eps)
    remaining = max(1.0 - normalized_t, constraint_config.time_eps)
    phi_t = constraint_config.omega_gain / (remaining**2)
    h = constraint_config.sigma2_max - variance_now
    rhs = phi_t * constraint_config.alpha_gain * h - np.sum(grad_x * mu, axis=1)
    grad_norm_sq = np.sum(grad_x**2, axis=1)

    u = np.zeros_like(mu, dtype=float)
    active = (rhs < 0.0) & (grad_norm_sq > constraint_config.grad_tol)
    if np.any(active):
        scale = rhs[active] / grad_norm_sq[active]
        u[active] = grad_x[active] * scale[:, None]
    return mu + u


def rk4_rollout(
    model: TimeSliceGPCollection,
    x_init: np.ndarray,
    t0: float,
    t1: float,
    n_steps: int,
    constraint_config: VarianceConstraintConfig | None = None,
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
        k1 = constrained_velocity_field(model, t, x, t0, t1, constraint_config)
        k2 = constrained_velocity_field(model, t + 0.5 * dt, x + 0.5 * dt * k1, t0, t1, constraint_config)
        k3 = constrained_velocity_field(model, t + 0.5 * dt, x + 0.5 * dt * k2, t0, t1, constraint_config)
        k4 = constrained_velocity_field(model, t + dt, x + dt * k3, t0, t1, constraint_config)
        x = x + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4)
        path[i + 1] = x

    return times, path
