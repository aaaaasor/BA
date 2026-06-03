from __future__ import annotations

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MPL_DIR = os.path.join(_THIS_DIR, ".matplotlib")
os.makedirs(_MPL_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = _MPL_DIR

import matplotlib.pyplot as plt
import numpy as np

from config import MixtureConfig
from distributions import source_pdf, target_pdf


def plot_results(
    output_path: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mixture_config: MixtureConfig,
    generated_samples: np.ndarray,
    rollout_path: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    rng = np.random.default_rng(random_seed)

    source_density = source_pdf(grid_x, grid_y)
    target_density = target_pdf(grid_x, grid_y, mixture_config)
    source_levels = np.linspace(0.05 * np.max(source_density), 0.95 * np.max(source_density), 8)
    target_levels = np.linspace(0.05 * np.max(target_density), 0.95 * np.max(target_density), 8)

    axes[0, 0].contour(grid_x, grid_y, source_density, levels=source_levels, linewidths=1.8, cmap="Blues")
    axes[0, 0].contour(grid_x, grid_y, target_density, levels=target_levels, linewidths=1.4, cmap="Oranges")
    axes[0, 0].set_title("Source and Target Densities")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].scatter(generated_samples[:, 0], generated_samples[:, 1], s=16, alpha=0.5, label="Generated")
    axes[0, 1].contour(grid_x, grid_y, target_density, levels=target_levels, linewidths=1.5, cmap="Oranges")
    axes[0, 1].set_title("Generated Samples vs Target")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].set_aspect("equal", adjustable="box")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    max_curves = min(rollout_path.shape[1], 40)
    for idx in range(max_curves):
        axes[1, 0].plot(rollout_path[:, idx, 0], rollout_path[:, idx, 1], alpha=0.75)
    axes[1, 0].set_title("Sample Trajectories in 2D")
    axes[1, 0].set_xlabel("x(t)")
    axes[1, 0].set_ylabel("y(t)")
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].grid(alpha=0.25)

    n_quiver = min(450, x_train.shape[0])
    quiver_idx = rng.choice(x_train.shape[0], size=n_quiver, replace=False)
    x_quiver = x_train[quiver_idx]
    y_quiver = y_train[quiver_idx]
    axes[1, 1].quiver(
        x_quiver[:, 1],
        x_quiver[:, 2],
        y_quiver[:, 0],
        y_quiver[:, 1],
        angles="xy",
        scale_units="xy",
        scale=8.0,
        width=0.003,
        alpha=0.75,
        color="tab:blue",
    )
    axes[1, 1].set_title("Training Velocity Field Arrows")
    axes[1, 1].set_xlabel("x_t")
    axes[1, 1].set_ylabel("y_t")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
