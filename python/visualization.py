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

    max_curves = min(rollout_path.shape[1], 20)
    for idx in range(max_curves):
        axes[1, 0].plot(rollout_path[:, idx, 0], rollout_path[:, idx, 1], alpha=0.75)
    axes[1, 0].set_title("Sample Trajectories in 2D")
    axes[1, 0].set_xlabel("x(t)")
    axes[1, 0].set_ylabel("y(t)")
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].grid(alpha=0.25)

    n_plot = min(3500, x_train.shape[0])
    sample_idx = rng.choice(x_train.shape[0], size=n_plot, replace=False)
    x_plot = x_train[sample_idx]
    y_plot = y_train[sample_idx]
    speed = np.linalg.norm(y_plot, axis=1)
    scatter = axes[1, 1].scatter(
        x_plot[:, 1],
        x_plot[:, 2],
        c=speed,
        cmap="viridis",
        s=12,
        alpha=0.55,
    )
    axes[1, 1].set_title("Training States Colored by |v|")
    axes[1, 1].set_xlabel("x_t")
    axes[1, 1].set_ylabel("y_t")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1, 1], label="speed norm")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
