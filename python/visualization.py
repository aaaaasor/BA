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
    grid: np.ndarray,
    mixture_config: MixtureConfig,
    generated_samples: np.ndarray,
    rollout_times: np.ndarray,
    rollout_path: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rng = np.random.default_rng(random_seed)

    axes[0, 0].plot(grid, source_pdf(grid), label="Source PDF", linewidth=2)
    axes[0, 0].plot(grid, target_pdf(grid, mixture_config), label="Target PDF", linewidth=2)
    axes[0, 0].set_title("Source and Target Distributions")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(generated_samples, bins=40, density=True, alpha=0.65, label="Generated")
    axes[0, 1].plot(grid, target_pdf(grid, mixture_config), label="Target PDF", linewidth=2)
    axes[0, 1].set_title("Generated vs Target")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    max_curves = min(rollout_path.shape[1], 20)
    for idx in range(max_curves):
        axes[1, 0].plot(rollout_times, rollout_path[:, idx], alpha=0.75)
    axes[1, 0].set_title("Sample Trajectories")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("x(t)")
    y_pad = 0.35
    axes[1, 0].set_ylim(np.min(rollout_path[:, :max_curves]) - y_pad, np.max(rollout_path[:, :max_curves]) + y_pad)
    axes[1, 0].grid(alpha=0.3)

    n_plot = min(3500, x_train.shape[0])
    sample_idx = rng.choice(x_train.shape[0], size=n_plot, replace=False)
    x_plot = x_train[sample_idx]
    y_plot = y_train[sample_idx]
    scatter = axes[1, 1].scatter(
        x_plot[:, 0],
        x_plot[:, 1],
        c=y_plot.ravel(),
        cmap="coolwarm",
        s=12,
        alpha=0.5,
    )
    axes[1, 1].set_title("Training Pairs Sample: (t, x_t) -> v")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("x_t")
    axes[1, 1].grid(alpha=0.3)
    fig.colorbar(scatter, ax=axes[1, 1], label="velocity")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
