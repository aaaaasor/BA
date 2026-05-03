from __future__ import annotations

import numpy as np

from config import MixtureConfig


def sample_source(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=1.0, size=(n_samples, 2))


def sample_target(
    n_samples: int,
    mixture_config: MixtureConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = np.asarray(mixture_config.weights, dtype=float)
    means = np.asarray(mixture_config.means, dtype=float)
    stds = np.asarray(mixture_config.stds, dtype=float)

    components = rng.choice(len(weights), size=n_samples, p=weights)
    noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 2))
    return means[components] + (stds[components] * noise)


def normal_pdf_2d(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mean: np.ndarray,
    std_xy: np.ndarray,
) -> np.ndarray:
    std_xy = np.asarray(std_xy, dtype=float)
    dx = grid_x - mean[0] 
    dy = grid_y - mean[1]
    sqdist = (dx**2 / (std_xy[0] ** 2)) + (dy**2 / (std_xy[1] ** 2))
    coeff = 1.0 / (2.0 * np.pi * std_xy[0] * std_xy[1])
    return coeff * np.exp(-0.5 * sqdist)


def source_pdf(grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    return normal_pdf_2d(grid_x, grid_y, mean=np.zeros(2, dtype=float), std_xy=np.ones(2, dtype=float))


def target_pdf(grid_x: np.ndarray, grid_y: np.ndarray, mixture_config: MixtureConfig) -> np.ndarray:
    weights = np.asarray(mixture_config.weights, dtype=float)
    means = np.asarray(mixture_config.means, dtype=float)
    stds = np.asarray(mixture_config.stds, dtype=float)

    pdf = np.zeros_like(grid_x, dtype=float)
    for w, mean, std in zip(weights, means, stds):
        pdf += w * normal_pdf_2d(grid_x, grid_y, mean=mean, std_xy=std)
    return pdf
