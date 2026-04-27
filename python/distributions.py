from __future__ import annotations

import numpy as np

from config import MixtureConfig


def sample_source(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=1.0, size=n_samples)


def sample_target(
    n_samples: int,
    mixture_config: MixtureConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = np.asarray(mixture_config.weights, dtype=float)
    means = np.asarray(mixture_config.means, dtype=float)
    stds = np.asarray(mixture_config.stds, dtype=float)

    components = rng.choice(len(weights), size=n_samples, p=weights)
    return rng.normal(loc=means[components], scale=stds[components])


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (x - mean) / std
    return np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * std)


def source_pdf(x: np.ndarray) -> np.ndarray:
    return normal_pdf(x, mean=0.0, std=1.0)


def target_pdf(x: np.ndarray, mixture_config: MixtureConfig) -> np.ndarray:
    weights = np.asarray(mixture_config.weights, dtype=float)
    means = np.asarray(mixture_config.means, dtype=float)
    stds = np.asarray(mixture_config.stds, dtype=float)

    pdf = np.zeros_like(x, dtype=float)
    for w, mean, std in zip(weights, means, stds):
        pdf += w * normal_pdf(x, mean=mean, std=std)
    return pdf

