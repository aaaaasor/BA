from dataclasses import dataclass


@dataclass(frozen=True)
class MixtureConfig:
    weights: tuple[float, float] = (0.5, 0.5)
    means: tuple[tuple[float, float], tuple[float, float]] = ((-2.0, -2.0), (2.0, 2.0))
    stds: tuple[tuple[float, float], tuple[float, float]] = ((0.45, 0.70), (0.80, 0.55))


@dataclass(frozen=True)
class TrainingConfig:
    n_train: int = 500
    n_time_slices: int = 100
    t_min: float = 0.0
    random_seed: int = 7


@dataclass(frozen=True)
class SamplingConfig:
    n_generated: int = 400
    n_trajectories: int = 40
    time_steps: int = 100


@dataclass(frozen=True)
class GPConfig:
    length_scale_xy: float = 1.0
    signal_variance: float = 1.0
    noise_variance: float = 1e-3


@dataclass(frozen=True)
class VarianceConstraintConfig:
    enabled: bool = True
    sigma2_max: float = 5e-3
    alpha_gain: float = 5
    omega_gain: float = 0.12
    time_eps: float = 5e-2
    grad_tol: float = 1e-10
