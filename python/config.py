from dataclasses import dataclass


@dataclass(frozen=True)
class MixtureConfig:
    weights: tuple[float, float] = (0.5, 0.5)
    means: tuple[float, float] = (-2.0, 2.0)
    stds: tuple[float, float] = (0.5, 0.7)


@dataclass(frozen=True)
class TrainingConfig:
    n_train: int = 500
    n_time_slices: int = 100
    t_min: float = 0.0
    random_seed: int = 7


@dataclass(frozen=True)
class SamplingConfig:
    n_generated: int = 400
    n_trajectories: int = 20
    time_steps: int = 100


@dataclass(frozen=True)
class GPConfig:
    length_scale_x: float = 1.0
    signal_variance: float = 1.0
    noise_variance: float = 1e-3
