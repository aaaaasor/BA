from __future__ import annotations

import os

import numpy as np

from config import GPConfig, MixtureConfig, SamplingConfig, TrainingConfig
from distributions import sample_source, sample_target
from flow_matching import build_training_data
from gp_model import TimeSliceGPCollection
from sampler import rk4_rollout
from visualization import plot_results


def main() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    training_config = TrainingConfig()
    sampling_config = SamplingConfig()
    mixture_config = MixtureConfig()
    gp_config = GPConfig()
    rng = np.random.default_rng(training_config.random_seed)

    x0_train = sample_source(training_config.n_train, rng)
    x1_train = sample_target(training_config.n_train, mixture_config, rng)
    slice_times, x_slices, y_slices, x_train, y_train = build_training_data(
        x0_train,
        x1_train,
        training_config.n_time_slices,
        t_min=training_config.t_min,
        t_max=1.0,
    )

    model = TimeSliceGPCollection(gp_config)
    model.fit(slice_times, x_slices, y_slices)

    x0_eval = sample_source(sampling_config.n_generated, rng)
    _, rollout_path = rk4_rollout(
        model=model,
        x_init=x0_eval,
        t0=training_config.t_min,
        t1=1.0,
        n_steps=sampling_config.time_steps,
    )
    generated_samples = rollout_path[-1]

    x0_traj = sample_source(sampling_config.n_trajectories, rng)
    _, traj_path = rk4_rollout(
        model=model,
        x_init=x0_traj,
        t0=training_config.t_min,
        t1=1.0,
        n_steps=sampling_config.time_steps,
    )

    axis_grid = np.linspace(-5.0, 5.0, 180)
    grid_x, grid_y = np.meshgrid(axis_grid, axis_grid)
    output_dir = os.path.join(this_dir, "outputs")
    output_path = os.path.join(output_dir, "gp_flow_matching_demo.png")
    os.makedirs(output_dir, exist_ok=True)
    plot_results(
        output_path=output_path,
        grid_x=grid_x,
        grid_y=grid_y,
        mixture_config=mixture_config,
        generated_samples=generated_samples,
        rollout_path=traj_path,
        x_train=x_train,
        y_train=y_train,
        random_seed=training_config.random_seed,
    )
    np.savetxt(
        os.path.join(output_dir, "generated_samples.csv"),
        generated_samples,
        delimiter=",",
        header="x_t1,y_t1",
        comments="",
    )

    traj_table = np.column_stack(
        [
            np.linspace(training_config.t_min, 1.0, sampling_config.time_steps + 1),
            traj_path[:, :, 0],
            traj_path[:, :, 1],
        ]
    )
    traj_headers = ["t"] + [f"path_{idx}_x" for idx in range(traj_path.shape[1])] + [f"path_{idx}_y" for idx in range(traj_path.shape[1])]
    np.savetxt(
        os.path.join(output_dir, "trajectory_samples.csv"),
        traj_table,
        delimiter=",",
        header=",".join(traj_headers),
        comments="",
    )

    print("Training samples:", training_config.n_train)
    print("Time slices:", training_config.n_time_slices)
    print("Total training pairs:", x_train.shape[0])
    print("Generated samples:", sampling_config.n_generated)
    print("Output figure:", os.path.abspath(output_path))
    print("Generated mean:", np.mean(generated_samples, axis=0))
    print("Generated std:", np.std(generated_samples, axis=0))


if __name__ == "__main__":
    main()
