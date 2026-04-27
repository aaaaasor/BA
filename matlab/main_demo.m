clear;
clc;

cfg = get_config();
rng(cfg.random_seed);

x0_train = sample_source(cfg.n_train);
x1_train = sample_target(cfg.n_train, cfg.mixture);
[slice_times, x_slices, y_slices, x_train, y_train] = build_training_data(x0_train, x1_train, cfg.n_time_slices, cfg.t_min, 1.0);

gp_model = fit_time_slice_gp_models(slice_times, x_slices, y_slices, cfg.gp);

x0_eval = sample_source(cfg.n_generated);
[~, rollout_path] = ode45_rollout(gp_model, x0_eval, cfg.t_min, 1.0, cfg.time_steps);
generated = rollout_path(end, :)';

x0_traj = sample_source(cfg.n_trajectories);
[traj_times, traj_path] = ode45_rollout(gp_model, x0_traj, cfg.t_min, 1.0, cfg.time_steps);

plot_results(cfg, generated, traj_times, traj_path, x_train, y_train);

disp(['Training samples: ', num2str(cfg.n_train)]);
disp(['Time slices: ', num2str(cfg.n_time_slices)]);
disp(['Total training pairs: ', num2str(size(x_train, 1))]);
disp(['Generated samples: ', num2str(cfg.n_generated)]);
disp(['Generated mean: ', num2str(mean(generated))]);
disp(['Generated std: ', num2str(std(generated))]);
