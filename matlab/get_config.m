function cfg = get_config()
cfg.n_train = 500;
cfg.n_generated = 400;
cfg.n_trajectories = 40;
cfg.n_time_slices = 100;
cfg.time_steps = 100;
cfg.t_min = 0.0;
cfg.random_seed = 7;

cfg.mixture.weights = [0.5, 0.5];
cfg.mixture.means = [-2.0, -2.0; 2.0, 2.0];
cfg.mixture.stds = [0.45, 0.70; 0.80, 0.55];

cfg.gp.length_scale_xy = 1.0;
cfg.gp.signal_variance = 1.0;
cfg.gp.noise_variance = 1e-3;
end
