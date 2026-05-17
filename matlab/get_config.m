% Collect all configuration values for the MATLAB demo.
% The structure covers:
% - training/data settings,
% - target mixture parameters,
% - LocalGP hyperparameters,
% - LocalGP variance-constraint/PT-CBF parameters.
function cfg = get_config()
cfg.n_train = 500;
cfg.n_trajectories = 40;
cfg.n_time_slices = 100;
cfg.time_steps = 100;
cfg.t_min = 0.0;
cfg.random_seed = 7;

cfg.mixture.weights = [0.5, 0.5];
cfg.mixture.means = [-2.0, -2.0; 2.0, 2.0];
cfg.mixture.stds = [0.45, 0.70; 0.80, 0.55];

cfg.gp.length_scale_vec = [1.0; 1.0];
cfg.gp.signal_std = 1.0;
cfg.gp.noise_std = sqrt(1e-3);
cfg.gp.max_data_quantity = cfg.n_train;

cfg.variance_constraint.enabled = true;
cfg.variance_constraint.sigma2_max = 3e-3;
cfg.variance_constraint.alpha_gain = 5.0;
cfg.variance_constraint.omega_gain = 0.12;
cfg.variance_constraint.time_eps = 5e-2;
cfg.variance_constraint.fd_eps_x = 1e-2;
cfg.variance_constraint.grad_tol = 1e-10;
end
