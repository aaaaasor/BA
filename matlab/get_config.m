% Collect all configuration values for the MATLAB demo.
function cfg = get_config()
%% Training Data
cfg.n_train = 500;
cfg.n_trajectories = 40;
cfg.n_time_slices = 100;
cfg.time_steps = 100;
cfg.t_min = 0.0;
cfg.random_seed = 7;

%% LocalGP Parameters
cfg.gp.length_scale_vec = ones(10, 1);
cfg.gp.signal_std = 1.0;
cfg.gp.noise_std = sqrt(1e-3);
cfg.gp.max_data_quantity = cfg.n_train;

%% Variance Constraint
cfg.variance_constraint.enabled = true;
cfg.variance_constraint.sigma2_max = 5e-5;
cfg.variance_constraint.alpha_gain = 5.0;
cfg.variance_constraint.omega_gain = 0.12;
cfg.variance_constraint.time_eps = 5e-2;
cfg.variance_constraint.grad_tol = 1e-10;
end
