% Collect all configuration values for the MATLAB demo.
function cfg = get_config()
%% Training Data
cfg.n_train = 500;
cfg.n_trajectories = 40;
cfg.n_time_slices = 100;
cfg.time_steps = 100;
cfg.t_min = 0.0;
cfg.random_seed = 7;

%% Animation
cfg.animation.enabled = true;
cfg.animation.trajectory_nr = 1;
cfg.animation.frame_stride = 2;
cfg.animation.delay_time = 0.05;

%% LoG-GP Parameters
cfg.gp.length_scale_vec = ones(11, 1);
cfg.gp.signal_std = 1.0;
cfg.gp.noise_std = sqrt(1e-3);
cfg.gp.max_data_quantity = cfg.n_train;
cfg.gp.optimize_hyperparameters = true;
cfg.gp.n_pretrain = 1000;
cfg.gp.pretrain_output_idx = 1:10;
cfg.gp.length_scale_bounds = [0.3, 5.0];
cfg.gp.signal_std_bounds = [0.3, 2.0];
cfg.gp.noise_std_bounds = [1e-3, 0.1];
cfg.gp.max_local_data_quantity = 500;
cfg.gp.max_local_gp_quantity = ceil(2.0 * cfg.n_train * cfg.n_time_slices / ...
    cfg.gp.max_local_data_quantity);
cfg.gp.aggregation_method = 'GPOE';

%% Variance Constraint
cfg.variance_constraint.enabled = true;
cfg.variance_constraint.sigma2_max = 1e-2;
cfg.variance_constraint.alpha_gain = 5.0;
cfg.variance_constraint.omega_gain = 0.12;
cfg.variance_constraint.time_eps = 5e-2;
cfg.variance_constraint.grad_tol = 1e-10;
end
