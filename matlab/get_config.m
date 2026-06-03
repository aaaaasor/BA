% Collect all configuration values for the MATLAB demo.
function cfg = get_config()
%% Training Data
cfg.n_train = 500;
cfg.n_trajectories = 1;
cfg.n_time_slices = 100;
cfg.time_steps = 100;
cfg.t_min = 0.0;
cfg.rollout_t_max = 0.99;
cfg.random_seed = 7;
cfg.first_level_fit_seed = cfg.random_seed + 1;
cfg.rollout_seed = cfg.random_seed + 2;
cfg.second_level_rollout_seed = cfg.random_seed + 3;
cfg.second_level_fit_seed = cfg.random_seed + 4;
cfg.segment_points_per_segment = 5;
cfg.run_second_level = true;
cfg.second_level_generation_samples = 10;
cfg.second_level_anchor_source = 'training_data'; % 'first_level_rollout' or 'training_data'
cfg.first_level_target_source = 'generated'; % 'generated' or 'dataset'
%% Animation
cfg.animation.enabled = true;
cfg.animation.trajectory_nr = 1;
cfg.animation.frame_stride = 2;
cfg.animation.delay_time = 0.05;

%% Diagnostics
cfg.debug.quick_second_level_training = false;
cfg.debug.second_level_max_training_pairs = 2000;
cfg.debug.second_level_training_subset_seed = 11;

%% Cache
cfg.cache.enabled = false;
cfg.cache.first_level_model_path = fullfile('outputs', ...
    'LoG_GP_FirstLevel_Model.mat');
cfg.cache.second_level_model_path = fullfile('outputs', ...
    'LoG_GP_SecondLevel_Model.mat');
cfg.cache.first_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_FirstLevel_Rollout.mat');
cfg.cache.second_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_SecondLevel_Rollout.mat');

%% LoG-GP Parameters
cfg.gp.length_scale_vec = [0.3553; 6711; 2.472e4; 295.6; 114.9; ...
    179.6; 37.76; 6189; 92.38; 5.585e4; 1.134e4];
cfg.gp.signal_std = 28.1481;
cfg.gp.noise_std = 0.45322;
cfg.gp.max_data_quantity = cfg.n_train;
cfg.gp.optimize_hyperparameters = true;
cfg.gp.reuse_saved_hyperparameters = true;
cfg.gp.use_per_output_models = true;
cfg.gp.n_pretrain = cfg.n_train * cfg.n_time_slices;
cfg.gp.pretrain_output_idx = [];
cfg.gp.hyperparameter_mat_path = fullfile('outputs', 'LoG_GP_Hyperparameter.mat');
cfg.gp.length_scale_bounds = [0.3, 5.0];
cfg.gp.signal_std_bounds = [0.3, 2.0];
cfg.gp.noise_std_bounds = [1e-3, 0.1];
cfg.gp.max_local_data_quantity = 200;
cfg.gp.max_local_gp_quantity = ceil(2.0 * cfg.n_train * cfg.n_time_slices / ...
    cfg.gp.max_local_data_quantity);
cfg.gp.aggregation_method = 'GPOE';
cfg.gp.selective_training_enabled = true;
cfg.gp.training_accuracy_threshold = 0.5;
cfg.gp.second_level_training_accuracy_threshold = 1.0;
cfg.gp.generation_accuracy_threshold = 0.8;
cfg.gp.second_level_generation_accuracy_threshold = 1.2;

%% Variance Constraint
cfg.variance_constraint.enabled = true;
cfg.variance_constraint.sigma2_max = cfg.gp.generation_accuracy_threshold;
cfg.variance_constraint.generation_accuracy_threshold = ...
    cfg.gp.generation_accuracy_threshold;
cfg.variance_constraint.alpha_gain = 5.0;
cfg.variance_constraint.omega_gain = 0.12;
cfg.variance_constraint.second_level_alpha_gain = 5.0;
cfg.variance_constraint.second_level_omega_gain = 0.02;
cfg.variance_constraint.second_level_enabled = true;
cfg.variance_constraint.grad_tol = 1e-10;
end
