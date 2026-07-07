% Collect all configuration values for the MATLAB demo.
function cfg = get_config()
%% Training Data
cfg.n_train = 30;
cfg.first_level_generation_samples = 5;
cfg.first_level_use_tangent_features = true;
cfg.n_time_slices = 15;
cfg.first_level_time_steps = 100;
cfg.second_level_time_steps = 100;
cfg.third_level_time_steps = 100;
cfg.t_min = 0.0;
cfg.rollout_t_max = 0.995;
cfg.random_seed = 7;
cfg.first_level_data_seed = cfg.random_seed + 1;
cfg.first_level_hyperparameter_seed = cfg.random_seed + 2;
cfg.first_level_fit_seed = cfg.random_seed + 3;
cfg.first_level_rollout_seed = cfg.random_seed + 4;
cfg.second_level_data_seed = cfg.random_seed + 5;
cfg.second_level_hyperparameter_seed = cfg.random_seed + 6;
cfg.second_level_fit_seed = cfg.random_seed + 7;
cfg.second_level_rollout_seed = cfg.random_seed + 8;
cfg.segment_points_per_segment = 5;
cfg.second_level_generation_samples = 1;
cfg.third_level_generation_samples = 1;
cfg.enable_third_level = false;
cfg.stop_after_first_level = false;
cfg.third_level_window_stride = cfg.segment_points_per_segment - 1;

%% Animation
cfg.animation.enabled = true;
cfg.animation.trajectory_nr = 1;
cfg.animation.frame_stride = 2;
cfg.animation.delay_time = 0.12;

%% Output
cfg.output.enabled = true;

%% Cache
cfg.cache.first_level_model_path = fullfile('outputs', 'LoG_GP_FirstLevel_Model.mat');
cfg.cache.second_level_model_path = fullfile('outputs', 'LoG_GP_SecondLevel_Model.mat');
cfg.cache.third_level_model_path = fullfile('outputs', 'LoG_GP_ThirdLevel_Model.mat');
cfg.cache.first_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_FirstLevel_Rollout_WithHOCBF_PTCBF.mat');
cfg.cache.second_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_SecondLevel_Rollout_WithUFrom0_alpha2_0p01_gamma_0p01_h0_0p01.mat');
cfg.cache.third_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_ThirdLevel_Rollout_WithUFrom0.mat');
cfg.cache.first_level_hyperparameter_path = fullfile('outputs', 'LoG_GP_FirstLevel_Hyperparameter.mat');
cfg.cache.second_level_hyperparameter_path = fullfile('outputs', 'LoG_GP_SecondLevel_Hyperparameter.mat');
cfg.cache.third_level_hyperparameter_path = fullfile('outputs', 'LoG_GP_ThirdLevel_Hyperparameter.mat');

%% LoG-GP Parameters
cfg.gp.first_level_n_pretrain = 450;
cfg.gp.second_level_n_pretrain = 500;
cfg.gp.second_level_length_scale_time_varying = true;
cfg.gp.second_level_length_scale_time_scale_start = 1.0;
cfg.gp.second_level_length_scale_time_scale_end = 0.5;
cfg.gp.third_level_n_pretrain = 1000;
cfg.gp.third_level_use_manual_hyperparameters = true;
cfg.gp.third_level_length_scale_scale = 30.0;
cfg.gp.third_level_length_scale_min = 1.0;
cfg.gp.third_level_noise_std_scale = 1.0;
cfg.gp.third_level_noise_std_vec = third_level_manual_noise_std_vec();
cfg.gp.third_level_noise_std_vec = cfg.gp.third_level_noise_std_scale * ...
    cfg.gp.third_level_noise_std_vec;
cfg.gp.third_level_signal_std_vec = third_level_manual_signal_std_vec();
cfg.gp.third_level_length_scale_mat = third_level_manual_length_scale_mat();
cfg.gp.third_level_length_scale_mat = ...
    cfg.gp.third_level_length_scale_scale * ...
    cfg.gp.third_level_length_scale_mat;
cfg.gp.third_level_length_scale_mat = max( ...
    cfg.gp.third_level_length_scale_mat, ...
    cfg.gp.third_level_length_scale_min);
cfg.gp.max_local_data_quantity = 200;
cfg.gp.max_local_gp_quantity = ceil(2.0 * cfg.n_train * cfg.n_time_slices / ...
    cfg.gp.max_local_data_quantity);
cfg.gp.aggregation_method = 'GPOE';
cfg.gp.first_level_training_accuracy_threshold = 0.8;
cfg.gp.second_level_training_accuracy_threshold = 2.0;
cfg.gp.third_level_training_accuracy_threshold = 0.0005;
%% Variance Constraint
cfg.variance_constraint.grad_tol = 1e-6;
cfg.variance_constraint.first_level_integral_uncertainty_budget = 10;
cfg.variance_constraint.first_level_hocbf_enabled = true;
cfg.variance_constraint.first_level_hocbf_alpha2 = 3.0;
cfg.variance_constraint.first_level_hocbf_relaxation_bound = 5;
cfg.variance_constraint.first_level_psi1_margin = 2;
cfg.variance_constraint.first_level_diagnostics = true;
cfg.variance_constraint.first_level_ptcbf_enabled = true;
cfg.variance_constraint.first_level_terminal_variance_beta_final = 50;
cfg.variance_constraint.first_level_terminal_variance_ptzf_initial_margin = 5;
cfg.variance_constraint.first_level_terminal_variance_ptzf_gamma = 0.05;
cfg.variance_constraint.first_level_terminal_variance_alpha = 2;
cfg.variance_constraint.first_level_ptclf_enabled = false;
cfg.variance_constraint.first_level_hocbf_slack_enabled = false;
cfg.variance_constraint.first_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.second_level_integral_uncertainty_budget = 3;
cfg.variance_constraint.second_level_hocbf_enabled = true;
cfg.variance_constraint.second_level_hocbf_alpha2 = 5.0;
cfg.variance_constraint.second_level_hocbf_relaxation_bound = 3;
cfg.variance_constraint.second_level_psi1_margin = 60;
cfg.variance_constraint.second_level_diagnostics = true;
cfg.variance_constraint.second_level_ptcbf_enabled = true;
cfg.variance_constraint.second_level_terminal_variance_beta_final = 5.0;
cfg.variance_constraint.second_level_terminal_variance_ptzf_initial_margin = 1;
cfg.variance_constraint.second_level_terminal_variance_ptzf_gamma = 0.5;
cfg.variance_constraint.second_level_terminal_variance_alpha = 5.0;
cfg.variance_constraint.second_level_ptclf_enabled = true;
cfg.variance_constraint.second_level_anchor_clf_ptzf_enabled = false;
cfg.variance_constraint.second_level_anchor_clf_ptzf_cg = 0.02;
cfg.variance_constraint.second_level_anchor_clf_cpt = 10.0;
cfg.variance_constraint.second_level_anchor_clf_ptzf_initial_margin = 3;
cfg.variance_constraint.second_level_slack_enabled = true;
cfg.variance_constraint.second_level_hocbf_slack_enabled = false;
cfg.variance_constraint.second_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.second_level_anchor_clf_slack_enabled = true;
cfg.variance_constraint.second_level_hocbf_slack_weight = 1e4;
cfg.variance_constraint.second_level_terminal_variance_slack_weight = 1e4;
cfg.variance_constraint.second_level_anchor_clf_slack_weight = 1.5;
cfg.variance_constraint.third_level_integral_uncertainty_budget = 5.76;
cfg.variance_constraint.third_level_hocbf_enabled = true;
cfg.variance_constraint.third_level_hocbf_alpha2 = 5.0;
cfg.variance_constraint.third_level_hocbf_relaxation_bound = 8.0;
cfg.variance_constraint.third_level_psi1_margin = 1.0;
cfg.variance_constraint.third_level_diagnostics = true;
cfg.variance_constraint.third_level_ptcbf_enabled = false;
cfg.variance_constraint.third_level_terminal_variance_beta_final = 1.0;
cfg.variance_constraint.third_level_terminal_variance_ptzf_initial_margin = 1e-6;
cfg.variance_constraint.third_level_terminal_variance_ptzf_gamma = 0.001;
cfg.variance_constraint.third_level_terminal_variance_alpha = 5.0;
cfg.variance_constraint.third_level_ptclf_enabled = false;
end
