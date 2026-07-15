% Run the original simple 2D point-flow case and plot LoG-GP utilization.
clear;
clc;

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);

%% Simple 2D Config
cfg.output.enabled = true;
cfg.random_seed = 7;
cfg.n_train = 500;
cfg.n_time_slices = 100;
cfg.t_min = 0.0;
cfg.training_utilization_moving_average_window = 25;

cfg.cache.simple_2d_hyperparameter_path = fullfile('outputs', ...
    'LoG_GP_Simple2D_Hyperparameter.mat');
cfg.cache.simple_2d_model_path = fullfile('outputs', ...
    'LoG_GP_Simple2D_Model.mat');

cfg.gp.n_pretrain = 450;
cfg.gp.max_local_data_quantity = 200;
cfg.gp.max_local_gp_quantity = ceil(2.0 * cfg.n_train * ...
    cfg.n_time_slices / cfg.gp.max_local_data_quantity);
cfg.gp.aggregation_method = 'GPOE';
cfg.gp.training_accuracy_threshold = 0.6;
cfg.gp.hyperparameter_mat_path = cfg.cache.simple_2d_hyperparameter_path;

%% Build 2D Flow-Matching Pairs
rng(cfg.random_seed);
source_points = randn(cfg.n_train, 2);
target_points = sample_simple_2d_target(cfg.n_train);
s_slices = linspace(cfg.t_min, 1.0, cfg.n_time_slices)';

velocity_vectors = target_points - source_points;
x_slices = zeros(numel(s_slices), cfg.n_train, 2);
y_slices = zeros(numel(s_slices), cfg.n_train, 2);
for slice_idx = 1:numel(s_slices)
    s = s_slices(slice_idx);
    x_slices(slice_idx, :, :) = (1.0 - s) * source_points + ...
        s * target_points;
    y_slices(slice_idx, :, :) = velocity_vectors;
end

%% Hyperparameters And LoG-GP Fit
disp('Optimizing or loading simple-2d LoG-GP hyperparameters...');
rng(cfg.random_seed + 1);
cfg.gp = optimize_gp_hyperparameters(x_slices, y_slices, cfg.gp, ...
    s_slices);

disp('Fitting simple-2d LoG-GP flow model...');
cfg.gp.training_trajectory_idx_per_sample = (1:size(x_slices, 2))';
rng(cfg.random_seed + 2);
model_cache_path = fullfile(this_dir, cfg.cache.simple_2d_model_path);
model_collection = fit_or_load_loggp_model(s_slices, x_slices, y_slices, ...
    cfg.gp, model_cache_path, 'simple-2d');

plot_training_trajectory_utilization(cfg, model_collection, 'simple-2d');

%% Console Summary
total_added = sum(model_collection.n_added_per_output);
total_candidates = model_collection.n_training_pairs * model_collection.y_dim;
disp(['Simple 2D training samples: ', num2str(cfg.n_train)]);
disp(['Simple 2D s slices: ', num2str(numel(s_slices))]);
disp(['Simple 2D LoG-GP input dimension: ', ...
    num2str(size(x_slices, 3) + 1)]);
disp(['Simple 2D LoG-GP output dimension: ', ...
    num2str(size(y_slices, 3))]);
disp(['Simple 2D added per output: ', ...
    mat2str(model_collection.n_added_per_output(:)')]);
disp(['Simple 2D skipped per output: ', ...
    mat2str(model_collection.n_skipped_per_output(:)')]);
disp(['Simple 2D overall utilization: ', ...
    num2str(total_added / max(total_candidates, 1))]);
