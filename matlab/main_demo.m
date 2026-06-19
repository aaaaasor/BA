% Main entry point for the MATLAB trajectory-space LoG-GP ODE demo.
clear;
clc;

%% Configuration
cfg = get_config();
rng(cfg.random_seed);

%% Training Data
disp('Building first-level trajectory-space training data...');
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
first_level_target_points = generate_original_training_points_2d( ...
    5, cfg.n_train);
disp(['First-level target source: generated 5-point trajectories with ', ...
    '[dx/ds, dy/ds] tangent features']);
rng(cfg.first_level_data_seed);
[s_slices, x_slices, y_slices, target_points, source_points, ...
    target_data, source_data, trajectory_t_slices, data_transform] = build_training_data( ...
    "", cfg.t_min, 1.0, cfg.n_train, ...
    cfg.n_time_slices, first_level_target_points);
first_level_feature_dim = size(target_points, 3);

%% LoG-GP Hyperparameters
disp('Optimizing or loading first-level LoG-GP hyperparameters...');
rng(cfg.first_level_hyperparameter_seed);
cfg.gp = optimize_gp_hyperparameters(x_slices, y_slices, cfg.gp, s_slices);

%% Fit LoG-GP Flow Model
disp('Fitting first-level LoG-GP flow model...');
rng(cfg.first_level_fit_seed);
if cfg.cache.model_enabled
    first_model_cache_path = fullfile(this_dir, ...
        cfg.cache.first_level_model_path);
else
    first_model_cache_path = "";
end
first_fit_timer = tic;
model_collection = fit_or_load_loggp_model(s_slices, x_slices, y_slices, ...
    cfg.gp, first_model_cache_path, 'first-level');
disp(['First-level LoG-GP fit/load elapsed: ', ...
    num2str(toc(first_fit_timer), '%.1f'), ' seconds']);

%% RK4 Rollout
disp('Running first-level RK4 rollout...');
n_available_traj = size(source_data, 2);
rng(cfg.rollout_seed);
traj_idx = randperm(n_available_traj, min(cfg.n_trajectories, n_available_traj));
x_init = source_data(:, traj_idx)';
n_eval = size(x_init, 1);
first_rollout_signature = rollout_cache_signature(cfg.variance_constraint, ...
    cfg.time_steps, cfg.t_min, cfg.rollout_t_max, x_init);
first_rollout_cache_hit = false;
first_rollout_cache_timer = tic;
if cfg.cache.rollout_enabled
    first_rollout_cache_path = fullfile(this_dir, ...
        cfg.cache.first_level_rollout_path);
    first_rollout_cache_dir = fileparts(first_rollout_cache_path);
    if ~exist(first_rollout_cache_dir, 'dir')
        mkdir(first_rollout_cache_dir);
    end
    if isfile(first_rollout_cache_path)
        cached_first_rollout = load(first_rollout_cache_path);
        if isfield(cached_first_rollout, 'rollout_times') && ...
                isfield(cached_first_rollout, 'traj_path_10d')
            expected_first_path_size = [cfg.time_steps + 1, ...
                n_eval, size(x_init, 2)];
            cache_signature_ok = ...
                isfield(cached_first_rollout, 'first_rollout_signature') && ...
                isequaln(cached_first_rollout.first_rollout_signature, ...
                first_rollout_signature);
            if isequal(size(cached_first_rollout.traj_path_10d), ...
                    expected_first_path_size) && cache_signature_ok
                rollout_times = cached_first_rollout.rollout_times;
                traj_path_10d = cached_first_rollout.traj_path_10d;
                first_rollout_cache_hit = true;
                disp(['Loaded cached first-level rollout: ', ...
                    first_rollout_cache_path]);
            else
                disp(['Cached first-level rollout setting mismatch. ', ...
                    'Recomputing rollout.']);
            end
        else
            disp('Cached first-level rollout is missing rollout data.');
        end
    end
end
if first_rollout_cache_hit
    disp(['First-level cached rollout load/check elapsed: ', ...
        num2str(toc(first_rollout_cache_timer), '%.1f'), ' seconds']);
end
if ~first_rollout_cache_hit
    first_rollout_timer = tic;
    [rollout_times, traj_path_10d] = rk4_rollout(model_collection, ...
        x_init, cfg.t_min, cfg.rollout_t_max, cfg.time_steps, ...
        cfg.variance_constraint);
    disp(['First-level RK4 rollout elapsed: ', ...
        num2str(toc(first_rollout_timer), '%.1f'), ' seconds']);
    if cfg.cache.rollout_enabled
        try
            save(first_rollout_cache_path, 'rollout_times', ...
                'traj_path_10d', 'first_rollout_signature');
        catch
            save(first_rollout_cache_path, 'rollout_times', ...
                'traj_path_10d', 'first_rollout_signature', '-v7.3');
        end
        disp(['Saved cached first-level rollout: ', ...
            first_rollout_cache_path]);
    end
end

traj_path_plot = zeros(size(traj_path_10d));
for time_idx = 1:numel(rollout_times)
    states_now = squeeze(traj_path_10d(time_idx, :, :));
    if n_eval == 1
        states_now = reshape(states_now, 1, []);
    end
    states_plot = states_now' .* data_transform.std + data_transform.mean;
    traj_path_plot(time_idx, :, :) = reshape(states_plot', 1, n_eval, []);
end

final_trajectory_data = squeeze(traj_path_plot(end, :, :));
if n_eval == 1
    final_trajectory_data = reshape(final_trajectory_data, 1, []);
end

reconstructed_points = zeros(numel(trajectory_t_slices), n_eval, ...
    first_level_feature_dim);
for sample_idx = 1:n_eval
    reconstructed_curve = reshape(final_trajectory_data(sample_idx, :), ...
        first_level_feature_dim, [])';
    reconstructed_points(:, sample_idx, :) = reconstructed_curve;
end
reconstructed_points = normalize_tangent_features(reconstructed_points);
first_level_reference_points = target_points(:, traj_idx, :);
first_level_position_error = reconstructed_points(:, :, 1:2) - ...
    first_level_reference_points(:, :, 1:2);
first_level_position_rmse = sqrt(mean(sum(first_level_position_error .^ 2, ...
    3), 1));
first_level_tangent_angle_error = nan(1, n_eval);
if first_level_feature_dim >= 4
    tangent_dot = sum(reconstructed_points(:, :, 3:4) .* ...
        first_level_reference_points(:, :, 3:4), 3);
    tangent_dot = min(max(tangent_dot, -1.0), 1.0);
    first_level_tangent_angle_error = mean(acosd(tangent_dot), 1);
end
disp(['First-level anchor position RMSE mean: ', ...
    num2str(mean(first_level_position_rmse))]);
disp(['First-level anchor position RMSE max: ', ...
    num2str(max(first_level_position_rmse))]);
if all(isfinite(first_level_tangent_angle_error))
    disp(['First-level anchor tangent angle error mean deg: ', ...
        num2str(mean(first_level_tangent_angle_error))]);
    disp(['First-level anchor tangent angle error max deg: ', ...
        num2str(max(first_level_tangent_angle_error))]);
end
if isfield(cfg, 'stop_after_first_level') && cfg.stop_after_first_level
    plot_first_level_diagnostics(cfg, target_points, source_points, ...
        reconstructed_points, first_level_reference_points);
    disp('Stopped after first-level rollout diagnostics.');
    return;
end

%% Segment-Level Flow Model
disp('Building second-level segment flow-matching data...');
first_level_anchor_idx = 1:(cfg.segment_points_per_segment - 1): ...
    ((size(target_points, 1) - 1) * ...
    (cfg.segment_points_per_segment - 1) + 1);
n_second_level_training_points = (size(target_points, 1) - 1) * ...
    (cfg.segment_points_per_segment - 1) + 1;
target_points_dense = generate_original_training_points_2d( ...
    n_second_level_training_points, size(target_points, 2));
anchor_reconstruction_error = max(abs( ...
    target_points_dense(first_level_anchor_idx, :, :) - target_points), ...
    [], 'all');
disp(['Generated dense target anchor reconstruction error: ', ...
    num2str(anchor_reconstruction_error)]);
rng(cfg.second_level_data_seed);
[segment_s_slices, segment_x_slices, segment_y_slices, ...
    ~, ~, segment_data_transform] = build_sliding_window_training_data( ...
    target_points_dense, cfg.t_min, 1.0, cfg.n_time_slices, ...
    cfg.segment_points_per_segment, 1);
segment_feature_dim = size(target_points_dense, 3);

segment_gp = cfg.gp;
if isfield(cfg.gp, 'second_level_length_scale_vec')
    segment_gp.length_scale_vec = cfg.gp.second_level_length_scale_vec;
end
segment_gp.training_accuracy_threshold = ...
    cfg.gp.second_level_training_accuracy_threshold;
segment_gp.hyperparameter_mat_path = fullfile('outputs', ...
    ['LoG_GP_SlidingWindow_Training_Hyperparameter_', ...
    cfg.slope.cache_tag, '.mat']);
if isfield(cfg.gp, 'second_level_hyperparameter_grouping') && ...
        ~strcmp(cfg.gp.second_level_hyperparameter_grouping, 'none')
    segment_gp.hyperparameter_mat_path = fullfile('outputs', ...
        ['LoG_GP_SlidingWindow_Training_Hyperparameter_', ...
        cfg.slope.cache_tag, '_', ...
        cfg.gp.second_level_hyperparameter_grouping, '.mat']);
end
if isfield(cfg, 'small_sample') && isfield(cfg.small_sample, 'enabled') && ...
        cfg.small_sample.enabled
    segment_gp.save_hyperparameters = true;
    segment_gp.hyperparameter_mat_path = fullfile('outputs', ...
        ['LoG_GP_SlidingWindow_Training_Hyperparameter_', ...
        cfg.slope.cache_tag, '_small.mat']);
    if isfield(cfg.gp, 'second_level_hyperparameter_grouping') && ...
            ~strcmp(cfg.gp.second_level_hyperparameter_grouping, 'none')
        segment_gp.hyperparameter_mat_path = fullfile('outputs', ...
            ['LoG_GP_SlidingWindow_Training_Hyperparameter_', ...
            cfg.slope.cache_tag, '_', ...
            cfg.gp.second_level_hyperparameter_grouping, '_small.mat']);
    end
    if isfield(cfg.small_sample, 'reuse_full_hyperparameters') && ...
            ~cfg.small_sample.reuse_full_hyperparameters
        segment_gp.hyperparameter_mat_path = "";
    end
end
segment_gp.n_pretrain = min(500, ...
    size(segment_x_slices, 1) * size(segment_x_slices, 2));
segment_gp.pretrain_output_idx = ...
    1:(segment_feature_dim * cfg.segment_points_per_segment);
if isfield(cfg.gp, 'second_level_hyperparameter_grouping')
    segment_gp.hyperparameter_grouping = ...
        cfg.gp.second_level_hyperparameter_grouping;
end
segment_gp.max_local_gp_quantity = ceil(2.0 * ...
    size(segment_x_slices, 1) * size(segment_x_slices, 2) / ...
    segment_gp.max_local_data_quantity);
use_manual_second_level_hyperparameters = ...
    isfield(cfg.gp, 'second_level_use_manual_hyperparameters') && ...
    cfg.gp.second_level_use_manual_hyperparameters;
if use_manual_second_level_hyperparameters
    disp('Using manual second-level segment LoG-GP hyperparameters from config...');
    segment_gp.length_scale_mat = cfg.gp.second_level_length_scale_mat;
    segment_gp.length_scale_vec = median(segment_gp.length_scale_mat, 2);
    segment_gp.signal_std_vec = cfg.gp.second_level_signal_std_vec;
    segment_gp.signal_std = median(segment_gp.signal_std_vec);
    segment_gp.noise_std_vec = cfg.gp.second_level_noise_std_vec;
    segment_gp.noise_std = median(segment_gp.noise_std_vec);
    segment_gp.optimize_hyperparameters = false;
    segment_gp.save_hyperparameters = false;
else
    disp('Optimizing or loading second-level segment LoG-GP hyperparameters...');
    rng(cfg.second_level_hyperparameter_seed);
    segment_gp = optimize_gp_hyperparameters(segment_x_slices, ...
        segment_y_slices, segment_gp, segment_s_slices);
end
disp('Fitting second-level segment LoG-GP flow model...');
rng(cfg.second_level_fit_seed);
if cfg.cache.model_enabled
    second_model_cache_path = fullfile(this_dir, ...
        cfg.cache.second_level_model_path);
else
    second_model_cache_path = "";
end
second_fit_timer = tic;
segment_model_collection = fit_or_load_loggp_model(segment_s_slices, ...
    segment_x_slices, segment_y_slices, segment_gp, ...
    second_model_cache_path, 'second-level');
disp(['Second-level LoG-GP fit/load elapsed: ', ...
    num2str(toc(second_fit_timer), '%.1f'), ' seconds']);

if isfield(cfg, 'second_level_use_training_anchors') && ...
        cfg.second_level_use_training_anchors
    second_level_anchor_points = first_level_reference_points;
    disp('Second-level anchor source: first-level training target');
else
    second_level_anchor_points = reconstructed_points;
    disp('Second-level anchor source: first-level rollout');
end

n_generation_segments = size(second_level_anchor_points, 1) - 1;
n_second_level_samples = cfg.second_level_generation_samples;
n_second_level_eval = n_eval * n_second_level_samples;
n_segment_eval = n_second_level_eval * n_generation_segments;
segment_feature_dim = size(second_level_anchor_points, 3);
n_segment_rows = segment_feature_dim * cfg.segment_points_per_segment;
rng(cfg.second_level_rollout_seed);
segment_x_init = randn(n_segment_eval, n_segment_rows);
segment_fixed_mask = false(n_segment_eval, n_segment_rows);
segment_fixed_values = segment_x_init;
fixed_anchor_init_mode = 'anchor';
if isfield(cfg, 'fixed_anchor_init_mode') && ...
        ~isempty(cfg.fixed_anchor_init_mode)
    fixed_anchor_init_mode = cfg.fixed_anchor_init_mode;
end
segment_start_idx = 1:segment_feature_dim;
segment_end_idx = (n_segment_rows - segment_feature_dim + 1):n_segment_rows;
for traj_eval_idx = 1:n_eval
    for second_sample_idx = 1:n_second_level_samples
        second_eval_idx = (traj_eval_idx - 1) * n_second_level_samples + ...
            second_sample_idx;
        for segment_idx = 1:n_generation_segments
            sample_idx = (second_eval_idx - 1) * n_generation_segments + ...
                segment_idx;
            start_point = squeeze(second_level_anchor_points(segment_idx, ...
                traj_eval_idx, :))';
            end_point = squeeze(second_level_anchor_points(segment_idx + 1, ...
                traj_eval_idx, :))';
            start_state = (start_point - ...
                segment_data_transform.mean(segment_start_idx)') ./ ...
                segment_data_transform.std(segment_start_idx)';
            end_state = (end_point - ...
                segment_data_transform.mean(segment_end_idx)') ./ ...
                segment_data_transform.std(segment_end_idx)';
            if strcmp(fixed_anchor_init_mode, 'anchor')
                segment_x_init(sample_idx, segment_start_idx) = start_state;
                segment_x_init(sample_idx, segment_end_idx) = end_state;
            end
            segment_fixed_values(sample_idx, [segment_start_idx, ...
                segment_end_idx]) = [start_state, end_state];
            fixed_idx = [segment_start_idx, segment_end_idx];
            segment_fixed_mask(sample_idx, fixed_idx) = true;
        end
    end
end
disp('Running second-level segment RK4 rollout...');
second_level_uncertainty_threshold = ...
    cfg.gp.second_level_generation_accuracy_threshold;
segment_variance_constraint = cfg.variance_constraint;
segment_variance_constraint.uncertainty_max = ...
    second_level_uncertainty_threshold;
segment_variance_constraint.generation_accuracy_threshold = ...
    second_level_uncertainty_threshold;
if isfield(cfg.variance_constraint, 'second_level_alpha_gain')
    segment_variance_constraint.alpha_gain = ...
        cfg.variance_constraint.second_level_alpha_gain;
end
if isfield(cfg.variance_constraint, 'second_level_omega_gain')
    segment_variance_constraint.omega_gain = ...
        cfg.variance_constraint.second_level_omega_gain;
end
if isfield(cfg.variance_constraint, 'second_level_grad_tol')
    segment_variance_constraint.grad_tol = ...
        cfg.variance_constraint.second_level_grad_tol;
end
if isfield(cfg.variance_constraint, 'second_level_ptzf_gamma')
    segment_variance_constraint.ptzf_gamma = ...
        cfg.variance_constraint.second_level_ptzf_gamma;
end
if isfield(cfg.variance_constraint, 'second_level_ptzf_initial_bound')
    segment_variance_constraint.ptzf_initial_bound = ...
        cfg.variance_constraint.second_level_ptzf_initial_bound;
end
if isfield(cfg.variance_constraint, 'second_level_slack_weight')
    segment_variance_constraint.slack_weight = ...
        cfg.variance_constraint.second_level_slack_weight;
end
if isfield(cfg.variance_constraint, 'second_level_diagnostics')
    segment_variance_constraint.diagnostics = ...
        cfg.variance_constraint.second_level_diagnostics;
end
if isfield(cfg.variance_constraint, 'second_level_diagnostics_version')
    segment_variance_constraint.diagnostics_version = ...
        cfg.variance_constraint.second_level_diagnostics_version;
end
segment_rollout_constraint = segment_variance_constraint;
segment_rollout_signature = rollout_cache_signature( ...
    segment_rollout_constraint, cfg.time_steps, cfg.t_min, ...
    cfg.rollout_t_max, segment_x_init, segment_fixed_mask, ...
    segment_fixed_values, cfg.fixed_clf);
segment_rollout_signature.fixed_anchor_init_mode = fixed_anchor_init_mode;
segment_rollout_cache_hit = false;
fixed_clf_diagnostics = struct();
segment_rollout_cache_timer = tic;
if cfg.cache.rollout_enabled
    segment_rollout_cache_path = fullfile(this_dir, ...
        cfg.cache.second_level_rollout_path);
    segment_rollout_cache_dir = fileparts(segment_rollout_cache_path);
    if ~exist(segment_rollout_cache_dir, 'dir')
        mkdir(segment_rollout_cache_dir);
    end
    if isfile(segment_rollout_cache_path)
        cached_rollout = load(segment_rollout_cache_path);
        if isfield(cached_rollout, 'segment_rollout_times') && ...
                isfield(cached_rollout, 'segment_traj_path_10d')
            expected_segment_path_size = [cfg.time_steps + 1, ...
                n_segment_eval, n_segment_rows];
            cache_signature_ok = isfield(cached_rollout, ...
                'segment_rollout_signature') && isequaln( ...
                cached_rollout.segment_rollout_signature, ...
                segment_rollout_signature);
            if isequal(size(cached_rollout.segment_traj_path_10d), ...
                    expected_segment_path_size) && cache_signature_ok
                segment_rollout_times = cached_rollout.segment_rollout_times;
                segment_traj_path_10d = cached_rollout.segment_traj_path_10d;
                if isfield(cached_rollout, 'fixed_clf_diagnostics')
                    fixed_clf_diagnostics = ...
                        cached_rollout.fixed_clf_diagnostics;
                end
                segment_rollout_cache_hit = true;
                disp(['Loaded cached second-level rollout: ', ...
                    segment_rollout_cache_path]);
            else
                disp(['Cached second-level rollout shape mismatch. ', ...
                    'Recomputing rollout.']);
            end
        else
            disp('Cached second-level rollout is missing rollout data.');
        end
    end
end
if segment_rollout_cache_hit
    disp(['Second-level cached rollout load/check elapsed: ', ...
        num2str(toc(segment_rollout_cache_timer), '%.1f'), ' seconds']);
end
if ~segment_rollout_cache_hit
    second_rollout_timer = tic;
    [segment_rollout_times, segment_traj_path_10d, fixed_clf_diagnostics] = ...
        rk4_rollout(segment_model_collection, segment_x_init, cfg.t_min, ...
        cfg.rollout_t_max, cfg.time_steps, segment_rollout_constraint, ...
        segment_fixed_mask, segment_fixed_values, cfg.fixed_clf);
    disp(['Second-level RK4 rollout elapsed: ', ...
        num2str(toc(second_rollout_timer), '%.1f'), ' seconds']);
    if cfg.cache.rollout_enabled
        try
            save(segment_rollout_cache_path, 'segment_rollout_times', ...
                'segment_traj_path_10d', 'segment_rollout_signature', ...
                'fixed_clf_diagnostics');
        catch
            save(segment_rollout_cache_path, 'segment_rollout_times', ...
                'segment_traj_path_10d', ...
                'segment_rollout_signature', 'fixed_clf_diagnostics', ...
                '-v7.3');
        end
        disp(['Saved cached second-level rollout: ', ...
            segment_rollout_cache_path]);
    end
end

segment_traj_path_plot = zeros(size(segment_traj_path_10d));
for time_idx = 1:numel(segment_rollout_times)
    states_now = squeeze(segment_traj_path_10d(time_idx, :, :));
    states_plot = states_now' .* segment_data_transform.std + ...
        segment_data_transform.mean;
    segment_traj_path_plot(time_idx, :, :) = reshape(states_plot', ...
        1, n_segment_eval, []);
end

final_segment_data = squeeze(segment_traj_path_plot(end, :, :));
if size(final_segment_data, 1) == 1
    final_segment_data = reshape(final_segment_data, 1, []);
end
reconstructed_points_refined = stitch_segment_points(final_segment_data, ...
    n_second_level_eval, n_generation_segments, ...
    cfg.segment_points_per_segment);
reconstructed_points_refined = normalize_tangent_features( ...
    reconstructed_points_refined);
second_level_anchor_idx = 1:(cfg.segment_points_per_segment - 1): ...
    size(reconstructed_points_refined, 1);
anchor_reference_points = zeros(size(second_level_anchor_points, 1), ...
    n_second_level_eval, segment_feature_dim);
for traj_eval_idx = 1:n_eval
    for second_sample_idx = 1:n_second_level_samples
        second_eval_idx = (traj_eval_idx - 1) * n_second_level_samples + ...
            second_sample_idx;
        anchor_reference_points(:, second_eval_idx, :) = ...
            second_level_anchor_points(:, traj_eval_idx, :);
    end
end
second_level_anchor_error = max(abs( ...
    reconstructed_points_refined(second_level_anchor_idx, :, :) - ...
    anchor_reference_points), [], 'all');
[c0_mean, c0_max, c1_mean, c1_max] = ...
    compute_segment_connection_diagnostics(final_segment_data, ...
    n_second_level_eval, n_generation_segments, ...
    cfg.segment_points_per_segment);
target_reference_points = zeros(size(target_points_dense, 1), ...
    n_second_level_eval, size(target_points_dense, 3));
for traj_eval_idx = 1:n_eval
    for second_sample_idx = 1:n_second_level_samples
        second_eval_idx = (traj_eval_idx - 1) * n_second_level_samples + ...
            second_sample_idx;
        target_reference_points(:, second_eval_idx, :) = ...
            target_points_dense(:, traj_idx(traj_eval_idx), :);
    end
end
target_error = reconstructed_points_refined(:, :, 1:2) - ...
    target_reference_points(:, :, 1:2);
target_rmse_per_sample = squeeze(sqrt(mean(sum(target_error .^ 2, 3), 1)));
target_rmse_mean = mean(target_rmse_per_sample);
target_rmse_max = max(target_rmse_per_sample);
second_level_max_abs_coordinate = max(abs( ...
    reconstructed_points_refined(:, :, 1:2)), [], 'all');
target_max_abs_coordinate = max(abs(target_points_dense(:, :, 1:2)), ...
    [], 'all');
divergence_coordinate_threshold = target_max_abs_coordinate + 0.5;
diverged_sample_mask = squeeze(any(any(abs( ...
    reconstructed_points_refined(:, :, 1:2)) > ...
    divergence_coordinate_threshold, 3), 1));
second_level_diverged_sample_count = sum(diverged_sample_mask);
final_segment_state = squeeze(segment_traj_path_10d(end, :, :));
if size(final_segment_state, 1) == 1
    final_segment_state = reshape(final_segment_state, 1, []);
end
second_level_fixed_error = 0;
if any(segment_fixed_mask, 'all')
    second_level_fixed_error = max(abs( ...
        final_segment_state(segment_fixed_mask) - ...
        segment_fixed_values(segment_fixed_mask)), [], 'all');
end
disp(['Second-level reconstructed size: ', ...
    mat2str(size(reconstructed_points_refined))]);
disp(['Second-level anchor max error: ', ...
    num2str(second_level_anchor_error)]);
disp(['Second-level fixed-state max error: ', ...
    num2str(second_level_fixed_error)]);
disp(['Second-level C0 connection error mean: ', num2str(c0_mean)]);
disp(['Second-level C0 connection error max: ', num2str(c0_max)]);
disp(['Second-level C1 tangent angle error mean deg: ', ...
    num2str(c1_mean)]);
disp(['Second-level C1 tangent angle error max deg: ', ...
    num2str(c1_max)]);
disp(['Second-level target RMSE mean: ', num2str(target_rmse_mean)]);
disp(['Second-level target RMSE max: ', num2str(target_rmse_max)]);
disp(['Second-level max abs coordinate: ', ...
    num2str(second_level_max_abs_coordinate)]);
disp(['Second-level diverged sample count: ', ...
    num2str(second_level_diverged_sample_count)]);

source_points_refined = source_points(:, traj_idx, :);
initial_segment_state = squeeze(segment_traj_path_10d(1, :, :));
if size(initial_segment_state, 1) == 1
    initial_segment_state = reshape(initial_segment_state, 1, []);
end
segment_source_points_standardized = stitch_segment_points(initial_segment_state, ...
    n_second_level_eval, n_generation_segments, ...
    cfg.segment_points_per_segment);
final_trajectory_data = reshape(permute(reconstructed_points_refined, ...
    [3, 1, 2]), [], n_second_level_eval)';
reconstruction_source_idx = repelem(traj_idx(:) - 1, ...
    n_second_level_samples);

target_points_plot = target_points_dense;
source_points_plot = source_points_refined;
reconstructed_points_plot = reconstructed_points_refined;
uncertainty_times = segment_rollout_times;
disp('Evaluating second-level LoG-GP predictive uncertainty...');
uncertainty_values = evaluate_rollout_uncertainty(segment_model_collection, ...
    segment_rollout_times, segment_traj_path_10d, segment_fixed_mask);
uncertainty_group_std = compute_uncertainty_group_std(uncertainty_values, ...
    segment_feature_dim);
uncertainty_threshold = second_level_uncertainty_threshold;
uncertainty_plot_title = ...
    'Second-Level LoG-GP Predictive Variance Along Segment Rollout';
uncertainty_group_plot_title = ...
    'Second-Level LoG-GP Predictive Std By Feature Group';
uncertainty_level_label = 'second-level';

%% Third-Level Flow Model
if isfield(cfg, 'enable_third_level') && cfg.enable_third_level
    disp('Building third-level segment flow-matching data...');
    third_level_stride = cfg.segment_points_per_segment - 1;
    if isfield(cfg, 'third_level_window_stride')
        third_level_stride = cfg.third_level_window_stride;
    end
    n_third_level_training_points = (size(target_points_dense, 1) - 1) * ...
        (cfg.segment_points_per_segment - 1) + 1;
    target_points_fine = generate_original_training_points_2d( ...
        n_third_level_training_points, size(target_points_dense, 2));
    third_level_anchor_idx = 1:(cfg.segment_points_per_segment - 1): ...
        size(target_points_fine, 1);
    third_anchor_reconstruction_error = max(abs( ...
        target_points_fine(third_level_anchor_idx, :, :) - ...
        target_points_dense), [], 'all');
    disp(['Generated fine target anchor reconstruction error: ', ...
        num2str(third_anchor_reconstruction_error)]);

    [third_segment_s_slices, third_segment_x_slices, ...
        third_segment_y_slices, ~, ~, third_segment_data_transform] = ...
        build_local_increment_sliding_window_training_data( ...
        target_points_fine, cfg.t_min, 1.0, cfg.n_time_slices, ...
        cfg.segment_points_per_segment, third_level_stride);

    third_segment_gp = cfg.gp;
    if isfield(cfg.gp, 'second_level_length_scale_vec')
        third_segment_gp.length_scale_vec = cfg.gp.second_level_length_scale_vec;
    end
    third_segment_gp.training_accuracy_threshold = ...
        cfg.gp.third_level_training_accuracy_threshold;
    third_segment_gp.hyperparameter_mat_path = fullfile('outputs', ...
        ['LoG_GP_ThirdLevel_Training_Hyperparameter_', ...
        cfg.slope.cache_tag, '_block_stride_', ...
        num2str(third_level_stride), '.mat']);
    if isfield(cfg, 'small_sample') && isfield(cfg.small_sample, ...
            'enabled') && cfg.small_sample.enabled
        third_segment_gp.hyperparameter_mat_path = fullfile('outputs', ...
            ['LoG_GP_ThirdLevel_Training_Hyperparameter_', ...
            cfg.slope.cache_tag, '_block_stride_', ...
            num2str(third_level_stride), '_small.mat']);
        if isfield(cfg.small_sample, 'reuse_full_hyperparameters') && ...
                ~cfg.small_sample.reuse_full_hyperparameters
            third_segment_gp.hyperparameter_mat_path = "";
        end
    end
    third_segment_gp.n_pretrain = min(cfg.gp.third_level_n_pretrain, ...
        size(third_segment_x_slices, 1) * size(third_segment_x_slices, 2));
    third_segment_gp.pretrain_output_idx = ...
        1:(segment_feature_dim * cfg.segment_points_per_segment);
    if isfield(cfg.gp, 'second_level_hyperparameter_grouping')
        third_segment_gp.hyperparameter_grouping = ...
            cfg.gp.second_level_hyperparameter_grouping;
    end
    third_segment_gp.max_local_gp_quantity = ceil(2.0 * ...
        size(third_segment_x_slices, 1) * ...
        size(third_segment_x_slices, 2) / ...
        third_segment_gp.max_local_data_quantity);
    use_manual_third_level_hyperparameters = ...
        isfield(cfg.gp, 'third_level_use_manual_hyperparameters') && ...
        cfg.gp.third_level_use_manual_hyperparameters;
    if use_manual_third_level_hyperparameters
        disp('Using manual third-level segment LoG-GP hyperparameters from config...');
        third_segment_gp.length_scale_mat = cfg.gp.third_level_length_scale_mat;
        third_segment_gp.length_scale_vec = ...
            median(third_segment_gp.length_scale_mat, 2);
        third_segment_gp.signal_std_vec = cfg.gp.third_level_signal_std_vec;
        third_segment_gp.signal_std = median(third_segment_gp.signal_std_vec);
        third_segment_gp.noise_std_vec = cfg.gp.third_level_noise_std_vec;
        third_segment_gp.noise_std = median(third_segment_gp.noise_std_vec);
        third_segment_gp.optimize_hyperparameters = false;
        third_segment_gp.save_hyperparameters = false;
    else
        disp('Optimizing or loading third-level segment LoG-GP hyperparameters...');
        rng(cfg.second_level_hyperparameter_seed + 100);
        third_segment_gp = optimize_gp_hyperparameters(third_segment_x_slices, ...
            third_segment_y_slices, third_segment_gp, third_segment_s_slices);
        override_third_level_noise = isfield(cfg.gp, ...
            'third_level_override_noise_std') && ...
            cfg.gp.third_level_override_noise_std;
        if override_third_level_noise && ...
                isfield(cfg.gp, 'third_level_noise_std') && ...
                ~isempty(cfg.gp.third_level_noise_std)
            third_segment_gp.noise_std = cfg.gp.third_level_noise_std;
            third_segment_gp.noise_std_vec = cfg.gp.third_level_noise_std * ...
                ones(1, size(third_segment_y_slices, 3));
            disp(['Overriding third-level LoG-GP noise std SigmaN to ', ...
                num2str(cfg.gp.third_level_noise_std)]);
        end
    end

    disp('Fitting third-level segment LoG-GP flow model...');
    rng(cfg.second_level_fit_seed + 100);
    third_model_cache_enabled = cfg.cache.model_enabled;
    if third_model_cache_enabled && isfield(cfg.cache, 'third_level_model_path')
        third_model_cache_path = fullfile(this_dir, ...
            cfg.cache.third_level_model_path);
    else
        third_model_cache_path = "";
    end
    third_fit_timer = tic;
    third_segment_model_collection = fit_or_load_loggp_model( ...
        third_segment_s_slices, third_segment_x_slices, ...
        third_segment_y_slices, third_segment_gp, third_model_cache_path, ...
        'third-level');
    disp(['Third-level LoG-GP fit/load elapsed: ', ...
        num2str(toc(third_fit_timer), '%.1f'), ' seconds']);

    third_level_anchor_points = reconstructed_points_refined;
    n_third_generation_segments = size(third_level_anchor_points, 1) - 1;
    n_third_anchor_eval = size(third_level_anchor_points, 2);
    if isfield(cfg, 'third_level_generation_samples')
        n_third_samples = cfg.third_level_generation_samples;
    else
        n_third_samples = 1;
    end
    n_third_level_eval = n_third_anchor_eval * n_third_samples;
    n_third_segment_eval = n_third_level_eval * n_third_generation_segments;
    n_third_segment_rows = segment_feature_dim * cfg.segment_points_per_segment;
    rng(cfg.second_level_rollout_seed + 100);
    third_segment_x_init = randn(n_third_segment_eval, ...
        n_third_segment_rows);
    third_segment_fixed_mask = false(n_third_segment_eval, ...
        n_third_segment_rows);
    third_segment_fixed_values = third_segment_x_init;
    third_endpoint_clf_A = zeros(n_third_segment_eval, 2, ...
        n_third_segment_rows);
    third_endpoint_clf_b = zeros(n_third_segment_eval, 2);
    third_fixed_anchor_init_mode = 'anchor';
    if isfield(cfg, 'fixed_anchor_init_mode') && ...
            ~isempty(cfg.fixed_anchor_init_mode)
        third_fixed_anchor_init_mode = cfg.fixed_anchor_init_mode;
    end
    third_fixed_clf_cfg = cfg.fixed_clf;
    if isfield(cfg, 'third_level_fixed_clf_alpha') && ...
            ~isempty(cfg.third_level_fixed_clf_alpha)
        third_fixed_clf_cfg.alpha = cfg.third_level_fixed_clf_alpha;
    end
    third_segment_start_idx = 1:segment_feature_dim;
    third_segment_end_idx = (n_third_segment_rows - segment_feature_dim + 1): ...
        n_third_segment_rows;
    third_end_tangent_idx = third_segment_end_idx(3:min(4, segment_feature_dim));
    for anchor_eval_idx = 1:n_third_anchor_eval
        for third_sample_idx = 1:n_third_samples
            third_eval_idx = (anchor_eval_idx - 1) * n_third_samples + ...
                third_sample_idx;
            for segment_idx = 1:n_third_generation_segments
                sample_idx = (third_eval_idx - 1) * ...
                    n_third_generation_segments + segment_idx;
                start_point = squeeze(third_level_anchor_points(segment_idx, ...
                    anchor_eval_idx, :))';
                end_point = squeeze(third_level_anchor_points(segment_idx + 1, ...
                    anchor_eval_idx, :))';
                reference_segment_local = anchor_segment_to_increment_state( ...
                    start_point, end_point, cfg.segment_points_per_segment, ...
                    segment_feature_dim);
                reference_segment_row = reshape(reference_segment_local', 1, []);
                reference_segment_state = (reference_segment_row - ...
                    third_segment_data_transform.mean') ./ ...
                    third_segment_data_transform.std';
                if strcmp(third_fixed_anchor_init_mode, 'anchor')
                    third_segment_x_init(sample_idx, :) = reference_segment_state;
                end
                [endpoint_A, endpoint_b] = endpoint_increment_clf_map( ...
                    end_point(1:2), third_segment_data_transform.mean', ...
                    third_segment_data_transform.std', segment_feature_dim, ...
                    cfg.segment_points_per_segment);
                third_endpoint_clf_A(sample_idx, :, :) = endpoint_A;
                third_endpoint_clf_b(sample_idx, :) = endpoint_b;
                fixed_idx = unique([third_segment_start_idx, ...
                    third_end_tangent_idx]);
                third_segment_fixed_mask(sample_idx, fixed_idx) = true;
                third_segment_fixed_values(sample_idx, fixed_idx) = ...
                    reference_segment_state(fixed_idx);
            end
        end
    end

    disp('Running third-level segment RK4 rollout...');
    third_level_uncertainty_threshold = ...
        cfg.gp.third_level_generation_accuracy_threshold;
    third_level_time_steps = cfg.time_steps;
    if isfield(cfg, 'third_level_time_steps') && ...
            ~isempty(cfg.third_level_time_steps)
        third_level_time_steps = cfg.third_level_time_steps;
    end
    third_level_rollout_t_max = cfg.rollout_t_max;
    if isfield(cfg, 'third_level_rollout_t_max') && ...
            ~isempty(cfg.third_level_rollout_t_max)
        third_level_rollout_t_max = cfg.third_level_rollout_t_max;
    end
    third_segment_variance_constraint = cfg.variance_constraint;
    third_segment_variance_constraint.uncertainty_max = ...
        third_level_uncertainty_threshold;
    third_segment_variance_constraint.generation_accuracy_threshold = ...
        third_level_uncertainty_threshold;
    if isfield(cfg.variance_constraint, 'third_level_alpha_gain')
        third_segment_variance_constraint.alpha_gain = ...
            cfg.variance_constraint.third_level_alpha_gain;
    end
    if isfield(cfg.variance_constraint, 'third_level_omega_gain')
        third_segment_variance_constraint.omega_gain = ...
            cfg.variance_constraint.third_level_omega_gain;
    end
    if isfield(cfg.variance_constraint, 'third_level_grad_tol')
        third_segment_variance_constraint.grad_tol = ...
            cfg.variance_constraint.third_level_grad_tol;
    end
    if isfield(cfg.variance_constraint, 'third_level_ptzf_gamma')
        third_segment_variance_constraint.ptzf_gamma = ...
            cfg.variance_constraint.third_level_ptzf_gamma;
    end
    if isfield(cfg.variance_constraint, 'third_level_ptzf_initial_bound')
        third_segment_variance_constraint.ptzf_initial_bound = ...
            cfg.variance_constraint.third_level_ptzf_initial_bound;
    end
    if isfield(cfg.variance_constraint, 'third_level_slack_weight')
        third_segment_variance_constraint.slack_weight = ...
            cfg.variance_constraint.third_level_slack_weight;
    end
    if isfield(cfg.variance_constraint, 'third_level_diagnostics')
        third_segment_variance_constraint.diagnostics = ...
            cfg.variance_constraint.third_level_diagnostics;
    end
    if isfield(cfg.variance_constraint, 'third_level_diagnostics_version')
        third_segment_variance_constraint.diagnostics_version = ...
            cfg.variance_constraint.third_level_diagnostics_version;
    end
    third_rollout_signature = rollout_cache_signature( ...
        third_segment_variance_constraint, third_level_time_steps, cfg.t_min, ...
        third_level_rollout_t_max, third_segment_x_init, third_segment_fixed_mask, ...
        third_segment_fixed_values, third_fixed_clf_cfg);
    third_rollout_signature.fixed_anchor_init_mode = ...
        third_fixed_anchor_init_mode;
    third_rollout_signature.local_increment = true;
    third_rollout_signature.no_delta_increment = true;
    third_rollout_signature.source_init_model = numeric_cache_signature( ...
        third_segment_x_init);
    third_rollout_signature.endpoint_clf_A = numeric_cache_signature( ...
        third_endpoint_clf_A);
    third_rollout_signature.endpoint_clf_b = numeric_cache_signature( ...
        third_endpoint_clf_b);
    third_fixed_clf_cfg.linear_clf.enabled = true;
    third_fixed_clf_cfg.linear_clf.A = third_endpoint_clf_A;
    third_fixed_clf_cfg.linear_clf.b = third_endpoint_clf_b;

    third_rollout_cache_hit = false;
    third_rollout_cache_enabled = cfg.cache.rollout_enabled;
    third_rollout_cache_timer = tic;
    if third_rollout_cache_enabled && isfield(cfg.cache, 'third_level_rollout_path')
        third_rollout_cache_path = fullfile(this_dir, ...
            cfg.cache.third_level_rollout_path);
        third_rollout_cache_dir = fileparts(third_rollout_cache_path);
        if ~exist(third_rollout_cache_dir, 'dir')
            mkdir(third_rollout_cache_dir);
        end
        if isfile(third_rollout_cache_path)
            cached_third_rollout = load(third_rollout_cache_path);
            if isfield(cached_third_rollout, 'third_rollout_times') && ...
                    isfield(cached_third_rollout, 'third_traj_path_10d')
                expected_third_path_size = [third_level_time_steps + 1, ...
                    n_third_segment_eval, n_third_segment_rows];
                cache_signature_ok = isfield(cached_third_rollout, ...
                    'third_rollout_signature') && isequaln( ...
                    cached_third_rollout.third_rollout_signature, ...
                    third_rollout_signature);
                if isequal(size(cached_third_rollout.third_traj_path_10d), ...
                        expected_third_path_size) && cache_signature_ok
                    third_rollout_times = ...
                        cached_third_rollout.third_rollout_times;
                    third_traj_path_10d = ...
                        cached_third_rollout.third_traj_path_10d;
                    if isfield(cached_third_rollout, ...
                            'third_fixed_clf_diagnostics')
                        third_fixed_clf_diagnostics = ...
                            cached_third_rollout.third_fixed_clf_diagnostics;
                    end
                    third_rollout_cache_hit = true;
                    disp(['Loaded cached third-level rollout: ', ...
                        third_rollout_cache_path]);
                else
                    disp(['Cached third-level rollout shape mismatch. ', ...
                        'Recomputing rollout.']);
                end
            end
        end
    end
    if third_rollout_cache_hit
        disp(['Third-level cached rollout load/check elapsed: ', ...
            num2str(toc(third_rollout_cache_timer), '%.1f'), ' seconds']);
    end
    if ~third_rollout_cache_hit
        third_rollout_timer = tic;
        [third_rollout_times, third_traj_path_10d, ...
                third_fixed_clf_diagnostics] = rk4_rollout( ...
            third_segment_model_collection, third_segment_x_init, ...
            cfg.t_min, third_level_rollout_t_max, third_level_time_steps, ...
            third_segment_variance_constraint, third_segment_fixed_mask, ...
            third_segment_fixed_values, third_fixed_clf_cfg);
        disp(['Third-level RK4 rollout elapsed: ', ...
            num2str(toc(third_rollout_timer), '%.1f'), ' seconds']);
        if third_rollout_cache_enabled && exist('third_rollout_cache_path', 'var')
            try
                save(third_rollout_cache_path, 'third_rollout_times', ...
                    'third_traj_path_10d', 'third_rollout_signature', ...
                    'third_fixed_clf_diagnostics');
            catch
                save(third_rollout_cache_path, 'third_rollout_times', ...
                    'third_traj_path_10d', 'third_rollout_signature', ...
                    'third_fixed_clf_diagnostics', '-v7.3');
            end
            disp(['Saved cached third-level rollout: ', ...
                third_rollout_cache_path]);
        end
    end

    third_traj_path_plot = zeros(size(third_traj_path_10d));
    for time_idx = 1:numel(third_rollout_times)
        states_now = squeeze(third_traj_path_10d(time_idx, :, :));
        if size(states_now, 1) == 1
            states_now = reshape(states_now, 1, []);
        end
        states_local = states_now' .* third_segment_data_transform.std + ...
            third_segment_data_transform.mean;
        states_plot = local_increment_rows_to_global(states_local', ...
            segment_feature_dim, cfg.segment_points_per_segment);
        third_traj_path_plot(time_idx, :, :) = reshape(states_plot, ...
            1, n_third_segment_eval, n_third_segment_rows);
    end
    final_third_segment_data = squeeze(third_traj_path_plot(end, :, :));
    if size(final_third_segment_data, 1) == 1
        final_third_segment_data = reshape(final_third_segment_data, 1, []);
    end
    reconstructed_points_fine = stitch_segment_points( ...
        final_third_segment_data, n_third_level_eval, ...
        n_third_generation_segments, cfg.segment_points_per_segment);
    reconstructed_points_fine = normalize_tangent_features( ...
        reconstructed_points_fine);
    initial_third_segment_data = squeeze(third_traj_path_plot(1, :, :));
    if size(initial_third_segment_data, 1) == 1
        initial_third_segment_data = reshape(initial_third_segment_data, 1, []);
    end
    third_source_points_raw = stitch_segment_points( ...
        initial_third_segment_data, n_third_level_eval, ...
        n_third_generation_segments, cfg.segment_points_per_segment);
    third_source_points_raw = normalize_tangent_features( ...
        third_source_points_raw);

    third_output_anchor_idx = 1:(cfg.segment_points_per_segment - 1): ...
        size(reconstructed_points_fine, 1);
    third_level_anchor_reference_points = zeros(size(third_level_anchor_points, 1), ...
        n_third_level_eval, size(third_level_anchor_points, 3));
    for anchor_eval_idx = 1:n_third_anchor_eval
        for third_sample_idx = 1:n_third_samples
            third_eval_idx = (anchor_eval_idx - 1) * n_third_samples + ...
                third_sample_idx;
            third_level_anchor_reference_points(:, third_eval_idx, :) = ...
                third_level_anchor_points(:, anchor_eval_idx, :);
        end
    end
    third_level_anchor_error = max(abs( ...
        reconstructed_points_fine(third_output_anchor_idx, :, :) - ...
        third_level_anchor_reference_points), [], 'all');
    third_target_reference_points = zeros(size(target_points_fine, 1), ...
        n_third_level_eval, size(target_points_fine, 3));
    for traj_eval_idx = 1:n_eval
        for second_sample_idx = 1:n_second_level_samples
            anchor_eval_idx = (traj_eval_idx - 1) * n_second_level_samples + ...
                second_sample_idx;
            for third_sample_idx = 1:n_third_samples
                third_eval_idx = (anchor_eval_idx - 1) * n_third_samples + ...
                    third_sample_idx;
                third_target_reference_points(:, third_eval_idx, :) = ...
                    target_points_fine(:, traj_idx(traj_eval_idx), :);
            end
        end
    end
    third_target_error = reconstructed_points_fine(:, :, 1:2) - ...
        third_target_reference_points(:, :, 1:2);
    third_target_rmse_per_sample = squeeze(sqrt(mean(sum( ...
        third_target_error .^ 2, 3), 1)));
    third_target_rmse_mean = mean(third_target_rmse_per_sample);
    third_target_rmse_max = max(third_target_rmse_per_sample);
    third_level_max_abs_coordinate = max(abs( ...
        reconstructed_points_fine(:, :, 1:2)), [], 'all');
    third_target_max_abs_coordinate = max(abs(target_points_fine(:, :, 1:2)), ...
        [], 'all');
    third_divergence_coordinate_threshold = ...
        third_target_max_abs_coordinate + 0.5;
    third_diverged_sample_mask = squeeze(any(any(abs( ...
        reconstructed_points_fine(:, :, 1:2)) > ...
        third_divergence_coordinate_threshold, 3), 1));
    third_level_diverged_sample_count = sum(third_diverged_sample_mask);
    disp(['Third-level reconstructed size: ', ...
        mat2str(size(reconstructed_points_fine))]);
    disp(['Third-level anchor max error: ', ...
        num2str(third_level_anchor_error)]);
    disp(['Third-level target RMSE mean: ', ...
        num2str(third_target_rmse_mean)]);
    disp(['Third-level target RMSE max: ', ...
        num2str(third_target_rmse_max)]);
    disp(['Third-level max abs coordinate: ', ...
        num2str(third_level_max_abs_coordinate)]);
    disp(['Third-level diverged sample count: ', ...
        num2str(third_level_diverged_sample_count)]);

    skip_uncertainty_evaluation = isfield(cfg, ...
        'skip_uncertainty_evaluation') && cfg.skip_uncertainty_evaluation;
    if skip_uncertainty_evaluation
        disp('Skipping third-level LoG-GP predictive uncertainty evaluation.');
        uncertainty_values = [];
        uncertainty_group_std = [];
    else
        disp('Evaluating third-level LoG-GP predictive uncertainty...');
        uncertainty_values = evaluate_rollout_uncertainty( ...
            third_segment_model_collection, third_rollout_times, ...
            third_traj_path_10d, third_segment_fixed_mask);
        uncertainty_group_std = compute_uncertainty_group_std( ...
            uncertainty_values, segment_feature_dim);
    end
    uncertainty_times = third_rollout_times;
    uncertainty_threshold = third_level_uncertainty_threshold;
    uncertainty_plot_title = ...
        'Third-Level LoG-GP Predictive Variance Along Segment Rollout';
    uncertainty_group_plot_title = ...
        'Third-Level LoG-GP Predictive Std By Feature Group';
    uncertainty_level_label = 'third-level';

    target_points_plot = target_points_fine;
    source_points_plot = third_source_points_raw;
    reconstructed_points_plot = reconstructed_points_fine;
    target_reference_points = third_target_reference_points;
    reconstructed_points_refined = reconstructed_points_fine;
    final_trajectory_data = reshape(permute(reconstructed_points_fine, ...
        [3, 1, 2]), [], n_third_level_eval)';
    reconstruction_source_idx = repelem(reconstruction_source_idx(:), ...
        n_third_samples);
    n_generation_segments = n_third_generation_segments;
    n_second_level_eval = n_third_level_eval;
    segment_rollout_times = third_rollout_times;
    segment_traj_path_plot = third_traj_path_plot;
    segment_traj_path_10d = third_traj_path_10d;
end

%% Plot Results
disp('Plotting and exporting results...');
plot_cfg = cfg;
plot_cfg.n_trajectories = size(reconstructed_points_plot, 2);
plot_cfg.reference_points = target_reference_points;
plot_cfg.reference_label = 'Training reference curve';
plot_cfg.reference_curve_count = n_eval;
if isfield(cfg, 'enable_third_level') && cfg.enable_third_level
    plot_cfg.sample_curve_label = 'Stage 3 sample curves';
    plot_cfg.generated_point_label = 'Stage 3 generated points';
    plot_cfg.anchor_label = 'Stage 2 fixed anchors';
end
plot_results(plot_cfg, target_points_plot, source_points_plot, ...
    reconstructed_points_plot);
if ~isempty(uncertainty_values)
    plot_uncertainty_vs_time(cfg, uncertainty_times, uncertainty_values, ...
        uncertainty_plot_title, uncertainty_threshold);
    plot_uncertainty_group_std_vs_time(cfg, uncertainty_times, ...
        uncertainty_group_std, uncertainty_threshold, uncertainty_group_plot_title);
else
    disp('Skipping uncertainty plots because uncertainty evaluation is disabled.');
end
animation_path = "";
if cfg.animation.enabled && n_eval > 0
    animation_segment_count = n_generation_segments;
    animation_point_count = size(reconstructed_points_refined, 1);
    animation_eval_count = n_second_level_eval;
    animation_traj_path = segment_traj_path_plot;
    animation_times = segment_rollout_times;
    animation_use_third_global_rollout = false;
    if isfield(cfg, 'enable_third_level') && cfg.enable_third_level && ...
            exist('reconstructed_points_fine', 'var')
        animation_segment_count = n_third_generation_segments;
        animation_point_count = size(reconstructed_points_fine, 1);
        animation_eval_count = n_third_level_eval;
        animation_traj_path = third_traj_path_plot;
        animation_times = third_rollout_times;
        animation_use_third_global_rollout = true;
    end
    animation_nr = min(cfg.animation.trajectory_nr, animation_eval_count);
    segment_animation_idx = ...
        ((animation_nr - 1) * animation_segment_count + 1): ...
        (animation_nr * animation_segment_count);
    traj_path_single = zeros(numel(animation_times), ...
        segment_feature_dim * animation_point_count);
    for time_idx = 1:numel(animation_times)
        segment_states_now = squeeze(animation_traj_path( ...
            time_idx, segment_animation_idx, :));
        stitched_now = stitch_segment_points(segment_states_now, ...
            1, animation_segment_count, cfg.segment_points_per_segment);
        stitched_now = squeeze(stitched_now);
        traj_path_single(time_idx, :) = reshape(stitched_now', 1, []);
    end
    animation_source_points = reshape(traj_path_single(1, :), ...
        segment_feature_dim, [])';
    animation_cfg = cfg;
    if animation_use_third_global_rollout
        animation_cfg.animation.space_label = 'Original Global Space';
        animation_cfg.animation.anchor_label = 'fixed dims';
    end
    animation_path = animate_single_trajectory(animation_cfg, animation_times, ...
        traj_path_single, animation_source_points);
end

%% Export Results
output_dir = fullfile(this_dir, 'outputs');
output_enabled = ~isfield(cfg, 'output') || ...
    ~isfield(cfg.output, 'enabled') || cfg.output.enabled;
if output_enabled
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    reconstruction_headers = ["trajectory_index", ...
        arrayfun(@(idx) "coord_" + string(idx), ...
        1:size(final_trajectory_data, 2))];
    reconstruction_table = [reconstruction_source_idx(:), final_trajectory_data];
    writematrix(reconstruction_headers, fullfile(output_dir, ...
        'trajectory_reconstruction_samples.csv'));
    writematrix(reconstruction_table, ...
        fullfile(output_dir, 'trajectory_reconstruction_samples.csv'), ...
        'WriteMode', 'append');

    if ~isempty(uncertainty_values)
        n_uncertainty_paths = size(uncertainty_values, 2);
        n_uncertainty_outputs = size(uncertainty_values, 3);
        traj_gp_var_table = zeros(numel(uncertainty_times), ...
            1 + n_uncertainty_paths * n_uncertainty_outputs);
        traj_gp_var_table(:, 1) = uncertainty_times;
        traj_gp_var_headers = strings(1, ...
            1 + n_uncertainty_paths * n_uncertainty_outputs);
        traj_gp_var_headers(1) = "s";
        for output_idx = 1:size(uncertainty_values, 3)
            column_idx = 2 + (output_idx - 1) * n_uncertainty_paths;
            column_range = column_idx:(column_idx + n_uncertainty_paths - 1);
            traj_gp_var_table(:, column_range) = ...
                uncertainty_values(:, :, output_idx);
            traj_gp_var_headers(column_range) = ...
                arrayfun(@(idx) "path_" + string(idx) + "_dim_" + ...
                string(output_idx) + "_variance", 0:(n_uncertainty_paths - 1));
        end
        writematrix(traj_gp_var_headers, fullfile(output_dir, ...
            'trajectory_gp_predictive_uncertainties.csv'));
        writematrix(traj_gp_var_table, ...
            fullfile(output_dir, 'trajectory_gp_predictive_uncertainties.csv'), ...
            'WriteMode', 'append');
    else
        disp('Skipping uncertainty CSV export because evaluation is disabled.');
    end
else
    disp('Output export disabled; skipping CSV result files.');
end

%% Console Summary
disp(['Training trajectories loaded: ', num2str(size(target_data, 2))]);
disp(['Trajectory points per sample: ', num2str(numel(trajectory_t_slices))]);
disp('Slope feature mode: [dx/ds, dy/ds] tangent');
disp(['Features per trajectory point: ', num2str(size(target_points, 3))]);
disp(['ODE s slices: ', num2str(numel(s_slices))]);
disp(['Rollout time interval: [', num2str(cfg.t_min), ', ', ...
    num2str(cfg.rollout_t_max), ']']);
disp(['LoG-GP input dimension: ', num2str(size(x_slices, 3) + 1)]);
disp(['LoG-GP output dimension: ', num2str(size(y_slices, 3))]);
disp(['GP model type: ', model_collection.model_type]);
disp(['LoG-GP noise std SigmaN: ', num2str(cfg.gp.noise_std)]);
disp(['LoG-GP signal std SigmaF: ', num2str(cfg.gp.signal_std)]);
disp(['LoG-GP length scale SigmaL: ', mat2str(cfg.gp.length_scale_vec(:)', 4)]);
disp(['Second-level LoG-GP noise std SigmaN: ', ...
    num2str(segment_gp.noise_std)]);
disp(['Second-level LoG-GP signal std SigmaF: ', ...
    num2str(segment_gp.signal_std)]);
disp(['Second-level LoG-GP length scale SigmaL: ', ...
    mat2str(segment_gp.length_scale_vec(:)', 4)]);
disp(['Second-level sliding-window samples: ', ...
    num2str(size(segment_x_slices, 2))]);
disp(['Second-level LoG-GP training pairs: ', ...
    num2str(segment_model_collection.n_training_pairs)]);
disp(['Second-level LoG-GP added pairs per output after accuracy check: ', ...
    mat2str(segment_model_collection.n_added_per_output(:)')]);
disp(['Second-level LoG-GP skipped pairs per output after accuracy check: ', ...
    mat2str(segment_model_collection.n_skipped_per_output(:)')]);
disp(['Second-level generation uncertainty threshold: ', ...
    num2str(second_level_uncertainty_threshold)]);
disp(['Second-level generated sample curves: ', ...
    num2str(n_second_level_eval)]);
if exist('third_segment_model_collection', 'var')
    disp(['Third-level LoG-GP noise std SigmaN: ', ...
        num2str(third_segment_gp.noise_std)]);
    if isfield(third_segment_gp, 'noise_std_vec')
        disp(['Third-level LoG-GP noise std SigmaN per output: ', ...
            mat2str(third_segment_gp.noise_std_vec(:)', 4)]);
    end
    disp(['Third-level block-window samples: ', ...
        num2str(size(third_segment_x_slices, 2))]);
    disp(['Third-level LoG-GP training pairs: ', ...
        num2str(third_segment_model_collection.n_training_pairs)]);
    disp(['Third-level LoG-GP added pairs per output after accuracy check: ', ...
        mat2str(third_segment_model_collection.n_added_per_output(:)')]);
    disp(['Third-level LoG-GP skipped pairs per output after accuracy check: ', ...
        mat2str(third_segment_model_collection.n_skipped_per_output(:)')]);
    disp(['Third-level generation uncertainty threshold: ', ...
        num2str(third_level_uncertainty_threshold)]);
end
disp(['Total training trajectories: ', num2str(size(x_slices, 2))]);
disp(['Rolled-out trajectories: ', num2str(n_eval)]);
disp(['First-level generation uncertainty threshold: ', ...
    num2str(cfg.variance_constraint.uncertainty_max)]);
disp(['First-level anchor position RMSE mean: ', ...
    num2str(mean(first_level_position_rmse))]);
disp(['First-level anchor position RMSE max: ', ...
    num2str(max(first_level_position_rmse))]);
if all(isfinite(first_level_tangent_angle_error))
    disp(['First-level anchor tangent angle error mean deg: ', ...
        num2str(mean(first_level_tangent_angle_error))]);
    disp(['First-level anchor tangent angle error max deg: ', ...
        num2str(max(first_level_tangent_angle_error))]);
end
if output_enabled
    disp(['Uncertainty-vs-time figure: ', ...
        fullfile(output_dir, 'trajectory_gp_uncertainty_vs_time_matlab.emf')]);
else
    disp('Uncertainty-vs-time figure export disabled.');
end
if strlength(animation_path) > 0
    disp(['Single-trajectory animation: ', char(animation_path)]);
end
if ~isempty(uncertainty_group_std)
    disp(['Final plotted ', uncertainty_level_label, ...
        ' LoG-GP total std mean: ', ...
        num2str(mean(uncertainty_group_std(end, :, 3)))]);
    disp(['Final plotted ', uncertainty_level_label, ...
        ' LoG-GP total std max: ', ...
        num2str(max(uncertainty_group_std(end, :, 3)))]);
    disp(['Final plotted ', uncertainty_level_label, ...
        ' LoG-GP position std mean: ', ...
        num2str(mean(uncertainty_group_std(end, :, 1)))]);
    disp(['Final plotted ', uncertainty_level_label, ...
        ' LoG-GP position std max: ', ...
        num2str(max(uncertainty_group_std(end, :, 1)))]);
    disp(['Final plotted ', uncertainty_level_label, ...
        ' LoG-GP tangent std mean: ', ...
        num2str(mean(uncertainty_group_std(end, :, 2)))]);
    disp(['Final plotted ', uncertainty_level_label, ...
        ' LoG-GP tangent std max: ', ...
        num2str(max(uncertainty_group_std(end, :, 2)))]);
else
    disp(['Final plotted ', uncertainty_level_label, ...
        ' LoG-GP std summary skipped.']);
end

function signature = rollout_cache_signature(constraint_cfg, n_steps, t0, t1, ...
    x_init, fixed_mask, fixed_values, extra_cfg)
signature.time_steps = n_steps;
signature.t0 = t0;
signature.t1 = t1;
signature.uncertainty_max = struct_field_or_nan(constraint_cfg, ...
    'uncertainty_max');
signature.generation_accuracy_threshold = struct_field_or_nan( ...
    constraint_cfg, 'generation_accuracy_threshold');
signature.alpha_gain = struct_field_or_nan(constraint_cfg, 'alpha_gain');
signature.omega_gain = struct_field_or_nan(constraint_cfg, 'omega_gain');
signature.diagnostics_version = struct_field_or_nan(constraint_cfg, ...
    'diagnostics_version');
signature.grad_tol = struct_field_or_nan(constraint_cfg, 'grad_tol');
signature.ptzf_gamma = struct_field_or_nan(constraint_cfg, 'ptzf_gamma');
signature.ptzf_initial_bound = struct_field_or_nan(constraint_cfg, ...
    'ptzf_initial_bound');
signature.slack_weight = struct_field_or_nan(constraint_cfg, ...
    'slack_weight');
if nargin >= 5
    signature.x_init = numeric_cache_signature(x_init);
end
if nargin >= 6
    signature.fixed_mask = numeric_cache_signature(double(fixed_mask));
end
if nargin >= 7
    signature.fixed_values = numeric_cache_signature(fixed_values);
end
if nargin >= 8 && ~isempty(extra_cfg)
    signature.extra_cfg = struct_cache_signature(extra_cfg);
end
end

function value = struct_field_or_nan(value_struct, field_name)
if isstruct(value_struct) && isfield(value_struct, field_name) && ...
        ~isempty(value_struct.(field_name))
    value = value_struct.(field_name);
else
    value = nan;
end
end

function signature = numeric_cache_signature(values)
signature.size = size(values);
values = values(:);
values = values(isfinite(values));
if isempty(values)
    signature.sum = 0;
    signature.sumsq = 0;
    signature.first = 0;
    signature.last = 0;
    signature.max_abs = 0;
    return;
end
signature.sum = sum(values);
signature.sumsq = sum(values .^ 2);
signature.first = values(1);
signature.last = values(end);
signature.max_abs = max(abs(values));
end

function signature = struct_cache_signature(value_struct)
fields = fieldnames(value_struct);
signature = struct();
for field_idx = 1:numel(fields)
    field_name = fields{field_idx};
    field_value = value_struct.(field_name);
    if isnumeric(field_value) || islogical(field_value)
        signature.(field_name) = numeric_cache_signature(double(field_value));
    elseif ischar(field_value) || isstring(field_value)
        signature.(field_name) = char(field_value);
    elseif isstruct(field_value)
        signature.(field_name) = struct_cache_signature(field_value);
    end
end
end

function plot_first_level_diagnostics(cfg, target_points, source_points, ...
    reconstructed_points, reference_points)
figure('Color', 'w', 'WindowStyle', 'normal');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
for idx = 1:size(target_points, 2)
    curve = squeeze(target_points(:, idx, 1:2));
    plot(curve(:, 1), curve(:, 2), '.-', 'LineWidth', 0.7, ...
        'HandleVisibility', 'off');
end
grid on;
axis equal;
xlabel('x');
ylabel('y');
title('First-Level Training Targets');

nexttile;
hold on;
max_curves = min(size(reconstructed_points, 2), cfg.n_trajectories);
for idx = 1:max_curves
    source_curve = squeeze(source_points(:, idx, 1:2));
    reference_curve = squeeze(reference_points(:, idx, 1:2));
    rollout_curve = squeeze(reconstructed_points(:, idx, 1:2));
    plot(source_curve(:, 1), source_curve(:, 2), '--', ...
        'Color', [0.45, 0.45, 0.45], 'LineWidth', 1.0, ...
        'DisplayName', 'source');
    plot(reference_curve(:, 1), reference_curve(:, 2), 'r-', ...
        'LineWidth', 2.2, 'DisplayName', 'reference anchors');
    plot(rollout_curve(:, 1), rollout_curve(:, 2), 'bo-', ...
        'LineWidth', 1.4, 'MarkerFaceColor', 'b', ...
        'DisplayName', 'first-level rollout');
end
grid on;
axis equal;
xlabel('x');
ylabel('y');
title('First-Level Rollout Check');
legend('Location', 'best');
end

function points = normalize_tangent_features(points)
if size(points, 3) < 4
    return;
end
tangent_norm = sqrt(points(:, :, 3) .^ 2 + points(:, :, 4) .^ 2);
tangent_norm = max(tangent_norm, eps);
points(:, :, 3) = points(:, :, 3) ./ tangent_norm;
points(:, :, 4) = points(:, :, 4) ./ tangent_norm;
end

function segment_state = anchor_segment_to_increment_state(start_point, ...
    end_point, n_points_per_segment, feature_dim)
segment_state = zeros(n_points_per_segment, feature_dim);
segment_state(1, :) = start_point;
if n_points_per_segment > 1
    increment_xy = (end_point(1:2) - start_point(1:2)) ./ ...
        (n_points_per_segment - 1);
    for point_idx = 2:n_points_per_segment
        segment_state(point_idx, :) = end_point;
        segment_state(point_idx, 1:2) = increment_xy;
    end
end
end

function [A, b] = endpoint_increment_clf_map(endpoint_xy, state_mean, ...
    state_std, feature_dim, n_points_per_segment)
A = zeros(2, feature_dim * n_points_per_segment);
b = zeros(2, 1);
for xy_dim = 1:2
    dim_idx = xy_dim:feature_dim:(feature_dim * n_points_per_segment);
    A(xy_dim, dim_idx) = state_std(dim_idx);
    decoded_mean = sum(state_mean(dim_idx));
    b(xy_dim) = endpoint_xy(xy_dim) - decoded_mean;
end
end

function global_rows = local_increment_rows_to_global(local_rows, ...
    feature_dim, n_points_per_segment)
if size(local_rows, 2) ~= feature_dim * n_points_per_segment
    error('Local increment row width does not match feature/window dimensions.');
end
n_samples = size(local_rows, 1);
global_rows = zeros(size(local_rows));
for sample_idx = 1:n_samples
    local_curve = reshape(local_rows(sample_idx, :), feature_dim, [])';
    global_curve = local_curve;
    for point_idx = 2:n_points_per_segment
        global_curve(point_idx, 1:2) = ...
            global_curve(point_idx - 1, 1:2) + ...
            local_curve(point_idx, 1:2);
    end
    global_rows(sample_idx, :) = reshape(global_curve', 1, []);
end
end

function [c0_mean, c0_max, c1_mean, c1_max] = ...
    compute_segment_connection_diagnostics(segment_data, n_trajectories, ...
    n_segments, n_points_per_segment)
feature_dim = size(segment_data, 2) / n_points_per_segment;
if feature_dim < 4 || n_segments < 2
    c0_mean = 0.0;
    c0_max = 0.0;
    c1_mean = 0.0;
    c1_max = 0.0;
    return;
end
c0_errors = zeros(n_trajectories * (n_segments - 1), 1);
c1_errors = zeros(n_trajectories * (n_segments - 1), 1);
error_idx = 1;
for traj_idx = 1:n_trajectories
    for segment_idx = 1:(n_segments - 1)
        left_sample_idx = (traj_idx - 1) * n_segments + segment_idx;
        right_sample_idx = left_sample_idx + 1;
        left_curve = reshape(segment_data(left_sample_idx, :), ...
            feature_dim, [])';
        right_curve = reshape(segment_data(right_sample_idx, :), ...
            feature_dim, [])';
        left_end = left_curve(end, :);
        right_start = right_curve(1, :);
        c0_errors(error_idx) = norm(left_end(1:2) - right_start(1:2));
        left_tangent = left_end(3:4);
        right_tangent = right_start(3:4);
        left_tangent = left_tangent ./ max(norm(left_tangent), eps);
        right_tangent = right_tangent ./ max(norm(right_tangent), eps);
        tangent_dot = min(max(dot(left_tangent, right_tangent), ...
            -1.0), 1.0);
        c1_errors(error_idx) = acosd(tangent_dot);
        error_idx = error_idx + 1;
    end
end
c0_mean = mean(c0_errors);
c0_max = max(c0_errors);
c1_mean = mean(c1_errors);
c1_max = max(c1_errors);
end

function group_std = compute_uncertainty_group_std(uncertainty_values, ...
    feature_dim)
n_outputs = size(uncertainty_values, 3);
feature_idx = mod((1:n_outputs) - 1, feature_dim) + 1;
position_idx = feature_idx <= 2;
tangent_idx = feature_idx >= 3 & feature_idx <= 4;
group_std = zeros(size(uncertainty_values, 1), ...
    size(uncertainty_values, 2), 3);
group_std(:, :, 1) = sqrt(sum(uncertainty_values(:, :, position_idx), 3));
group_std(:, :, 2) = sqrt(sum(uncertainty_values(:, :, tangent_idx), 3));
group_std(:, :, 3) = sqrt(sum(uncertainty_values, 3));
end

function plot_uncertainty_group_std_vs_time(cfg, traj_times, group_std, ...
    threshold, plot_title)
if nargin < 5 || isempty(plot_title)
    plot_title = 'LoG-GP Predictive Std By Feature Group';
end
group_names = {'position std', 'tangent std', 'total std'};
fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.10, 0.12, 0.72, 0.60]);
movegui(fig, 'center');
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for group_idx = 1:numel(group_names)
    nexttile;
    hold on;
    group_values = group_std(:, :, group_idx);
    for path_idx = 1:size(group_values, 2)
        plot(traj_times, group_values(:, path_idx), ...
            'Color', [0.25, 0.52, 0.88], 'LineWidth', 1.0, ...
            'HandleVisibility', 'off');
    end
    yline(threshold, '--', 'Color', [0.85, 0.20, 0.20], ...
        'LineWidth', 1.0, 'DisplayName', 'total std threshold');
    grid on;
    xlabel('s');
    ylabel(group_names{group_idx});
    title(group_names{group_idx});
    if group_idx == 1
        legend('Location', 'best');
    end
end
sgtitle(plot_title);

output_enabled = ~isfield(cfg, 'output') || ...
    ~isfield(cfg.output, 'enabled') || cfg.output.enabled;
if output_enabled
    this_file = mfilename('fullpath');
    output_dir = fullfile(fileparts(this_file), 'outputs');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    exportgraphics(fig, fullfile(output_dir, ...
        'trajectory_gp_uncertainty_group_std_matlab.emf'), ...
        'ContentType', 'vector');
end
end

function rollout_uncertainty = evaluate_rollout_uncertainty( ...
    model_collection, rollout_times, rollout_path, fixed_mask)
if nargin < 4
    fixed_mask = [];
end
model = model_collection.model;
if isfield(model, 'output_models')
    n_outputs = numel(model.output_models);
else
    n_outputs = 1;
end
rollout_uncertainty = zeros(numel(rollout_times), size(rollout_path, 2), ...
    n_outputs);
for sample_idx = 1:size(rollout_path, 2)
    for time_idx = 1:numel(rollout_times)
        s_now = rollout_times(time_idx);
        z_now = [s_now; squeeze(rollout_path(time_idx, sample_idx, :))];
        if isfield(model, 'output_models')
            variance_set = zeros(numel(model.output_models), 1);
            for output_idx = 1:numel(model.output_models)
                variance_set(output_idx) = ...
                    model.output_models{output_idx}.predict_variance(z_now);
            end
            if ~isempty(fixed_mask)
                fixed_mask_now = reshape(fixed_mask(sample_idx, :), [], 1);
                variance_set(fixed_mask_now) = 0.0;
            end
            rollout_uncertainty(time_idx, sample_idx, :) = ...
                reshape(variance_set, 1, 1, []);
        else
            variance_now = model.local_gp.predict_variance(z_now);
            rollout_uncertainty(time_idx, sample_idx, 1) = variance_now;
        end
    end
    if mod(sample_idx, 10) == 0 || sample_idx == size(rollout_path, 2)
        fprintf('  Evaluated uncertainty for %d / %d samples...\n', ...
            sample_idx, size(rollout_path, 2));
    end
end
end
