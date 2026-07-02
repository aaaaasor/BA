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
    cfg.t_min, 1.0, cfg.n_time_slices, first_level_target_points);
first_level_feature_dim = size(target_points, 3);

%% LoG-GP Hyperparameters
disp('Optimizing or loading first-level LoG-GP hyperparameters...');
rng(cfg.first_level_hyperparameter_seed);
cfg.gp.hyperparameter_mat_path = cfg.cache.first_level_hyperparameter_path;
cfg.gp.n_pretrain = cfg.gp.first_level_n_pretrain;
cfg.gp = optimize_gp_hyperparameters(x_slices, y_slices, cfg.gp, s_slices);

%% Fit LoG-GP Flow Model
cfg.gp.training_accuracy_threshold = cfg.gp.first_level_training_accuracy_threshold;
disp('Fitting first-level LoG-GP flow model...');
rng(cfg.first_level_fit_seed);
first_model_cache_path = fullfile(this_dir, cfg.cache.first_level_model_path);
first_fit_timer = tic;
model_collection = fit_or_load_loggp_model(s_slices, x_slices, y_slices, ...
    cfg.gp, first_model_cache_path, 'first-level');
disp(['First-level LoG-GP fit/load elapsed: ', ...
    num2str(toc(first_fit_timer), '%.1f'), ' seconds']);

%% RK4 Rollout
disp('Running first-level RK4 rollout...');
n_available_traj = size(source_data, 2);
rng(cfg.first_level_rollout_seed);
traj_idx = randperm(n_available_traj, min(cfg.first_level_generation_samples, n_available_traj));
x_init = source_data(:, traj_idx)';
n_eval = size(x_init, 1);
first_rollout_cache_path = fullfile(this_dir, cfg.cache.first_level_rollout_path);
first_rollout_cache_dir = fileparts(first_rollout_cache_path);
if ~exist(first_rollout_cache_dir, 'dir'); mkdir(first_rollout_cache_dir); end
use_first_rollout_cache = false;
if isfile(first_rollout_cache_path)
    cached = load(first_rollout_cache_path);
    if size(cached.traj_path_10d, 2) == n_eval
        rollout_times = cached.rollout_times;
        traj_path_10d = cached.traj_path_10d;
        use_first_rollout_cache = true;
        disp(['Loaded cached first-level rollout: ', first_rollout_cache_path]);
    else
        disp(['Ignoring first-level rollout cache with sample count ', ...
            num2str(size(cached.traj_path_10d, 2)), ...
            '; expected ', num2str(n_eval), '.']);
    end
end
if ~use_first_rollout_cache
    first_rollout_timer = tic;
    [rollout_times, traj_path_10d] = rk4_rollout(model_collection, ...
        x_init, cfg.t_min, cfg.rollout_t_max, cfg.first_level_time_steps, ...
        []);
    disp(['First-level RK4 rollout elapsed: ', ...
        num2str(toc(first_rollout_timer), '%.1f'), ' seconds']);
    try
        save(first_rollout_cache_path, 'rollout_times', 'traj_path_10d');
    catch
        save(first_rollout_cache_path, 'rollout_times', 'traj_path_10d', '-v7.3');
    end
    disp(['Saved cached first-level rollout: ', first_rollout_cache_path]);
end

traj_path_plot = zeros(size(traj_path_10d));
for time_idx = 1:numel(rollout_times)
    states_now = squeeze(traj_path_10d(time_idx, :, :));
    if n_eval == 1
        states_now = reshape(states_now, 1, []);
    end
    states_plot = states_now .* data_transform.std' + data_transform.mean';
    traj_path_plot(time_idx, :, :) = reshape(states_plot, 1, n_eval, []);
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
segment_gp = cfg.gp;
segment_gp.training_accuracy_threshold = ...
    cfg.gp.second_level_training_accuracy_threshold;
segment_gp.hyperparameter_mat_path = cfg.cache.second_level_hyperparameter_path;
segment_gp.n_pretrain = min(cfg.gp.second_level_n_pretrain, size(segment_x_slices, 1) * size(segment_x_slices, 2));
if isfield(cfg.gp, 'second_level_length_scale_time_varying')
    segment_gp.length_scale_time_varying = ...
        cfg.gp.second_level_length_scale_time_varying;
    segment_gp.length_scale_time_scale_start = ...
        cfg.gp.second_level_length_scale_time_scale_start;
    segment_gp.length_scale_time_scale_end = ...
        cfg.gp.second_level_length_scale_time_scale_end;
end
segment_gp.max_local_gp_quantity = ceil(2.0 * ...
    size(segment_x_slices, 1) * size(segment_x_slices, 2) / ...
    segment_gp.max_local_data_quantity);
disp('Optimizing or loading second-level segment LoG-GP hyperparameters...');
rng(cfg.second_level_hyperparameter_seed);
segment_gp = optimize_gp_hyperparameters(segment_x_slices, ...
    segment_y_slices, segment_gp, segment_s_slices);
segment_gp.noise_std_vec = segment_gp.noise_std_vec / 50;
disp('Fitting second-level segment LoG-GP flow model...');
rng(cfg.second_level_fit_seed);
second_model_cache_path = fullfile(this_dir, cfg.cache.second_level_model_path);
second_fit_timer = tic;
segment_model_collection = fit_or_load_loggp_model(segment_s_slices, ...
    segment_x_slices, segment_y_slices, segment_gp, ...
    second_model_cache_path, 'second-level');
disp(['Second-level LoG-GP fit/load elapsed: ', ...
    num2str(toc(second_fit_timer), '%.1f'), ' seconds']);

second_level_anchor_points = first_level_reference_points;

n_generation_segments = size(second_level_anchor_points, 1) - 1;
n_second_level_samples = cfg.second_level_generation_samples;
n_second_level_eval = n_eval * n_second_level_samples;
n_segment_eval = n_second_level_eval * n_generation_segments;
segment_feature_dim = size(second_level_anchor_points, 3);
n_segment_rows = segment_feature_dim * cfg.segment_points_per_segment;
rng(cfg.second_level_rollout_seed);
segment_x_init = randn(n_segment_eval, n_segment_rows);
segment_variance_constraint = cfg.variance_constraint;
if isfield(cfg.variance_constraint, 'second_level_integral_uncertainty_budget')
    segment_variance_constraint.integral_uncertainty_budget = ...
        cfg.variance_constraint.second_level_integral_uncertainty_budget;
end
if isfield(cfg.variance_constraint, 'second_level_hocbf_alpha2')
    segment_variance_constraint.hocbf_alpha2 = ...
        cfg.variance_constraint.second_level_hocbf_alpha2;
end
segment_variance_constraint.grad_tol = cfg.variance_constraint.grad_tol;
if isfield(cfg.variance_constraint, 'second_level_hocbf_relaxation_bound')
    segment_variance_constraint.hocbf_relaxation_bound = ...
        cfg.variance_constraint.second_level_hocbf_relaxation_bound;
end
if isfield(cfg.variance_constraint, 'second_level_psi1_margin')
    segment_variance_constraint.psi1_margin = ...
        cfg.variance_constraint.second_level_psi1_margin;
end
if isfield(cfg.variance_constraint, 'second_level_diagnostics')
    segment_variance_constraint.diagnostics = ...
        cfg.variance_constraint.second_level_diagnostics;
end
if isfield(cfg.variance_constraint, 'second_level_terminal_variance_enabled')
    segment_variance_constraint.terminal_variance_enabled = ...
        cfg.variance_constraint.second_level_terminal_variance_enabled;
end
if isfield(cfg.variance_constraint, 'second_level_terminal_variance_beta_final')
    segment_variance_constraint.terminal_variance_beta_final = ...
        cfg.variance_constraint.second_level_terminal_variance_beta_final;
end
if isfield(cfg.variance_constraint, 'second_level_terminal_variance_ptzf_initial_margin')
    segment_variance_constraint.terminal_variance_ptzf_initial_margin = ...
        cfg.variance_constraint.second_level_terminal_variance_ptzf_initial_margin;
end
if isfield(cfg.variance_constraint, 'second_level_terminal_variance_ptzf_gamma')
    segment_variance_constraint.terminal_variance_ptzf_gamma = ...
        cfg.variance_constraint.second_level_terminal_variance_ptzf_gamma;
end
if isfield(cfg.variance_constraint, 'second_level_terminal_variance_alpha')
    segment_variance_constraint.terminal_variance_alpha = ...
        cfg.variance_constraint.second_level_terminal_variance_alpha;
end
segment_rollout_constraint = segment_variance_constraint;
segment_rollout_cache_path = fullfile(this_dir, cfg.cache.second_level_rollout_path);
segment_rollout_cache_dir = fileparts(segment_rollout_cache_path);
if ~exist(segment_rollout_cache_dir, 'dir'); mkdir(segment_rollout_cache_dir); end
use_segment_rollout_cache = false;
if isfile(segment_rollout_cache_path)
    cached = load(segment_rollout_cache_path);
    if size(cached.segment_traj_path_10d, 2) == n_segment_eval && ...
            size(cached.segment_traj_path_10d, 3) == n_segment_rows
        cache_has_terminal_diagnostics = isfield(cached, 'rollout_diagnostics') && ...
            isfield(cached.rollout_diagnostics, 'hocbf') && ...
            isfield(cached.rollout_diagnostics.hocbf, ...
            'trace_terminal_inequality_h') && ...
            isfield(cached.rollout_diagnostics.hocbf, ...
            'trace_terminal_bound') && ...
            isfield(cached.rollout_diagnostics.hocbf, ...
            'trace_integral_bound') && ...
            isfield(cached.rollout_diagnostics.hocbf, ...
            'trace_terminal_ptzf_initial_bound') && ...
            any(isfinite(cached.rollout_diagnostics.hocbf.trace_terminal_inequality_h));
        needs_terminal_diagnostics = struct_field_default( ...
            segment_rollout_constraint, 'terminal_variance_enabled', false);
        cache_matches_constraint = isfield(cached, ...
            'saved_segment_rollout_constraint') && ...
            isequaln(cached.saved_segment_rollout_constraint, ...
            segment_rollout_constraint);
        if (~needs_terminal_diagnostics || cache_has_terminal_diagnostics) && ...
                cache_matches_constraint
            segment_rollout_times = cached.segment_rollout_times;
            segment_traj_path_10d = cached.segment_traj_path_10d;
            rollout_diagnostics = cached.rollout_diagnostics;
            use_segment_rollout_cache = true;
            disp(['Loaded cached second-level rollout: ', segment_rollout_cache_path]);
        elseif ~cache_matches_constraint
            disp(['Ignoring second-level rollout cache because rollout ', ...
                'constraint config changed.']);
        else
            disp(['Ignoring second-level rollout cache without terminal ', ...
                'PTCBF diagnostics.']);
        end
    else
        disp(['Ignoring second-level rollout cache with shape mismatch. ', ...
            'Expected [*, ', num2str(n_segment_eval), ', ', ...
            num2str(n_segment_rows), '].']);
    end
end
if ~use_segment_rollout_cache
    disp('Running second-level segment RK4 rollout...');
    second_rollout_timer = tic;
    [segment_rollout_times, segment_traj_path_10d, rollout_diagnostics] = ...
        rk4_rollout(segment_model_collection, segment_x_init, cfg.t_min, ...
        cfg.rollout_t_max, cfg.second_level_time_steps, segment_rollout_constraint);
    disp(['Second-level RK4 rollout elapsed: ', ...
        num2str(toc(second_rollout_timer), '%.1f'), ' seconds']);
    saved_segment_rollout_constraint = segment_rollout_constraint;
    try
        save(segment_rollout_cache_path, 'segment_rollout_times', ...
            'segment_traj_path_10d', 'rollout_diagnostics', ...
            'saved_segment_rollout_constraint');
    catch
        save(segment_rollout_cache_path, 'segment_rollout_times', ...
            'segment_traj_path_10d', 'rollout_diagnostics', ...
            'saved_segment_rollout_constraint', '-v7.3');
    end
    disp(['Saved cached second-level rollout: ', segment_rollout_cache_path]);
end
segment_plot_cfg = cfg;
segment_plot_cfg.hocbf_focus_n_segments = n_generation_segments;
plot_hocbf_psi_trace_diagnostics(segment_plot_cfg, rollout_diagnostics);
uncertainty_rollout_diagnostics = rollout_diagnostics;

segment_traj_path_plot = zeros(size(segment_traj_path_10d));
for time_idx = 1:numel(segment_rollout_times)
    states_now = squeeze(segment_traj_path_10d(time_idx, :, :));
    if n_segment_eval == 1
        states_now = reshape(states_now, 1, []);
    end
    states_plot = states_now .* segment_data_transform.std' + ...
        segment_data_transform.mean';
    segment_traj_path_plot(time_idx, :, :) = reshape(states_plot, ...
        1, n_segment_eval, []);
end

final_segment_data = squeeze(segment_traj_path_plot(end, :, :));
if size(final_segment_data, 1) == 1
    final_segment_data = reshape(final_segment_data, 1, []);
end

target_points_plot = target_points_dense;
initial_segment_data = squeeze(segment_traj_path_plot(1, :, :));
if size(initial_segment_data, 1) == 1
    initial_segment_data = reshape(initial_segment_data, 1, []);
end
source_points_plot = stitch_segment_points(initial_segment_data, ...
    n_second_level_eval, n_generation_segments, cfg.segment_points_per_segment);
source_points_plot = normalize_tangent_features(source_points_plot);
reconstructed_points_plot = zeros(0, n_second_level_eval, ...
    size(target_points_dense, 3));
target_reference_points = target_points_dense(:, traj_idx, :);
uncertainty_times = segment_rollout_times;
disp('Evaluating second-level LoG-GP predictive uncertainty...');
uncertainty_values = evaluate_rollout_uncertainty(segment_model_collection, ...
    segment_rollout_times, segment_traj_path_10d);
print_uncertainty_sigma_diagnostic('Second-level', uncertainty_values);
disp('Evaluating second-level training-data LoG-GP predictive uncertainty...');
uncertainty_level_label = 'second-level';
uncertainty_integral_budget = segment_rollout_constraint.integral_uncertainty_budget;

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
    third_segment_gp.training_accuracy_threshold = ...
        cfg.gp.third_level_training_accuracy_threshold;
    third_segment_gp.hyperparameter_mat_path = cfg.cache.third_level_hyperparameter_path;
    third_segment_gp.n_pretrain = min(cfg.gp.third_level_n_pretrain, ...
        size(third_segment_x_slices, 1) * size(third_segment_x_slices, 2));
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
        third_segment_gp.signal_std_vec = cfg.gp.third_level_signal_std_vec;
        third_segment_gp.noise_std_vec = cfg.gp.third_level_noise_std_vec / 50;
    else
        disp('Optimizing or loading third-level segment LoG-GP hyperparameters...');
        rng(cfg.second_level_hyperparameter_seed + 100);
        third_segment_gp = optimize_gp_hyperparameters(third_segment_x_slices, ...
            third_segment_y_slices, third_segment_gp, third_segment_s_slices);
        third_segment_gp.noise_std_vec = third_segment_gp.noise_std_vec / 50;
    end

    disp('Fitting third-level segment LoG-GP flow model...');
    rng(cfg.second_level_fit_seed + 100);
    third_model_cache_path = fullfile(this_dir, cfg.cache.third_level_model_path);
    third_fit_timer = tic;
    third_segment_model_collection = fit_or_load_loggp_model( ...
        third_segment_s_slices, third_segment_x_slices, ...
        third_segment_y_slices, third_segment_gp, third_model_cache_path, ...
        'third-level');
    disp(['Third-level LoG-GP fit/load elapsed: ', ...
        num2str(toc(third_fit_timer), '%.1f'), ' seconds']);

    third_level_anchor_points = stitch_segment_points(final_segment_data, ...
        n_second_level_eval, n_generation_segments, ...
        cfg.segment_points_per_segment);
    third_level_anchor_points = normalize_tangent_features( ...
        third_level_anchor_points);
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
    third_segment_x_init = randn(n_third_segment_eval, n_third_segment_rows);

    disp('Running third-level segment RK4 rollout...');
    third_level_time_steps = cfg.first_level_time_steps;
    if isfield(cfg, 'third_level_time_steps') && ...
            ~isempty(cfg.third_level_time_steps)
        third_level_time_steps = cfg.third_level_time_steps;
    end
    third_level_rollout_t_max = cfg.rollout_t_max;
    third_segment_variance_constraint = cfg.variance_constraint;
    if isfield(cfg.variance_constraint, 'third_level_integral_uncertainty_budget')
        third_segment_variance_constraint.integral_uncertainty_budget = ...
            cfg.variance_constraint.third_level_integral_uncertainty_budget;
    end
    if isfield(cfg.variance_constraint, 'third_level_hocbf_alpha2')
        third_segment_variance_constraint.hocbf_alpha2 = ...
            cfg.variance_constraint.third_level_hocbf_alpha2;
    end
    third_segment_variance_constraint.grad_tol = cfg.variance_constraint.grad_tol;
    if isfield(cfg.variance_constraint, 'third_level_hocbf_relaxation_bound')
        third_segment_variance_constraint.hocbf_relaxation_bound = ...
            cfg.variance_constraint.third_level_hocbf_relaxation_bound;
    end
    if isfield(cfg.variance_constraint, 'third_level_psi1_margin')
        third_segment_variance_constraint.psi1_margin = ...
            cfg.variance_constraint.third_level_psi1_margin;
    end
    if isfield(cfg.variance_constraint, 'third_level_diagnostics')
        third_segment_variance_constraint.diagnostics = ...
            cfg.variance_constraint.third_level_diagnostics;
    end
    if isfield(cfg.variance_constraint, 'third_level_terminal_variance_enabled')
        third_segment_variance_constraint.terminal_variance_enabled = ...
            cfg.variance_constraint.third_level_terminal_variance_enabled;
    end
    if isfield(cfg.variance_constraint, 'third_level_terminal_variance_beta_final')
        third_segment_variance_constraint.terminal_variance_beta_final = ...
            cfg.variance_constraint.third_level_terminal_variance_beta_final;
    end
    if isfield(cfg.variance_constraint, 'third_level_terminal_variance_ptzf_initial_margin')
        third_segment_variance_constraint.terminal_variance_ptzf_initial_margin = ...
            cfg.variance_constraint.third_level_terminal_variance_ptzf_initial_margin;
    end
    if isfield(cfg.variance_constraint, 'third_level_terminal_variance_ptzf_gamma')
        third_segment_variance_constraint.terminal_variance_ptzf_gamma = ...
            cfg.variance_constraint.third_level_terminal_variance_ptzf_gamma;
    end
    if isfield(cfg.variance_constraint, 'third_level_terminal_variance_alpha')
        third_segment_variance_constraint.terminal_variance_alpha = ...
            cfg.variance_constraint.third_level_terminal_variance_alpha;
    end
    third_rollout_cache_path = fullfile(this_dir, cfg.cache.third_level_rollout_path);
    third_rollout_cache_dir = fileparts(third_rollout_cache_path);
    if ~exist(third_rollout_cache_dir, 'dir'); mkdir(third_rollout_cache_dir); end
    use_third_rollout_cache = false;
    if isfile(third_rollout_cache_path)
        cached = load(third_rollout_cache_path);
        if size(cached.third_traj_path_10d, 2) == n_third_segment_eval
            needs_third_terminal_diagnostics = struct_field_default( ...
                third_segment_variance_constraint, ...
                'terminal_variance_enabled', false);
            cache_has_third_diagnostics = isfield(cached, ...
                'third_rollout_diagnostics') && ...
                isfield(cached.third_rollout_diagnostics, 'hocbf') && ...
                isfield(cached.third_rollout_diagnostics.hocbf, ...
                'trace_terminal_inequality_h') && ...
                isfield(cached.third_rollout_diagnostics.hocbf, ...
                'trace_terminal_bound') && ...
                isfield(cached.third_rollout_diagnostics.hocbf, ...
                'trace_integral_bound') && ...
                isfield(cached.third_rollout_diagnostics.hocbf, ...
                'trace_terminal_ptzf_initial_bound');
            cache_matches_third_constraint = isfield(cached, ...
                'saved_third_segment_variance_constraint') && ...
                isequaln(cached.saved_third_segment_variance_constraint, ...
                third_segment_variance_constraint);
            if (~needs_third_terminal_diagnostics || ...
                    cache_has_third_diagnostics) && ...
                    cache_matches_third_constraint
                third_rollout_times = cached.third_rollout_times;
                third_traj_path_10d = cached.third_traj_path_10d;
                if cache_has_third_diagnostics
                    third_rollout_diagnostics = cached.third_rollout_diagnostics;
                end
                use_third_rollout_cache = true;
                disp(['Loaded cached third-level rollout: ', third_rollout_cache_path]);
            elseif ~cache_matches_third_constraint
                disp(['Ignoring third-level rollout cache because rollout ', ...
                    'constraint config changed.']);
            else
                disp(['Ignoring third-level rollout cache without terminal ', ...
                    'PTCBF diagnostics.']);
            end
        else
            disp(['Ignoring third-level rollout cache with sample count ', ...
                num2str(size(cached.third_traj_path_10d, 2)), ...
                '; expected ', num2str(n_third_segment_eval), '.']);
        end
    end
    if ~use_third_rollout_cache
        third_rollout_timer = tic;
        [third_rollout_times, third_traj_path_10d, third_rollout_diagnostics] = ...
            rk4_rollout(third_segment_model_collection, third_segment_x_init, ...
            cfg.t_min, third_level_rollout_t_max, third_level_time_steps, ...
            third_segment_variance_constraint);
        disp(['Third-level RK4 rollout elapsed: ', ...
            num2str(toc(third_rollout_timer), '%.1f'), ' seconds']);
        saved_third_segment_variance_constraint = ...
            third_segment_variance_constraint;
        try
            save(third_rollout_cache_path, 'third_rollout_times', ...
                'third_traj_path_10d', 'third_rollout_diagnostics', ...
                'saved_third_segment_variance_constraint');
        catch
            save(third_rollout_cache_path, 'third_rollout_times', ...
                'third_traj_path_10d', 'third_rollout_diagnostics', ...
                'saved_third_segment_variance_constraint', '-v7.3');
        end
        disp(['Saved cached third-level rollout: ', third_rollout_cache_path]);
        third_segment_plot_cfg = cfg;
        third_segment_plot_cfg.hocbf_focus_n_segments = n_third_generation_segments;
        plot_hocbf_psi_trace_diagnostics(third_segment_plot_cfg, third_rollout_diagnostics);
    end

    third_traj_path_plot = zeros(size(third_traj_path_10d));
    for time_idx = 1:numel(third_rollout_times)
        states_now = squeeze(third_traj_path_10d(time_idx, :, :));
        if size(states_now, 1) == 1
            states_now = reshape(states_now, 1, []);
        end
        states_local = states_now .* third_segment_data_transform.std' + ...
            third_segment_data_transform.mean';
        states_plot = local_increment_rows_to_global(states_local, ...
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

    disp('Evaluating third-level LoG-GP predictive uncertainty...');
    uncertainty_values = evaluate_rollout_uncertainty( ...
        third_segment_model_collection, third_rollout_times, ...
        third_traj_path_10d);
    print_uncertainty_sigma_diagnostic('Third-level', uncertainty_values);
    uncertainty_times = third_rollout_times;
    uncertainty_level_label = 'third-level';
    uncertainty_integral_budget = third_segment_variance_constraint.integral_uncertainty_budget;
    if exist('third_rollout_diagnostics', 'var')
        uncertainty_rollout_diagnostics = third_rollout_diagnostics;
    else
        uncertainty_rollout_diagnostics = [];
    end

    target_points_plot = target_points_fine;
    source_points_plot = third_source_points_raw;
    reconstructed_points_plot = reconstructed_points_fine;
    target_reference_points = third_target_reference_points;
    reconstructed_points_refined = reconstructed_points_fine;
    final_trajectory_data = reshape(permute(reconstructed_points_fine, ...
        [3, 1, 2]), [], n_third_level_eval)';
    n_generation_segments = n_third_generation_segments;
    n_second_level_eval = n_third_level_eval;
    segment_rollout_times = third_rollout_times;
    segment_traj_path_plot = third_traj_path_plot;
    segment_traj_path_10d = third_traj_path_10d;
end

%% Plot Results
disp('Plotting and exporting results...');
plot_cfg = cfg;
plot_cfg.first_level_generation_samples = size(reconstructed_points_plot, 2);
if exist('target_reference_points', 'var')
    plot_cfg.reference_points = target_reference_points;
end
plot_cfg.reference_label = 'Training reference curve';
plot_cfg.reference_curve_count = n_eval;
final_segment_endpoints = squeeze(segment_traj_path_plot(end, :, :));
if n_segment_eval == 1
    final_segment_endpoints = reshape(final_segment_endpoints, 1, []);
end
plot_cfg.segment_plot_data = final_segment_endpoints;
plot_cfg.segment_plot_count = n_generation_segments;
plot_cfg.segment_plot_points_per_segment = cfg.segment_points_per_segment;
if isfield(cfg, 'enable_third_level') && cfg.enable_third_level
    plot_cfg.sample_curve_label = 'Stage 3 sample curves';
    plot_cfg.generated_point_label = 'Stage 3 generated points';
    if exist('final_third_segment_data', 'var')
        plot_cfg.segment_plot_data = final_third_segment_data;
        plot_cfg.segment_plot_count = n_third_generation_segments;
    end
end
plot_results(plot_cfg, target_points_plot, source_points_plot, ...
    reconstructed_points_plot);
if ~isempty(uncertainty_values)
    plot_sigma_vs_time(cfg, uncertainty_times, uncertainty_values, ...
        uncertainty_level_label, uncertainty_rollout_diagnostics);
else
    disp('Skipping uncertainty plots because uncertainty evaluation is disabled.');
end
animation_path = "";
if cfg.animation.enabled && n_eval > 0
    animation_segment_count = n_generation_segments;
    animation_point_count = animation_segment_count * ...
        cfg.segment_points_per_segment;
    animation_eval_count = n_second_level_eval;
    animation_traj_path = segment_traj_path_plot;
    animation_times = segment_rollout_times;
    animation_use_third_global_rollout = false;
    if isfield(cfg, 'enable_third_level') && cfg.enable_third_level && ...
            exist('reconstructed_points_fine', 'var')
        animation_segment_count = n_third_generation_segments;
        animation_point_count = animation_segment_count * ...
            cfg.segment_points_per_segment;
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
        if size(segment_states_now, 1) == 1
            segment_states_now = reshape(segment_states_now, 1, []);
        end
        segment_points_now = reshape(segment_states_now', ...
            segment_feature_dim, [])';
        traj_path_single(time_idx, :) = reshape(segment_points_now', 1, []);
    end
    source_segment_rows = squeeze(animation_traj_path( ...
        1, segment_animation_idx, :));
    if size(source_segment_rows, 1) == 1
        source_segment_rows = reshape(source_segment_rows, 1, []);
    end
    animation_source_points = stitch_segment_points(source_segment_rows, ...
        1, animation_segment_count, cfg.segment_points_per_segment);
    animation_source_points = squeeze(animation_source_points(:, 1, :));
    animation_cfg = cfg;
    animation_cfg.animation.segment_plot_path = ...
        animation_traj_path(:, segment_animation_idx, :);
    animation_cfg.animation.segment_plot_count = animation_segment_count;
    animation_cfg.animation.segment_plot_points_per_segment = ...
        cfg.segment_points_per_segment;
    if animation_use_third_global_rollout
        animation_cfg.animation.space_label = 'Original Global Space';
    end
    animation_path = animate_single_trajectory(animation_cfg, animation_times, ...
        traj_path_single, animation_source_points);
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
disp(['Second-level LoG-GP noise std SigmaN per output: ', ...
    mat2str(segment_gp.noise_std_vec(:)', 4)]);
if isfield(segment_gp, 'length_scale_time_varying') && ...
        segment_gp.length_scale_time_varying
    disp(['Second-level SigmaL(t) scale: ', ...
        num2str(segment_gp.length_scale_time_scale_start), ...
        ' at t=0 -> ', ...
        num2str(segment_gp.length_scale_time_scale_end), ...
        ' at t=1']);
end
disp(['Second-level sliding-window samples: ', ...
    num2str(size(segment_x_slices, 2))]);
disp(['Second-level LoG-GP training pairs: ', ...
    num2str(segment_model_collection.n_training_pairs)]);
disp(['Second-level LoG-GP added pairs per output after accuracy check: ', ...
    mat2str(segment_model_collection.n_added_per_output(:)')]);
disp(['Second-level LoG-GP skipped pairs per output after accuracy check: ', ...
    mat2str(segment_model_collection.n_skipped_per_output(:)')]);
disp(['Second-level generated sample curves: ',...
    num2str(n_second_level_eval)]);
if exist('third_segment_model_collection', 'var')
    disp(['Third-level LoG-GP noise std SigmaN per output: ', ...
        mat2str(third_segment_gp.noise_std_vec(:)', 4)]);
    disp(['Third-level block-window samples: ', ...
        num2str(size(third_segment_x_slices, 2))]);
    disp(['Third-level LoG-GP training pairs: ', ...
        num2str(third_segment_model_collection.n_training_pairs)]);
    disp(['Third-level LoG-GP added pairs per output after accuracy check: ', ...
        mat2str(third_segment_model_collection.n_added_per_output(:)')]);
    disp(['Third-level LoG-GP skipped pairs per output after accuracy check: ', ...
        mat2str(third_segment_model_collection.n_skipped_per_output(:)')]);
end
disp(['Total training trajectories: ', num2str(size(x_slices, 2))]);
disp(['Rolled-out trajectories: ', num2str(n_eval)]);
disp(['First-level anchor position RMSE mean: ',...
    num2str(mean(first_level_position_rmse))]);
disp(['First-level anchor position RMSE max: ', ...
    num2str(max(first_level_position_rmse))]);
if all(isfinite(first_level_tangent_angle_error))
    disp(['First-level anchor tangent angle error mean deg: ', ...
        num2str(mean(first_level_tangent_angle_error))]);
    disp(['First-level anchor tangent angle error max deg: ', ...
        num2str(max(first_level_tangent_angle_error))]);
end
if strlength(animation_path) > 0
    disp(['Single-trajectory animation: ', char(animation_path)]);
end
