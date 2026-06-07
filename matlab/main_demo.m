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
model_collection = fit_or_load_loggp_model(s_slices, x_slices, y_slices, ...
    cfg.gp, first_model_cache_path, 'first-level');

%% RK4 Rollout
disp('Running first-level RK4 rollout...');
n_available_traj = size(source_data, 2);
rng(cfg.rollout_seed);
traj_idx = randperm(n_available_traj, min(cfg.n_trajectories, n_available_traj));
x_init = source_data(:, traj_idx)';
n_eval = size(x_init, 1);
first_rollout_cache_hit = false;
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
            rollout_times = cached_first_rollout.rollout_times;
            traj_path_10d = cached_first_rollout.traj_path_10d;
            first_rollout_cache_hit = true;
            disp(['Loaded cached first-level rollout: ', ...
                first_rollout_cache_path]);
        else
            disp('Cached first-level rollout is missing rollout data.');
        end
    end
end
if ~first_rollout_cache_hit
    [rollout_times, traj_path_10d] = rk4_rollout(model_collection, ...
        x_init, cfg.t_min, cfg.rollout_t_max, cfg.time_steps, ...
        cfg.variance_constraint);
    if cfg.cache.rollout_enabled
        try
            save(first_rollout_cache_path, 'rollout_times', 'traj_path_10d');
        catch
            save(first_rollout_cache_path, 'rollout_times', ...
                'traj_path_10d', '-v7.3');
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
    cfg.segment_points_per_segment);
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
    segment_gp.save_hyperparameters = false;
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
disp('Optimizing or loading second-level segment LoG-GP hyperparameters...');
rng(cfg.second_level_hyperparameter_seed);
segment_gp = optimize_gp_hyperparameters(segment_x_slices, segment_y_slices, ...
    segment_gp, segment_s_slices);
disp('Fitting second-level segment LoG-GP flow model...');
rng(cfg.second_level_fit_seed);
if cfg.cache.model_enabled
    second_model_cache_path = fullfile(this_dir, ...
        cfg.cache.second_level_model_path);
else
    second_model_cache_path = "";
end
segment_model_collection = fit_or_load_loggp_model(segment_s_slices, ...
    segment_x_slices, segment_y_slices, segment_gp, ...
    second_model_cache_path, 'second-level');

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
            segment_x_init(sample_idx, segment_start_idx) = start_state;
            segment_x_init(sample_idx, segment_end_idx) = end_state;
            fixed_idx = [segment_start_idx, segment_end_idx];
            segment_fixed_mask(sample_idx, fixed_idx) = true;
            segment_fixed_values(sample_idx, fixed_idx) = ...
                [start_state, end_state];
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
if isfield(cfg.variance_constraint, 'second_level_max_phi')
    segment_variance_constraint.max_phi = ...
        cfg.variance_constraint.second_level_max_phi;
end
segment_rollout_cache_hit = false;
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
            segment_rollout_times = cached_rollout.segment_rollout_times;
            segment_traj_path_10d = cached_rollout.segment_traj_path_10d;
            segment_rollout_cache_hit = true;
            disp(['Loaded cached second-level rollout: ', ...
                segment_rollout_cache_path]);
        else
            disp('Cached second-level rollout is missing rollout data.');
        end
    end
end
if ~segment_rollout_cache_hit
    [segment_rollout_times, segment_traj_path_10d] = rk4_rollout( ...
        segment_model_collection, segment_x_init, cfg.t_min, ...
        cfg.rollout_t_max, cfg.time_steps, segment_variance_constraint, ...
        segment_fixed_mask, segment_fixed_values);
    if cfg.cache.rollout_enabled
        try
            save(segment_rollout_cache_path, 'segment_rollout_times', ...
                'segment_traj_path_10d');
        catch
            save(segment_rollout_cache_path, 'segment_rollout_times', ...
                'segment_traj_path_10d', '-v7.3');
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
final_segment_state = squeeze(segment_traj_path_10d(end, :, :));
if size(final_segment_state, 1) == 1
    final_segment_state = reshape(final_segment_state, 1, []);
end
second_level_fixed_error = max(abs( ...
    final_segment_state(segment_fixed_mask) - ...
    segment_fixed_values(segment_fixed_mask)), [], 'all');
disp(['Second-level reconstructed size: ', ...
    mat2str(size(reconstructed_points_refined))]);
disp(['Second-level anchor max error: ', ...
    num2str(second_level_anchor_error)]);
disp(['Second-level fixed-state max error: ', ...
    num2str(second_level_fixed_error)]);
disp(['Second-level target RMSE mean: ', num2str(target_rmse_mean)]);
disp(['Second-level target RMSE max: ', num2str(target_rmse_max)]);

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
uncertainty_level_label = 'second-level';

%% Plot Results
disp('Plotting and exporting results...');
plot_cfg = cfg;
plot_cfg.n_trajectories = size(reconstructed_points_plot, 2);
plot_cfg.reference_points = target_reference_points;
plot_cfg.reference_label = 'Training reference curve';
plot_cfg.reference_curve_count = n_eval;
plot_results(plot_cfg, target_points_plot, source_points_plot, ...
    reconstructed_points_plot);
plot_uncertainty_vs_time(cfg, uncertainty_times, uncertainty_values, ...
    uncertainty_plot_title, uncertainty_threshold);
plot_uncertainty_group_std_vs_time(cfg, uncertainty_times, ...
    uncertainty_group_std, uncertainty_threshold);
animation_path = "";
if cfg.animation.enabled && n_eval > 0
    animation_nr = min(cfg.animation.trajectory_nr, n_second_level_eval);
    segment_animation_idx = ((animation_nr - 1) * n_generation_segments + 1): ...
        (animation_nr * n_generation_segments);
    traj_path_single = zeros(numel(segment_rollout_times), ...
        segment_feature_dim * size(reconstructed_points_refined, 1));
    for time_idx = 1:numel(segment_rollout_times)
        segment_states_now = squeeze(segment_traj_path_plot( ...
            time_idx, segment_animation_idx, :));
        stitched_now = stitch_segment_points(segment_states_now, ...
            1, n_generation_segments, cfg.segment_points_per_segment);
        stitched_now = squeeze(stitched_now);
        traj_path_single(time_idx, :) = reshape(stitched_now', 1, []);
    end
    animation_times = segment_rollout_times;
    animation_source_points = reshape(traj_path_single(1, :), ...
        segment_feature_dim, [])';
    animation_path = animate_single_trajectory(cfg, animation_times, ...
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
    threshold)
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
sgtitle('Second-Level LoG-GP Predictive Std By Feature Group');

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
