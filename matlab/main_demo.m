% Main entry point for the MATLAB 10D trajectory-space LoG-GP ODE demo.
clear;
clc;

%% Configuration
cfg = get_config();
rng(cfg.random_seed);

%% Training Data
disp('Building first-level 10D training data...');
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
trajectory_mat_path = fullfile(this_dir, 'trajectory_data', 'data_2D.mat');
first_level_target_points = [];
if isfield(cfg, 'first_level_target_source') && ...
        strcmpi(cfg.first_level_target_source, 'generated')
    first_level_target_points = generate_original_training_points_2d( ...
        5, cfg.n_train);
    disp('First-level target source: generated 5-point trajectories');
else
    disp('First-level target source: dataset data_2D.mat');
end
[s_slices, x_slices, y_slices, target_points, source_points, ...
    target_data, source_data, trajectory_t_slices, data_transform] = build_training_data( ...
    trajectory_mat_path, cfg.t_min, 1.0, cfg.n_train, ...
    cfg.n_time_slices, first_level_target_points);

%% LoG-GP Hyperparameters
disp('Optimizing or loading first-level LoG-GP hyperparameters...');
cfg.gp = optimize_gp_hyperparameters(x_slices, y_slices, cfg.gp, s_slices);

%% Fit LoG-GP Flow Model
disp('Fitting first-level 10D LoG-GP flow model...');
rng(cfg.first_level_fit_seed);
if cfg.cache.enabled
    first_model_cache_path = fullfile(this_dir, ...
        cfg.cache.first_level_model_path);
else
    first_model_cache_path = "";
end
model_collection = fit_or_load_loggp_model(s_slices, x_slices, y_slices, ...
    cfg.gp, first_model_cache_path, 'first-level');

%% RK4 Rollout
disp('Running first-level 10D RK4 rollout...');
n_available_traj = size(source_data, 2);
rng(cfg.rollout_seed);
traj_idx = randperm(n_available_traj, min(cfg.n_trajectories, n_available_traj));
x_init = source_data(:, traj_idx)';
n_eval = size(x_init, 1);
first_rollout_cache_key = build_first_level_rollout_cache_key( ...
    cfg, traj_idx, x_init, cfg.variance_constraint, model_collection);
first_rollout_cache_hit = false;
if cfg.cache.enabled
    first_rollout_cache_path = fullfile(this_dir, ...
        cfg.cache.first_level_rollout_path);
    first_rollout_cache_dir = fileparts(first_rollout_cache_path);
    if ~exist(first_rollout_cache_dir, 'dir')
        mkdir(first_rollout_cache_dir);
    end
    if isfile(first_rollout_cache_path)
        cached_first_rollout = load(first_rollout_cache_path);
        if isfield(cached_first_rollout, 'first_rollout_cache_key') && ...
                strcmp(cached_first_rollout.first_rollout_cache_key, ...
                first_rollout_cache_key) && ...
                isfield(cached_first_rollout, 'rollout_times') && ...
                isfield(cached_first_rollout, 'traj_path_10d')
            rollout_times = cached_first_rollout.rollout_times;
            traj_path_10d = cached_first_rollout.traj_path_10d;
            first_rollout_cache_hit = true;
            disp(['Loaded cached first-level rollout: ', ...
                first_rollout_cache_path]);
        else
            disp('Cached first-level rollout does not match current settings.');
        end
    end
end
if ~first_rollout_cache_hit
    [rollout_times, traj_path_10d] = rk4_rollout(model_collection, ...
        x_init, cfg.t_min, cfg.rollout_t_max, cfg.time_steps, ...
        cfg.variance_constraint);
    if cfg.cache.enabled
        try
            save(first_rollout_cache_path, 'first_rollout_cache_key', ...
                'rollout_times', 'traj_path_10d');
        catch
            save(first_rollout_cache_path, 'first_rollout_cache_key', ...
                'rollout_times', 'traj_path_10d', '-v7.3');
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
    states_plot = data_transform.std * states_now' + data_transform.mean;
    traj_path_plot(time_idx, :, :) = reshape(states_plot', 1, n_eval, []);
end

final_trajectory_data = squeeze(traj_path_plot(end, :, :));
if n_eval == 1
    final_trajectory_data = reshape(final_trajectory_data, 1, []);
end

reconstructed_points = zeros(numel(trajectory_t_slices), n_eval, 2);
for sample_idx = 1:n_eval
    reconstructed_curve = reshape(final_trajectory_data(sample_idx, :), 2, [])';
    reconstructed_points(:, sample_idx, :) = reconstructed_curve;
end

target_points_plot = target_points;
source_points_plot = source_points(:, traj_idx, :);
reconstructed_points_plot = reconstructed_points;
reconstruction_source_idx = traj_idx(:) - 1;
uncertainty_times = rollout_times;
uncertainty_values = evaluate_rollout_uncertainty(model_collection, ...
    rollout_times, traj_path_10d);
uncertainty_threshold = cfg.variance_constraint.sigma2_max;
uncertainty_plot_title = ...
    'First-Level LoG-GP Predictive Uncertainty Along 10D ODE Rollout';
uncertainty_level_label = 'first-level';

%% Segment-Level Flow Model
if cfg.run_second_level
disp('Building second-level segment flow-matching data...');
rng(cfg.random_seed + 1);
first_level_anchor_idx = 1:(cfg.segment_points_per_segment - 1): ...
    ((size(target_points, 1) - 1) * ...
    (cfg.segment_points_per_segment - 1) + 1);
n_second_level_training_points = (size(target_points, 1) - 1) * ...
    (cfg.segment_points_per_segment - 1) + 1;
target_points_original = generate_original_training_points_2d( ...
    n_second_level_training_points, size(target_points, 2));
original_anchor_error = max(abs( ...
    target_points_original(first_level_anchor_idx, :, :) - target_points), ...
    [], 'all');
disp(['Original training curve anchor reconstruction error: ', ...
    num2str(original_anchor_error)]);
target_points_refined = target_points_original;
[segment_s_slices, segment_x_slices, segment_y_slices, ...
    ~, ~, segment_data_transform, ~] = ...
    build_sliding_window_training_data( ...
    target_points_refined, cfg.t_min, 1.0, cfg.n_time_slices, ...
    cfg.segment_points_per_segment, first_level_anchor_idx);

segment_gp = cfg.gp;
segment_gp.training_accuracy_threshold = ...
    cfg.gp.second_level_training_accuracy_threshold;
if cfg.debug.quick_second_level_training
    segment_gp.max_training_pairs = cfg.debug.second_level_max_training_pairs;
    segment_gp.training_subset_seed = ...
        cfg.debug.second_level_training_subset_seed;
end
segment_gp.hyperparameter_mat_path = fullfile('outputs', ...
    'LoG_GP_SlidingWindow_Training_Hyperparameter.mat');
segment_gp.n_pretrain = min(60000, ...
    size(segment_x_slices, 1) * size(segment_x_slices, 2));
segment_gp.max_local_gp_quantity = ceil(2.0 * ...
    size(segment_x_slices, 1) * size(segment_x_slices, 2) / ...
    segment_gp.max_local_data_quantity);
disp('Optimizing or loading second-level segment LoG-GP hyperparameters...');
segment_gp = optimize_gp_hyperparameters(segment_x_slices, segment_y_slices, ...
    segment_gp, segment_s_slices);
disp('Fitting second-level segment LoG-GP flow model...');
rng(cfg.second_level_fit_seed);
if cfg.cache.enabled
    second_model_cache_path = fullfile(this_dir, ...
        cfg.cache.second_level_model_path);
else
    second_model_cache_path = "";
end
segment_model_collection = fit_or_load_loggp_model(segment_s_slices, ...
    segment_x_slices, segment_y_slices, segment_gp, ...
    second_model_cache_path, 'second-level');

second_level_anchor_points = reconstructed_points;
second_level_anchor_source = 'first_level_rollout';
if isfield(cfg, 'second_level_anchor_source')
    second_level_anchor_source = cfg.second_level_anchor_source;
end
if strcmpi(second_level_anchor_source, 'training_data')
    second_level_anchor_points = target_points(:, traj_idx, :);
elseif ~strcmpi(second_level_anchor_source, 'first_level_rollout')
    error('Unknown second_level_anchor_source: %s', ...
        second_level_anchor_source);
end
disp(['Second-level anchor source: ', second_level_anchor_source]);

n_generation_segments = size(second_level_anchor_points, 1) - 1;
n_second_level_samples = cfg.second_level_generation_samples;
n_second_level_eval = n_eval * n_second_level_samples;
n_segment_eval = n_second_level_eval * n_generation_segments;
n_segment_rows = 2 * cfg.segment_points_per_segment;
rng(cfg.second_level_rollout_seed);
segment_x_init = randn(n_segment_eval, n_segment_rows);
segment_fixed_mask = false(n_segment_eval, n_segment_rows);
segment_fixed_values = segment_x_init;
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
            start_state = (start_point - segment_data_transform.mean) ./ ...
                segment_data_transform.std;
            end_state = (end_point - segment_data_transform.mean) ./ ...
                segment_data_transform.std;
            segment_x_init(sample_idx, 1:2) = start_state;
            segment_x_init(sample_idx, (end - 1):end) = end_state;
            segment_fixed_mask(sample_idx, [1, 2, end - 1, end]) = true;
            segment_fixed_values(sample_idx, [1, 2, end - 1, end]) = ...
                [start_state, end_state];
        end
    end
end
disp('Running second-level segment RK4 rollout...');
segment_variance_constraint = cfg.variance_constraint;
if isfield(cfg.variance_constraint, 'second_level_enabled')
    segment_variance_constraint.enabled = ...
        cfg.variance_constraint.second_level_enabled;
end
segment_variance_constraint.sigma2_max = ...
    cfg.gp.second_level_generation_accuracy_threshold;
segment_variance_constraint.generation_accuracy_threshold = ...
    cfg.gp.second_level_generation_accuracy_threshold;
if isfield(cfg.variance_constraint, 'second_level_alpha_gain')
    segment_variance_constraint.alpha_gain = ...
        cfg.variance_constraint.second_level_alpha_gain;
end
if isfield(cfg.variance_constraint, 'second_level_omega_gain')
    segment_variance_constraint.omega_gain = ...
        cfg.variance_constraint.second_level_omega_gain;
end
segment_rollout_cache_key = build_second_level_rollout_cache_key( ...
    cfg, second_level_anchor_points, segment_x_init, segment_fixed_mask, ...
    segment_fixed_values, segment_data_transform, ...
    segment_variance_constraint, segment_model_collection);
segment_rollout_cache_hit = false;
if cfg.cache.enabled
    segment_rollout_cache_path = fullfile(this_dir, ...
        cfg.cache.second_level_rollout_path);
    segment_rollout_cache_dir = fileparts(segment_rollout_cache_path);
    if ~exist(segment_rollout_cache_dir, 'dir')
        mkdir(segment_rollout_cache_dir);
    end
    if isfile(segment_rollout_cache_path)
        cached_rollout = load(segment_rollout_cache_path);
        if isfield(cached_rollout, 'segment_rollout_cache_key') && ...
                strcmp(cached_rollout.segment_rollout_cache_key, ...
                segment_rollout_cache_key) && ...
                isfield(cached_rollout, 'segment_rollout_times') && ...
                isfield(cached_rollout, 'segment_traj_path_10d')
            segment_rollout_times = cached_rollout.segment_rollout_times;
            segment_traj_path_10d = cached_rollout.segment_traj_path_10d;
            segment_rollout_cache_hit = true;
            disp(['Loaded cached second-level rollout: ', ...
                segment_rollout_cache_path]);
        else
            disp('Cached second-level rollout does not match current settings.');
        end
    end
end
if ~segment_rollout_cache_hit
    [segment_rollout_times, segment_traj_path_10d] = rk4_rollout( ...
        segment_model_collection, segment_x_init, cfg.t_min, ...
        cfg.rollout_t_max, cfg.time_steps, segment_variance_constraint, ...
        segment_fixed_mask, segment_fixed_values);
    if cfg.cache.enabled
        try
            save(segment_rollout_cache_path, 'segment_rollout_cache_key', ...
                'segment_rollout_times', 'segment_traj_path_10d');
        catch
            save(segment_rollout_cache_path, 'segment_rollout_cache_key', ...
                'segment_rollout_times', 'segment_traj_path_10d', ...
                '-v7.3');
        end
        disp(['Saved cached second-level rollout: ', ...
            segment_rollout_cache_path]);
    end
end

segment_traj_path_plot = zeros(size(segment_traj_path_10d));
for time_idx = 1:numel(segment_rollout_times)
    states_now = squeeze(segment_traj_path_10d(time_idx, :, :));
    states_plot = segment_data_transform.std * states_now' + ...
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
second_level_anchor_idx = 1:(cfg.segment_points_per_segment - 1): ...
    size(reconstructed_points_refined, 1);
anchor_reference_points = zeros(size(second_level_anchor_points, 1), ...
    n_second_level_eval, 2);
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
target_reference_points = zeros(size(target_points_original, 1), ...
    n_second_level_eval, 2);
for traj_eval_idx = 1:n_eval
    for second_sample_idx = 1:n_second_level_samples
        second_eval_idx = (traj_eval_idx - 1) * n_second_level_samples + ...
            second_sample_idx;
        target_reference_points(:, second_eval_idx, :) = ...
            target_points_original(:, traj_idx(traj_eval_idx), :);
    end
end
target_error = reconstructed_points_refined - target_reference_points;
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

target_points_plot = target_points_original;
source_points_plot = source_points_refined;
reconstructed_points_plot = reconstructed_points_refined;
uncertainty_times = segment_rollout_times;
disp('Evaluating second-level LoG-GP predictive uncertainty...');
uncertainty_values = evaluate_rollout_uncertainty(segment_model_collection, ...
    segment_rollout_times, segment_traj_path_10d);
uncertainty_threshold = segment_variance_constraint.sigma2_max;
uncertainty_plot_title = ...
    'Second-Level LoG-GP Predictive Uncertainty Along Segment Rollout';
uncertainty_level_label = 'second-level';
end

%% Plot Results
disp('Plotting and exporting results...');
plot_cfg = cfg;
if cfg.run_second_level
    plot_cfg.n_trajectories = size(reconstructed_points_plot, 2);
    if exist('target_reference_points', 'var')
        plot_cfg.reference_points = target_reference_points;
        plot_cfg.reference_label = 'Training reference curve';
        plot_cfg.reference_curve_count = n_eval;
    end
end
plot_results(plot_cfg, target_points_plot, source_points_plot, ...
    reconstructed_points_plot);
plot_uncertainty_vs_time(cfg, uncertainty_times, uncertainty_values, ...
    uncertainty_plot_title, uncertainty_threshold);
animation_path = "";
if cfg.animation.enabled && n_eval > 0
    if cfg.run_second_level
        animation_nr = min(cfg.animation.trajectory_nr, n_second_level_eval);
        segment_animation_idx = ((animation_nr - 1) * n_generation_segments + 1): ...
            (animation_nr * n_generation_segments);
        traj_path_single = zeros(numel(segment_rollout_times), ...
            2 * size(reconstructed_points_refined, 1));
        for time_idx = 1:numel(segment_rollout_times)
            segment_states_now = squeeze(segment_traj_path_10d( ...
                time_idx, segment_animation_idx, :));
            stitched_now = stitch_segment_points(segment_states_now, ...
                1, n_generation_segments, cfg.segment_points_per_segment);
            stitched_now = squeeze(stitched_now);
            traj_path_single(time_idx, :) = reshape(stitched_now', 1, []);
        end
        animation_times = segment_rollout_times;
        animation_source_points = squeeze(segment_source_points_standardized(:, ...
            animation_nr, :));
    else
        animation_nr = min(cfg.animation.trajectory_nr, n_eval);
        traj_path_single = squeeze(traj_path_10d(:, animation_nr, :));
        animation_times = rollout_times;
        animation_source_points = reshape(x_init(animation_nr, :), 2, [])';
    end
    animation_path = animate_single_trajectory(cfg, animation_times, ...
        traj_path_single, animation_source_points);
end

%% Export Results
output_dir = fullfile(this_dir, 'outputs');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
reconstruction_headers = ["trajectory_index", ...
    arrayfun(@(idx) "coord_" + string(idx), ...
    1:size(final_trajectory_data, 2))];
reconstruction_table = [reconstruction_source_idx(:), final_trajectory_data];
writematrix(reconstruction_headers, fullfile(output_dir, 'trajectory_reconstruction_samples.csv'));
writematrix(reconstruction_table, ...
    fullfile(output_dir, 'trajectory_reconstruction_samples.csv'), ...
    'WriteMode', 'append');

traj_gp_var_table = [uncertainty_times, uncertainty_values(:, :, 1)];
traj_gp_var_headers = ["s", ...
    arrayfun(@(idx) "path_" + string(idx) + "_uncertainty", ...
    0:(size(uncertainty_values, 2) - 1))];
writematrix(traj_gp_var_headers, fullfile(output_dir, ...
    'trajectory_gp_predictive_uncertainties.csv'));
writematrix(traj_gp_var_table, ...
    fullfile(output_dir, 'trajectory_gp_predictive_uncertainties.csv'), ...
    'WriteMode', 'append');

%% Console Summary
disp(['Training trajectories loaded: ', num2str(size(target_data, 2))]);
disp(['Trajectory points per sample: ', num2str(numel(trajectory_t_slices))]);
disp(['10D ODE s slices: ', num2str(numel(s_slices))]);
disp(['Rollout time interval: [', num2str(cfg.t_min), ', ', ...
    num2str(cfg.rollout_t_max), ']']);
disp(['LoG-GP input dimension: ', num2str(size(x_slices, 3) + 1)]);
disp(['LoG-GP output dimension: ', num2str(size(y_slices, 3))]);
disp(['GP model type: ', model_collection.model_type]);
disp(['LoG-GP noise std SigmaN: ', num2str(cfg.gp.noise_std)]);
disp(['LoG-GP signal std SigmaF: ', num2str(cfg.gp.signal_std)]);
disp(['LoG-GP length scale SigmaL: ', mat2str(cfg.gp.length_scale_vec(:)', 4)]);
if cfg.run_second_level
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
    disp(['Second-level PT-CBF enabled: ', ...
        logical_to_text(segment_variance_constraint.enabled)]);
    disp(['Second-level generation uncertainty threshold: ', ...
        num2str(segment_variance_constraint.sigma2_max)]);
    disp(['Second-level generated sample curves: ', ...
        num2str(n_second_level_eval)]);
end
disp(['Total training trajectories: ', num2str(size(x_slices, 2))]);
disp(['Rolled-out trajectories: ', num2str(n_eval)]);
disp(['First-level PT-CBF enabled: ', ...
    logical_to_text(cfg.variance_constraint.enabled)]);
disp(['First-level generation uncertainty threshold: ', ...
    num2str(cfg.variance_constraint.sigma2_max)]);
disp(['Uncertainty-vs-time figure: ', ...
    fullfile(output_dir, 'trajectory_gp_uncertainty_vs_time_matlab.emf')]);
if strlength(animation_path) > 0
    disp(['Single-trajectory animation: ', char(animation_path)]);
end
disp(['Final plotted ', uncertainty_level_label, ...
    ' LoG-GP uncertainty mean: ', ...
    num2str(mean(uncertainty_values(end, :, 1)))]);
disp(['Final plotted ', uncertainty_level_label, ...
    ' LoG-GP uncertainty max: ', ...
    num2str(max(uncertainty_values(end, :, 1)))]);

function cache_key = build_first_level_rollout_cache_key(cfg, traj_idx, ...
    x_init, constraint_cfg, model_collection)
metadata.cache_label = 'first-level-rollout';
metadata.cache_version = 3;
metadata.traj_idx = traj_idx(:)';
metadata.x_init = array_signature(x_init);
metadata.t_min = cfg.t_min;
metadata.t_max = cfg.rollout_t_max;
metadata.time_steps = cfg.time_steps;
metadata.rollout_seed = cfg.rollout_seed;
metadata.constraint = constraint_cache_metadata(constraint_cfg);
metadata.model = model_collection_cache_metadata(model_collection);
cache_key = jsonencode(metadata);
end

function rollout_uncertainty = evaluate_rollout_uncertainty( ...
    model_collection, rollout_times, rollout_path)
rollout_uncertainty = zeros(numel(rollout_times), size(rollout_path, 2), 1);
for sample_idx = 1:size(rollout_path, 2)
    for time_idx = 1:numel(rollout_times)
        s_now = rollout_times(time_idx);
        model = model_collection.model;
        z_now = [s_now; squeeze(rollout_path(time_idx, sample_idx, :))];
        if isfield(model, 'output_models')
            variance_set = zeros(numel(model.output_models), 1);
            for output_idx = 1:numel(model.output_models)
                [~, variance_set(output_idx)] = ...
                    model.output_models{output_idx}.predict_variance_grad(z_now);
            end
            uncertainty_now = sqrt(max(sum(variance_set), 0.0));
        else
            [~, uncertainty_now] = model.local_gp.predict_variance_grad(z_now);
        end
        rollout_uncertainty(time_idx, sample_idx, 1) = uncertainty_now;
    end
    if mod(sample_idx, 10) == 0 || sample_idx == size(rollout_path, 2)
        fprintf('  Evaluated uncertainty for %d / %d samples...\n', ...
            sample_idx, size(rollout_path, 2));
    end
end
end

function cache_key = build_second_level_rollout_cache_key(cfg, ...
    reconstructed_points, segment_x_init, segment_fixed_mask, ...
    segment_fixed_values, segment_data_transform, ...
    segment_variance_constraint, segment_model_collection)
metadata.cache_label = 'second-level-rollout';
metadata.cache_version = 9;
metadata.reconstructed_points = array_signature(reconstructed_points);
metadata.segment_x_init = array_signature(segment_x_init);
metadata.segment_fixed_mask = array_signature(segment_fixed_mask);
metadata.segment_fixed_values = array_signature(segment_fixed_values);
metadata.segment_mean = segment_data_transform.mean;
metadata.segment_std = segment_data_transform.std;
metadata.t_min = cfg.t_min;
metadata.t_max = cfg.rollout_t_max;
metadata.time_steps = cfg.time_steps;
metadata.segment_points_per_segment = cfg.segment_points_per_segment;
metadata.second_level_generation_samples = ...
    cfg.second_level_generation_samples;
metadata.second_level_rollout_seed = cfg.second_level_rollout_seed;
metadata.constraint = constraint_cache_metadata(segment_variance_constraint);
metadata.model = model_collection_cache_metadata(segment_model_collection);
cache_key = jsonencode(metadata);
end

function metadata = constraint_cache_metadata(constraint_cfg)
field_names = {'enabled', 'sigma2_max', 'generation_accuracy_threshold', ...
    'alpha_gain', 'omega_gain', 'grad_tol'};
metadata = struct();
for field_idx = 1:numel(field_names)
    field_name = field_names{field_idx};
    if isfield(constraint_cfg, field_name)
        metadata.(field_name) = constraint_cfg.(field_name);
    end
end
end

function metadata = model_collection_cache_metadata(model_collection)
field_names = {'cache_key', 'cache_label', 'model_type', 'n_training_pairs', ...
    'n_added_pairs', ...
    'n_skipped_pairs', 'n_added_per_output', 'n_skipped_per_output', ...
    'y_dim'};
metadata = struct();
for field_idx = 1:numel(field_names)
    field_name = field_names{field_idx};
    if isfield(model_collection, field_name)
        metadata.(field_name) = model_collection.(field_name);
    end
end
end

function signature = array_signature(data)
data = double(data(:));
if isempty(data)
    signature = [0, 0, 0, 0, 0, 0];
else
    signature = [numel(data), sum(data), mean(data), std(data), ...
        min(data), max(data)];
end
end

function text_value = logical_to_text(value)
if value
    text_value = 'true';
else
    text_value = 'false';
end
end
