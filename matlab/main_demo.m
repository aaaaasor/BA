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
first_level_target_points = generate_original_training_points_2d( ...
    5, cfg.n_train);
disp('First-level target source: generated 5-point trajectories');
rng(cfg.first_level_data_seed);
[s_slices, x_slices, y_slices, target_points, source_points, ...
    target_data, source_data, trajectory_t_slices, data_transform] = build_training_data( ...
    "", cfg.t_min, 1.0, cfg.n_train, ...
    cfg.n_time_slices, first_level_target_points);

%% LoG-GP Hyperparameters
disp('Optimizing or loading first-level LoG-GP hyperparameters...');
rng(cfg.first_level_hyperparameter_seed);
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
    if cfg.cache.enabled
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

segment_gp = cfg.gp;
segment_gp.training_accuracy_threshold = ...
    cfg.gp.second_level_training_accuracy_threshold;
segment_gp.hyperparameter_mat_path = fullfile('outputs', ...
    'LoG_GP_SlidingWindow_Training_Hyperparameter.mat');
if isfield(cfg, 'small_sample') && isfield(cfg.small_sample, 'enabled') && ...
        cfg.small_sample.enabled
    segment_gp.save_hyperparameters = false;
    if isfield(cfg.small_sample, 'reuse_full_hyperparameters') && ...
            ~cfg.small_sample.reuse_full_hyperparameters
        segment_gp.hyperparameter_mat_path = "";
    end
end
segment_gp.n_pretrain = min(60000, ...
    size(segment_x_slices, 1) * size(segment_x_slices, 2));
segment_gp.max_local_gp_quantity = ceil(2.0 * ...
    size(segment_x_slices, 1) * size(segment_x_slices, 2) / ...
    segment_gp.max_local_data_quantity);
disp('Optimizing or loading second-level segment LoG-GP hyperparameters...');
rng(cfg.second_level_hyperparameter_seed);
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
disp('Second-level anchor source: first-level rollout');

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
    if cfg.cache.enabled
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
target_reference_points = zeros(size(target_points_dense, 1), ...
    n_second_level_eval, 2);
for traj_eval_idx = 1:n_eval
    for second_sample_idx = 1:n_second_level_samples
        second_eval_idx = (traj_eval_idx - 1) * n_second_level_samples + ...
            second_sample_idx;
        target_reference_points(:, second_eval_idx, :) = ...
            target_points_dense(:, traj_idx(traj_eval_idx), :);
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

target_points_plot = target_points_dense;
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
animation_path = "";
if cfg.animation.enabled && n_eval > 0
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

    traj_gp_var_table = [uncertainty_times, uncertainty_values(:, :, 1)];
    traj_gp_var_headers = ["s", ...
        arrayfun(@(idx) "path_" + string(idx) + "_uncertainty", ...
        0:(size(uncertainty_values, 2) - 1))];
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
disp(['10D ODE s slices: ', num2str(numel(s_slices))]);
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
    num2str(segment_variance_constraint.sigma2_max)]);
disp(['Second-level generated sample curves: ', ...
    num2str(n_second_level_eval)]);
disp(['Total training trajectories: ', num2str(size(x_slices, 2))]);
disp(['Rolled-out trajectories: ', num2str(n_eval)]);
disp(['First-level generation uncertainty threshold: ', ...
    num2str(cfg.variance_constraint.sigma2_max)]);
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
    ' LoG-GP uncertainty mean: ', ...
    num2str(mean(uncertainty_values(end, :, 1)))]);
disp(['Final plotted ', uncertainty_level_label, ...
    ' LoG-GP uncertainty max: ', ...
    num2str(max(uncertainty_values(end, :, 1)))]);

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

