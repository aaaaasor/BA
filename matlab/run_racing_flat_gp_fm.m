function run_racing_flat_gp_fm()
% Non-hierarchical GP flow matching: one LoG-GP over the full 65-point
% trajectory (65 x 4 = 260-D state), no levels, no QP correction at all.
% This is the "no hierarchy" ablation, not the safety ablation in
% run_racing_fm_baseline.m (which keeps the three-level structure).

root = 'C:\Users\JieJi\BA\matlab';
cd(root);
tag = 'Racing_FlatGPFM';
cfg = get_config();
rng(cfg.random_seed);

n_points   = 65;
n_samples  = cfg.first_level_generation_samples;   % 100
n_steps    = cfg.first_level_time_steps;

%% Geometry (evaluation only -- never fed back into generation)
[~, track_segment] = scenario_training_points(cfg, 5, cfg.n_train);
cfg.track_segment = track_segment;
cfg.obstacle = configure_racing_obstacles(track_segment, cfg.obstacle);
cfg.track_boundary.geometry = build_track_boundary_geometry(track_segment, ...
    struct_field_default(cfg.track_boundary, 'n_spline_points', 400), ...
    struct_field_default(cfg.track_boundary, 'spline_type', 'spline'));

%% Flat 260-D training data: one window covering all 65 points
target_points = scenario_training_points(cfg, n_points, cfg.n_train);
fprintf('target_points size: %s\n', mat2str(size(target_points)));
rng(cfg.first_level_data_seed);
[s_slices, x_slices, y_slices, ~, ~, data_transform] = ...
    build_sliding_window_training_data(target_points, cfg.t_min, 1.0, ...
    cfg.n_time_slices, n_points, 1);
n_rows = size(x_slices, 3);
fprintf('flat state dim = %d, distinct targets = %d, FM pairs = %d\n', ...
    n_rows, size(x_slices, 2), numel(s_slices) * size(x_slices, 2));
assert(n_rows == 260);

%% Fit one LoG-GP over the whole trajectory
rng(cfg.first_level_hyperparameter_seed);
flat_gp = cfg.gp;
flat_gp.o_ratio = cfg.gp.first_level_o_ratio;
flat_gp.n_pretrain = cfg.gp.first_level_n_pretrain;
flat_gp.hyperparameter_mat_path = fullfile('outputs', [tag '_Hyperparameter.mat']);
hp_timer = tic;
flat_gp = optimize_gp_hyperparameters(x_slices, y_slices, flat_gp, s_slices);
fprintf('hyperparameter time: %.1f s\n', toc(hp_timer));
flat_gp.training_accuracy_threshold = cfg.gp.first_level_training_accuracy_threshold;
rng(cfg.first_level_fit_seed);
fit_timer = tic;
model_collection = fit_or_load_loggp_model(s_slices, x_slices, y_slices, ...
    flat_gp, fullfile(root, 'outputs', [tag '_Model.mat']), 'flat');
fprintf('fit time: %.1f s\n', toc(fit_timer));
model_collection = strip_model_for_prediction(model_collection, 'Flat');

%% Pure FM rollout: no constraints, u == 0
rng(cfg.first_level_rollout_seed);
x_init = randn(n_samples, n_rows);
roll_timer = tic;
[rollout_times, traj_path] = rk4_rollout(model_collection, x_init, ...
    cfg.t_min, cfg.rollout_t_max, n_steps, [], [], cfg.parallel);
rollout_seconds = toc(roll_timer);
fprintf('rollout time: %.1f s (%.3f s/traj)\n', rollout_seconds, ...
    rollout_seconds / n_samples);

variance_values = evaluate_rollout_uncertainty(model_collection, ...
    rollout_times, traj_path);

%% Un-standardize and reshape to 65 x n_samples x 4
final_states = squeeze(traj_path(end, :, :));                 % n_samples x 260
physical = final_states .* data_transform.std(:)' + data_transform.mean(:)';
generated_points = permute(reshape(physical, n_samples, 4, n_points), [3 1 2]);
dataset_points = target_points;

%% Metrics with the same evaluator (n_segments = 1, u identically zero)
eval_cfg = cfg;
eval_cfg.segment_points_per_segment = n_points;
eval_cfg.safe_flow_evaluation.enabled = true;
eval_cfg.safe_flow_evaluation.output_path = fullfile('outputs', ...
    [tag '_Metrics_Variance_U.mat']);
n_trace = n_samples * n_steps;
zero_diag.hocbf = struct( ...
    'trace_u', zeros(n_trace, n_rows), ...
    'trace_t', repmat(rollout_times(1:n_steps), n_samples, 1), ...
    'trace_sample_idx', repelem((1:n_samples)', n_steps), ...
    'trace_step_idx', repmat((1:n_steps)', n_samples, 1), ...
    'trace_stage_idx', ones(n_trace, 1));
evaluation_time = struct('seconds_per_final_trajectory', ...
    rollout_seconds / n_samples, 'rollout_seconds', rollout_seconds);
metrics = evaluate_and_save_safeflow_metrics(eval_cfg, generated_points, ...
    dataset_points, struct(), rollout_times, variance_values, zero_diag, 1, ...
    struct('enabled', false), evaluation_time, struct(), struct(), root, ...
    physical);

fprintf('\n===== FLAT 260-D GP-FM =====\n');
disp(metrics);
end
