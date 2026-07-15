% Main entry point for the MATLAB trajectory-space LoG-GP ODE demo.
% 支持外部注入配置: 若调用前在工作区里定义了非空 cfg_external，则用它作为
% 配置、并跳过默认的 clear（供 run_obstacle_ood_experiment 之类的 A/B 驱动
% 复用本脚本）。独立运行时行为不变（clear + get_config）。
if exist('cfg_external', 'var') && ~isempty(cfg_external)
    cfg = cfg_external;
    clc;
else
    clear;
    clc;
    %% Configuration
    cfg = get_config();
end
rng(cfg.random_seed);

%% Training Data
disp('Building first-level trajectory-space training data...');
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
first_level_target_points = generate_original_training_points_2d( ...
    5, cfg.n_train);
if isfield(cfg, 'first_level_use_tangent_features') && ...
        ~cfg.first_level_use_tangent_features
    first_level_target_points = first_level_target_points(:, :, 1:2);
    cfg.cache.first_level_model_path = fullfile('outputs', ...
        'LoG_GP_FirstLevel_PositionOnly_Model.mat');
    cfg.cache.first_level_rollout_path = fullfile('outputs', ...
        'LoG_GP_FirstLevel_PositionOnly_Rollout_WithHOCBF_PTCBF.mat');
    cfg.cache.first_level_hyperparameter_path = fullfile('outputs', ...
        'LoG_GP_FirstLevel_PositionOnly_Hyperparameter.mat');
    first_level_feature_mode = 'position only [x, y]';
else
    first_level_feature_mode = '[x, y, dx/ds, dy/ds] tangent';
end
disp(['First-level target source: generated 5-point trajectories with ', ...
    first_level_feature_mode, ' features']);
rng(cfg.first_level_data_seed);
[s_slices, x_slices, y_slices, target_points, source_points, ...
    target_data, source_data, trajectory_t_slices, data_transform] = build_training_data( ...
    cfg.t_min, 1.0, cfg.n_time_slices, first_level_target_points);
first_level_feature_dim = size(target_points, 3);

%% LoG-GP Hyperparameters
% 第一层使用 cfg.gp 的独立副本，避免把第一层专用的运行时字段
% （样本级索引、超参数等）写回公共模板、泄漏给后续层。
disp('Optimizing or loading first-level LoG-GP hyperparameters...');
rng(cfg.first_level_hyperparameter_seed);
first_gp = cfg.gp;
first_gp.hyperparameter_mat_path = cfg.cache.first_level_hyperparameter_path;
first_gp.n_pretrain = cfg.gp.first_level_n_pretrain;
first_gp = optimize_gp_hyperparameters(x_slices, y_slices, first_gp, s_slices);

%% Fit LoG-GP Flow Model
first_gp.training_accuracy_threshold = ...
    cfg.gp.first_level_training_accuracy_threshold;
first_gp.training_trajectory_idx_per_sample = (1:size(x_slices, 2))';
disp('Fitting first-level LoG-GP flow model...');
rng(cfg.first_level_fit_seed);
first_model_cache_path = fullfile(this_dir, cfg.cache.first_level_model_path);
first_fit_timer = tic;
model_collection = fit_or_load_loggp_model(s_slices, x_slices, y_slices, ...
    first_gp, first_model_cache_path, 'first-level');
disp(['First-level LoG-GP fit/load elapsed: ', ...
    num2str(toc(first_fit_timer), '%.1f'), ' seconds']);
plot_training_trajectory_utilization(cfg, model_collection, 'first-level');

%% RK4 Rollout
disp('Running first-level RK4 rollout...');
n_rows = size(source_data, 1);
rng(cfg.first_level_rollout_seed);
n_eval = cfg.first_level_generation_samples;
x_init = randn(n_eval, n_rows);
first_rollout_cache_path = fullfile(this_dir, cfg.cache.first_level_rollout_path);
first_rollout_cache_dir = fileparts(first_rollout_cache_path);
if ~exist(first_rollout_cache_dir, 'dir'); mkdir(first_rollout_cache_dir); end
first_rollout_constraint = make_level_variance_constraint(cfg, 'first_level');
% 障碍 CBF: 第一层状态是整条 5 点轨迹的绝对标准化坐标，点->物理位置用
% 选择子映射(absolute)。作用于全部骨架点。
if first_rollout_constraint.obstacle_enabled
    n_points_first = n_rows / first_level_feature_dim;
    first_rollout_constraint.obstacle_point_maps = build_obstacle_point_maps( ...
        data_transform, first_level_feature_dim, n_points_first, ...
        first_rollout_constraint.obstacle_points, 'absolute');
end
use_first_rollout_cache = false;
if isfile(first_rollout_cache_path)
    cached = load(first_rollout_cache_path);
    if size(cached.traj_path_10d, 2) == n_eval
        cache_has_diagnostics = isfield(cached, 'first_rollout_diagnostics') && ...
            isfield(cached.first_rollout_diagnostics, 'hocbf');
        cache_matches_constraint = isfield(cached, ...
            'saved_first_rollout_constraint') && ...
            isequaln(cached.saved_first_rollout_constraint, ...
            first_rollout_constraint);
        if cache_has_diagnostics && cache_matches_constraint
            rollout_times = cached.rollout_times;
            traj_path_10d = cached.traj_path_10d;
            first_rollout_diagnostics = cached.first_rollout_diagnostics;
            use_first_rollout_cache = true;
            disp(['Loaded cached first-level rollout: ', first_rollout_cache_path]);
        elseif ~cache_matches_constraint
            disp(['Ignoring first-level rollout cache because rollout ', ...
                'constraint config changed.']);
        else
            disp(['Ignoring first-level rollout cache without HOCBF/PTCBF ', ...
                'diagnostics.']);
        end
    else
        disp(['Ignoring first-level rollout cache with sample count ', ...
            num2str(size(cached.traj_path_10d, 2)), ...
            '; expected ', num2str(n_eval), '.']);
    end
end
if ~use_first_rollout_cache
    first_rollout_timer = tic;
    [rollout_times, traj_path_10d, first_rollout_diagnostics] = rk4_rollout(model_collection, ...
        x_init, cfg.t_min, cfg.rollout_t_max, cfg.first_level_time_steps, ...
        first_rollout_constraint);
    disp(['First-level RK4 rollout elapsed: ', ...
        num2str(toc(first_rollout_timer), '%.1f'), ' seconds']);
    saved_first_rollout_constraint = first_rollout_constraint;
    try
        save(first_rollout_cache_path, 'rollout_times', 'traj_path_10d', ...
            'first_rollout_diagnostics', 'saved_first_rollout_constraint');
    catch
        save(first_rollout_cache_path, 'rollout_times', 'traj_path_10d', ...
            'first_rollout_diagnostics', 'saved_first_rollout_constraint', ...
            '-v7.3');
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
% 终端安全投影已移除: 障碍 CBF 在三层、从 t=0 起全程生效, 规定时间
% blow-up 项保证前向不变性, 原则上不需要额外的终端投影(导师反馈)。
if isfield(cfg, 'stop_after_first_level') && cfg.stop_after_first_level
    disp('Stopping after first level because cfg.stop_after_first_level is true.');
    first_stop_plot_cfg = cfg;
    first_stop_plot_cfg.sample_curve_label = 'Stage 1 sample curves';
    plot_results(first_stop_plot_cfg, target_points, source_points, ...
        reconstructed_points);
    if isfield(cfg, 'obstacle') && cfg.obstacle.enabled
        before_cache_path = fullfile(this_dir, 'outputs', ...
            'LoG_GP_FirstLevel_Rollout_WithHOCBF_PTCBF.mat');
        if exist(before_cache_path, 'file')
            before_data = load(before_cache_path);
            before_states = squeeze(before_data.traj_path_10d(end, :, :));
            if size(before_states, 1) ~= size(before_data.traj_path_10d, 2)
                before_states = reshape(before_states, ...
                    size(before_data.traj_path_10d, 2), []);
            end
            before_physical = before_states .* data_transform.std' + ...
                data_transform.mean';
            n_before = size(before_physical, 1);
            before_points = zeros(numel(trajectory_t_slices), n_before, ...
                first_level_feature_dim);
            for before_idx = 1:n_before
                before_points(:, before_idx, :) = reshape( ...
                    before_physical(before_idx, :), ...
                    first_level_feature_dim, [])';
            end
            before_points = normalize_tangent_features(before_points);

            fig_compare = figure('Name', ...
                'First-level obstacle before/after comparison', ...
                'Color', 'w', 'WindowStyle', 'normal', ...
                'Units', 'normalized', ...
                'Position', [0.12, 0.18, 0.68, 0.62]);
            hold on;
            target_xy_compare = reshape(target_points(:, :, 1:2), [], 2);
            before_xy_compare = reshape(before_points(:, :, 1:2), [], 2);
            after_xy_compare = reshape(reconstructed_points(:, :, 1:2), [], 2);
            all_xy_compare = [target_xy_compare; before_xy_compare; ...
                after_xy_compare];
            all_xy_compare = all_xy_compare( ...
                all(isfinite(all_xy_compare), 2), :);
            x_limits_compare = [min(all_xy_compare(:, 1)), ...
                max(all_xy_compare(:, 1))];
            y_limits_compare = [min(all_xy_compare(:, 2)), ...
                max(all_xy_compare(:, 2))];
            x_padding_compare = 0.08 * max(diff(x_limits_compare), eps);
            y_padding_compare = 0.08 * max(diff(y_limits_compare), eps);
            xlim(x_limits_compare + [-x_padding_compare, x_padding_compare]);
            ylim(y_limits_compare + [-y_padding_compare, y_padding_compare]);
            draw_obstacles(cfg.obstacle, 'HandleVisibility', 'off');

            for before_idx = 1:size(before_points, 2)
                curve = squeeze(before_points(:, before_idx, 1:2));
                if before_idx == 1
                    plot(curve(:, 1), curve(:, 2), '--', ...
                        'Color', [0.45 0.45 0.45], 'LineWidth', 1.2, ...
                        'DisplayName', 'without obstacle CBF');
                else
                    plot(curve(:, 1), curve(:, 2), '--', ...
                        'Color', [0.45 0.45 0.45], 'LineWidth', 1.2, ...
                        'HandleVisibility', 'off');
                end
            end

            after_colors = lines(size(reconstructed_points, 2));
            for after_idx = 1:size(reconstructed_points, 2)
                curve = squeeze(reconstructed_points(:, after_idx, 1:2));
                if after_idx == 1
                    plot(curve(:, 1), curve(:, 2), '-', ...
                        'Color', after_colors(after_idx, :), ...
                        'LineWidth', 1.8, ...
                        'DisplayName', 'with obstacle CBF');
                    plot(curve(:, 1), curve(:, 2), 'o', ...
                        'Color', after_colors(after_idx, :), ...
                        'MarkerFaceColor', after_colors(after_idx, :), ...
                        'MarkerSize', 4, 'LineStyle', 'none', ...
                        'DisplayName', 'generated points');
                else
                    plot(curve(:, 1), curve(:, 2), '-', ...
                        'Color', after_colors(after_idx, :), ...
                        'LineWidth', 1.8, 'HandleVisibility', 'off');
                    plot(curve(:, 1), curve(:, 2), 'o', ...
                        'Color', after_colors(after_idx, :), ...
                        'MarkerFaceColor', after_colors(after_idx, :), ...
                        'MarkerSize', 4, 'LineStyle', 'none', ...
                        'HandleVisibility', 'off');
                end
            end
            grid on;
            axis equal;
            xlabel('x');
            ylabel('y');
            title('First-level obstacle avoidance: before vs after');
            legend('Location', 'northeastoutside');
            if cfg.output.enabled
                before_after_path = fullfile(this_dir, 'outputs', ...
                    'FirstLevel_Obstacle_BeforeAfter_NearP3.png');
                exportgraphics(fig_compare, before_after_path, ...
                    'Resolution', 200);
                disp(['Saved before/after obstacle comparison: ', ...
                    before_after_path]);
            end

            disp(['Evaluating first-level predictive variance for ', ...
                'before/after obstacle comparison...']);
            before_uncertainty = evaluate_rollout_uncertainty( ...
                model_collection, before_data.rollout_times, ...
                before_data.traj_path_10d);
            after_uncertainty = evaluate_rollout_uncertainty( ...
                model_collection, rollout_times, traj_path_10d);
            % 每个输出维度先除以其 GP 先验方差 SigmaF_i^2，再跨输出维度聚合。
            % 这样消除不同输出核振幅的量纲影响，同时保留避障前后的 OOD 差异。
            sigma_f_squared = reshape( ...
                max(first_gp.signal_std_vec(:).^2, eps), 1, 1, []);
            before_normalized_variance = before_uncertainty ./ ...
                sigma_f_squared;
            after_normalized_variance = after_uncertainty ./ ...
                sigma_f_squared;
            n_variance_outputs = size(before_uncertainty, 3);
            before_sigma = sqrt(aggregate_variance_values( ...
                before_normalized_variance) ./ n_variance_outputs);
            after_sigma = sqrt(aggregate_variance_values( ...
                after_normalized_variance) ./ n_variance_outputs);

            fig_variance = figure('Name', ...
                'First-level predictive variance before/after obstacle', ...
                'Color', 'w', 'WindowStyle', 'normal', ...
                'Units', 'normalized', ...
                'Position', [0.16, 0.20, 0.58, 0.46]);
            hold on;
            for before_idx = 1:size(before_sigma, 2)
                if before_idx == 1
                    plot(before_data.rollout_times, before_sigma(:, before_idx), ...
                        '--', 'Color', [0.45 0.45 0.45], ...
                        'LineWidth', 1.1, ...
                        'DisplayName', 'without obstacle CBF');
                else
                    plot(before_data.rollout_times, before_sigma(:, before_idx), ...
                        '--', 'Color', [0.45 0.45 0.45], ...
                        'LineWidth', 1.1, 'HandleVisibility', 'off');
                end
            end
            for after_idx = 1:size(after_sigma, 2)
                if after_idx == 1
                    plot(rollout_times, after_sigma(:, after_idx), ...
                        '-', 'Color', [0.00 0.45 0.74], ...
                        'LineWidth', 1.4, ...
                        'DisplayName', 'with obstacle CBF');
                else
                    plot(rollout_times, after_sigma(:, after_idx), ...
                        '-', 'Color', [0.00 0.45 0.74], ...
                        'LineWidth', 1.4, 'HandleVisibility', 'off');
                end
            end
            grid on;
            xlabel('s');
            ylabel(['normalized RMS uncertainty ', ...
                'sqrt(mean_i(\sigma_i^2 / \Sigma_{F,i}^2))']);
            title(['First-level normalized predictive uncertainty: ', ...
                'before vs after obstacle CBF']);
            legend('Location', 'best');
            if cfg.output.enabled
                variance_path = fullfile(this_dir, 'outputs', ...
                    'FirstLevel_Obstacle_BeforeAfter_Variance_NearP3.png');
                exportgraphics(fig_variance, variance_path, ...
                    'Resolution', 200);
                disp(['Saved before/after obstacle variance plot: ', ...
                    variance_path]);
            end
        else
            disp(['Skipping before/after obstacle comparison because ', ...
                'baseline cache is missing: ', before_cache_path]);
        end

    end
    plot_hocbf_psi_trace_diagnostics(cfg, first_rollout_diagnostics);

    % 导师要求的避障 CBF 函数值随生成时间变化。第一层同时约束5个
    % 轨迹点，因此用最小值表示当前时刻最危险点的安全裕度。
    if isfield(first_rollout_diagnostics, 'hocbf') && ...
            isfield(first_rollout_diagnostics.hocbf, ...
            'trace_obstacle_h_min')
        obstacle_trace = first_rollout_diagnostics.hocbf;
        obstacle_h = obstacle_trace.trace_obstacle_h_min(:);
        obstacle_t = obstacle_trace.trace_t(:);
        obstacle_sample_idx = obstacle_trace.trace_sample_idx(:);
        obstacle_step_idx = obstacle_trace.trace_step_idx(:);
        obstacle_stage_idx = obstacle_trace.trace_stage_idx(:);
        valid_obstacle_rows = isfinite(obstacle_h) & isfinite(obstacle_t);
        if any(valid_obstacle_rows)
            obstacle_h = obstacle_h(valid_obstacle_rows);
            obstacle_t = obstacle_t(valid_obstacle_rows);
            obstacle_sample_idx = obstacle_sample_idx(valid_obstacle_rows);
            obstacle_step_idx = obstacle_step_idx(valid_obstacle_rows);
            obstacle_stage_idx = obstacle_stage_idx(valid_obstacle_rows);
            [~, obstacle_sort_idx] = sortrows([obstacle_sample_idx, ...
                obstacle_step_idx, obstacle_stage_idx]);
            obstacle_h = obstacle_h(obstacle_sort_idx);
            obstacle_t = obstacle_t(obstacle_sort_idx);
            obstacle_sample_idx = obstacle_sample_idx(obstacle_sort_idx);

            fig_obstacle_h = figure('Name', ...
                'First-level obstacle CBF value over time', ...
                'Color', 'w', 'WindowStyle', 'normal', ...
                'Units', 'normalized', ...
                'Position', [0.17, 0.18, 0.60, 0.48]);
            hold on;
            obstacle_samples = unique(obstacle_sample_idx)';
            for obstacle_sample = obstacle_samples
                sample_rows = obstacle_sample_idx == obstacle_sample;
                plot(obstacle_t(sample_rows), obstacle_h(sample_rows), ...
                    'Color', [0.00, 0.45, 0.74], 'LineWidth', 1.0, ...
                    'HandleVisibility', 'off');
            end
            plot(nan, nan, '-', 'Color', [0.00, 0.45, 0.74], ...
                'LineWidth', 1.2, 'DisplayName', ...
                'minimum obstacle CBF value');
            yline(0, '--', 'Color', [0.85, 0.20, 0.20], ...
                'LineWidth', 1.2, 'DisplayName', 'h = 0 safety boundary');
            grid on;
            xlabel('t');
            ylabel('h_{min}(t) = min_j h_j(x_j(t))');
            title('First-level obstacle CBF value over time');
            legend('Location', 'best');
            if cfg.output.enabled
                obstacle_h_path = fullfile(this_dir, 'outputs', ...
                    'FirstLevel_Obstacle_CBF_Value_Over_Time.emf');
                export_graphics_compat(fig_obstacle_h, obstacle_h_path);
                disp(['Saved obstacle CBF time trace: ', obstacle_h_path]);
            end
        else
            disp(['Skipping obstacle CBF time trace because no finite ', ...
                'obstacle h values were recorded.']);
        end
    end
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
    target_points_dense(first_level_anchor_idx, :, 1:first_level_feature_dim) - ...
    target_points), [], 'all');
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
segment_gp.training_trajectory_idx_per_sample = ...
    segment_data_transform.trajectory_idx_per_sample;
segment_gp.training_sample_order = segment_data_transform.sample_order;
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

second_level_anchor_points = reconstructed_points;

n_generation_segments = size(second_level_anchor_points, 1) - 1;
n_second_level_samples = cfg.second_level_generation_samples;
n_second_level_eval = n_eval * n_second_level_samples;
n_segment_eval = n_second_level_eval * n_generation_segments;
segment_feature_dim = size(second_level_anchor_points, 3);
n_segment_rows = segment_feature_dim * cfg.segment_points_per_segment;
rng(cfg.second_level_rollout_seed);
segment_x_init = randn(n_segment_eval, n_segment_rows);
segment_variance_constraint = make_level_variance_constraint(cfg, ...
    'second_level');
segment_rollout_constraint = segment_variance_constraint;
if segment_rollout_constraint.ptclf_enabled
    [anchor_clf_targets, anchor_clf_indices] = ...
        build_anchor_clf_targets(second_level_anchor_points, n_eval, ...
        n_second_level_samples, n_generation_segments, ...
        cfg.segment_points_per_segment, segment_data_transform);
    % 对照实验: PTCLF 只约束位置(x,y),不约束斜率(dx/ds,dy/ds)。
    % anchor_clf_indices/targets 按每个 anchor 点 [x,y,dx/ds,dy/ds] 4 维
    % 一组排列(见 build_anchor_clf_targets.m),每组前 2 维是位置。
    if struct_field_default(segment_rollout_constraint, ...
            'anchor_clf_position_only', false)
        anchor_feature_dim = 4;
        keep_mask = mod(0:(numel(anchor_clf_indices) - 1), ...
            anchor_feature_dim) < 2;
        anchor_clf_indices = anchor_clf_indices(keep_mask);
        anchor_clf_targets = anchor_clf_targets(:, keep_mask);
    end
    segment_rollout_constraint.anchor_clf_targets = anchor_clf_targets;
    segment_rollout_constraint.anchor_clf_indices = anchor_clf_indices;
end
% 障碍 CBF: 第二层每段是 5 点绝对标准化坐标(选择子映射)，只约束内部点。
if segment_rollout_constraint.obstacle_enabled
    segment_rollout_constraint.obstacle_point_maps = build_obstacle_point_maps( ...
        segment_data_transform, segment_feature_dim, ...
        cfg.segment_points_per_segment, ...
        segment_rollout_constraint.obstacle_points, 'absolute');
end
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
            segment_rollout_constraint, 'ptcbf_enabled', false);
        cache_matches_constraint = isfield(cached, ...
            'saved_segment_rollout_constraint') && ...
            isequaln(cached.saved_segment_rollout_constraint, ...
            segment_rollout_constraint);
        cache_matches_seed = isfield(cached, ...
            'saved_second_level_rollout_seed') && ...
            isequaln(cached.saved_second_level_rollout_seed, ...
            cfg.second_level_rollout_seed);
        if (~needs_terminal_diagnostics || cache_has_terminal_diagnostics) && ...
                cache_matches_constraint && cache_matches_seed
            segment_rollout_times = cached.segment_rollout_times;
            segment_traj_path_10d = cached.segment_traj_path_10d;
            rollout_diagnostics = cached.rollout_diagnostics;
            use_segment_rollout_cache = true;
            disp(['Loaded cached second-level rollout: ', segment_rollout_cache_path]);
        elseif ~cache_matches_constraint
            disp(['Ignoring second-level rollout cache because rollout ', ...
                'constraint config changed.']);
        elseif ~cache_matches_seed
            disp(['Ignoring second-level rollout cache because rollout ', ...
                'seed changed.']);
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
    second_level_refine_cfg = struct( ...
        'start_t', cfg.second_level_time_refine_start_t, ...
        'extra_steps', cfg.second_level_time_refine_extra_steps);
    [segment_rollout_times, segment_traj_path_10d, rollout_diagnostics] = ...
        rk4_rollout(segment_model_collection, segment_x_init, cfg.t_min, ...
        cfg.rollout_t_max, cfg.second_level_time_steps, segment_rollout_constraint, ...
        second_level_refine_cfg);
    disp(['Second-level RK4 rollout elapsed: ', ...
        num2str(toc(second_rollout_timer), '%.1f'), ' seconds']);
    saved_segment_rollout_constraint = segment_rollout_constraint;
    saved_second_level_rollout_seed = cfg.second_level_rollout_seed;
    try
        save(segment_rollout_cache_path, 'segment_rollout_times', ...
            'segment_traj_path_10d', 'rollout_diagnostics', ...
            'saved_segment_rollout_constraint', ...
            'saved_second_level_rollout_seed');
    catch
        save(segment_rollout_cache_path, 'segment_rollout_times', ...
            'segment_traj_path_10d', 'rollout_diagnostics', ...
            'saved_segment_rollout_constraint', ...
            'saved_second_level_rollout_seed', '-v7.3');
    end
    disp(['Saved cached second-level rollout: ', segment_rollout_cache_path]);
end
% psi/QP 诊断图导出到固定文件名；开启第三层时只导出第三层的版本，
% 避免这里的第二层图被覆盖或反过来占住文件名。
if ~(isfield(cfg, 'enable_third_level') && cfg.enable_third_level)
    segment_plot_cfg = cfg;
    segment_plot_cfg.hocbf_focus_n_segments = n_generation_segments;
    plot_hocbf_psi_trace_diagnostics(segment_plot_cfg, rollout_diagnostics);
end
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

% PTCLF 只是软化/硬化约束把每段首尾点"拉近"第一层生成点，收敛后仍有
% 残余误差（即前面 "internal anchor gap" 诊断打印的量）。这里直接用
% anchor CLF 的目标值（就是第一层生成点本身）覆盖 rollout 结果里对应
% 的首尾点特征，消除这部分残余误差，保证第三层 PTCLF 目标端点精确。
if segment_rollout_constraint.ptclf_enabled
    anchor_clf_targets_physical = anchor_clf_targets .* ...
        segment_data_transform.std(anchor_clf_indices)' + ...
        segment_data_transform.mean(anchor_clf_indices)';
    final_segment_data(:, anchor_clf_indices) = anchor_clf_targets_physical;
end

target_points_plot = target_points_dense;
initial_segment_data = squeeze(segment_traj_path_plot(1, :, :));
if size(initial_segment_data, 1) == 1
    initial_segment_data = reshape(initial_segment_data, 1, []);
end
source_points_plot = segment_points_to_point_list(initial_segment_data, ...
    n_second_level_eval, n_generation_segments, cfg.segment_points_per_segment);
source_points_plot = normalize_tangent_features(source_points_plot);
reconstructed_points_plot = zeros(0, n_second_level_eval, ...
    size(target_points_dense, 3));
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
    % 第三层训练使用 65 点 fine 轨迹，按 stride = 4 切成 16 个非滑窗
    % 窗口（1-5, 5-9, ..., 61-65），每个窗口跨度正好对应 17 点网格的
    % 一个间隔；窗口内用局部增量表示 [S0, S1-S0, ..., S4-S3]。
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
    % 提供第三层自己的样本->轨迹索引（用于按轨迹的利用率统计）：
    % 训练样本顺序是外层窗口、内层轨迹。
    third_segment_gp.training_trajectory_idx_per_sample = repmat( ...
        (1:size(target_points_fine, 2))', ...
        numel(third_segment_data_transform.window_start_idx), 1);
    third_segment_gp.hyperparameter_mat_path = cfg.cache.third_level_hyperparameter_path;
    third_segment_gp.n_pretrain = min(cfg.gp.third_level_n_pretrain, ...
        size(third_segment_x_slices, 1) * size(third_segment_x_slices, 2));
    third_segment_gp.max_local_gp_quantity = ceil(2.0 * ...
        size(third_segment_x_slices, 1) * ...
        size(third_segment_x_slices, 2) / ...
        third_segment_gp.max_local_data_quantity);
    disp('Optimizing or loading third-level segment LoG-GP hyperparameters...');
    rng(cfg.second_level_hyperparameter_seed + 100);
    third_segment_gp = optimize_gp_hyperparameters(third_segment_x_slices, ...
        third_segment_y_slices, third_segment_gp, third_segment_s_slices);
    third_segment_gp.noise_std_vec = third_segment_gp.noise_std_vec / 50;

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

    % 第三层是细化层：不拼接第二层各段，直接以第二层每段自己的 5 个
    % 点为参考，每个父段细分出 4 个第三层 segment（共 4x4=16 段），
    % PTCLF 只钉每个第三层 segment 的首尾两点（父段曲线的第 j / j+1
    % 点），中间 3 个点自由细化。相邻父段边界两侧的目标互不混合。
    n_third_anchor_eval = n_second_level_eval;
    n_third_generation_segments = n_generation_segments * ...
        (cfg.segment_points_per_segment - 1);
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
    % 与第二层一致：统一从 make_level_variance_constraint 构造约束
    % （含 slack 切换：t < switch 时 PTCLF 带 slack、HOCBF/PTCBF 硬，
    % 之后互换）。
    third_segment_variance_constraint = make_level_variance_constraint( ...
        cfg, 'third_level');
    % 目标同时供 PTCLF 约束和后面的端点误差诊断使用。
    [third_anchor_clf_targets, third_anchor_clf_matrix, ...
        third_anchor_clf_offset] = build_increment_endpoint_clf_targets( ...
        final_segment_data, n_third_anchor_eval, n_third_samples, ...
        n_generation_segments, cfg.segment_points_per_segment, ...
        third_segment_data_transform);
    if third_segment_variance_constraint.ptclf_enabled
        third_segment_variance_constraint.anchor_clf_targets = ...
            third_anchor_clf_targets;
        third_segment_variance_constraint.anchor_clf_matrix = ...
            third_anchor_clf_matrix;
        third_segment_variance_constraint.anchor_clf_offset = ...
            third_anchor_clf_offset;
    end
    % 障碍 CBF: 第三层是局部增量表示，点->物理位置用累加矩阵映射(increment)，
    % 只约束每段内部点(端点被硬覆盖成已避障的 L2 点)。
    if third_segment_variance_constraint.obstacle_enabled
        third_segment_variance_constraint.obstacle_point_maps = ...
            build_obstacle_point_maps(third_segment_data_transform, ...
            segment_feature_dim, cfg.segment_points_per_segment, ...
            third_segment_variance_constraint.obstacle_points, 'increment');
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
                'ptcbf_enabled', false);
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
        % 在 PTZF 包络急剧收紧的尾段加密时间步。第三层可用独立的加密
        % 配置(更密的尾段步),避免牵连第二层;未配置时回退到第二层的值。
        third_refine_start_t = struct_field_default(cfg, ...
            'third_level_time_refine_start_t', ...
            cfg.second_level_time_refine_start_t);
        third_refine_extra_steps = struct_field_default(cfg, ...
            'third_level_time_refine_extra_steps', ...
            cfg.second_level_time_refine_extra_steps);
        third_level_refine_cfg = struct( ...
            'start_t', third_refine_start_t, ...
            'extra_steps', third_refine_extra_steps);
        [third_rollout_times, third_traj_path_10d, third_rollout_diagnostics] = ...
            rk4_rollout(third_segment_model_collection, third_segment_x_init, ...
            cfg.t_min, third_level_rollout_t_max, third_level_time_steps, ...
            third_segment_variance_constraint, third_level_refine_cfg);
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
    end
    % 无论 rollout 是否走缓存，psi/QP 诊断图都画第三层的版本。
    if exist('third_rollout_diagnostics', 'var')
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
    % 端点硬覆盖(仅障碍避障用): 把每段首/末点覆盖成 L2 的目标点，继承已
    % 避障的 L2 端点安全(障碍 CBF 只作用内部点 2,3,4)。一般的段间对齐已
    % 由 rk4_rollout 里的"倒数 anchor_snap_flow_steps 步投影钉住"在积分
    % 过程中完成，不再需要事后覆盖。
    obstacle_enabled_for_snap = isfield(cfg, 'obstacle') && ...
        struct_field_default(cfg.obstacle, 'enabled', false);
    if obstacle_enabled_for_snap
        F_endpoint = segment_feature_dim;
        last_block0 = (cfg.segment_points_per_segment - 1) * F_endpoint;
        final_third_segment_data(:, 1:F_endpoint) = ...
            third_anchor_clf_targets(:, 1:F_endpoint);
        final_third_segment_data(:, last_block0 + (1:F_endpoint)) = ...
            third_anchor_clf_targets(:, F_endpoint + (1:F_endpoint));
    end
    % 不拼接：每段 5 个点全部保留（相邻段边界点各自独立，接不上时
    % 如实显示），每条轨迹输出 16段 x 5点 = 80 个点。
    reconstructed_points_fine = segment_points_to_point_list( ...
        final_third_segment_data, n_third_level_eval, ...
        n_third_generation_segments, cfg.segment_points_per_segment);
    reconstructed_points_fine = normalize_tangent_features( ...
        reconstructed_points_fine);
    % source 面板直接画第三层 ODE 的原始 randn 初值（标准化增量空间，
    % 每条轨迹 16段 x 5点 = 80 个点），不做反标准化/增量还原。
    third_source_points_raw = segment_points_to_point_list( ...
        third_segment_x_init, n_third_level_eval, ...
        n_third_generation_segments, cfg.segment_points_per_segment);

    % 端点误差诊断：每个第三层 segment 自己的首尾点(切向归一化后)
    % 与它的 PTCLF 目标逐段比较，即 CLF 的实际跟踪误差。
    third_realized_endpoints = [ ...
        final_third_segment_data(:, 1:segment_feature_dim), ...
        final_third_segment_data(:, (cfg.segment_points_per_segment - 1) * ...
        segment_feature_dim + (1:segment_feature_dim))];
    for endpoint_block0 = [0, segment_feature_dim]
        tangent_cols = endpoint_block0 + (3:4);
        tangent_norm = max(sqrt(sum( ...
            third_realized_endpoints(:, tangent_cols) .^ 2, 2)), eps);
        third_realized_endpoints(:, tangent_cols) = ...
            third_realized_endpoints(:, tangent_cols) ./ tangent_norm;
    end
    third_level_anchor_error = max(abs( ...
        third_realized_endpoints - third_anchor_clf_targets), [], 'all');
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
    n_generation_segments = n_third_generation_segments;
    n_second_level_eval = n_third_level_eval;
    segment_rollout_times = third_rollout_times;
    segment_traj_path_plot = third_traj_path_plot;
    segment_traj_path_10d = third_traj_path_10d;
end

%% Plot Results
disp('Plotting and exporting results...');
plot_cfg = cfg;
plot_cfg.obstacle = cfg.obstacle;   % 供绘图叠加椭圆(若 plot_results 支持)
plot_cfg.first_level_generation_samples = size(reconstructed_points_plot, 2);
final_segment_endpoints = squeeze(segment_traj_path_plot(end, :, :));
if n_segment_eval == 1
    final_segment_endpoints = reshape(final_segment_endpoints, 1, []);
end
% Raw (unstitched) per-segment anchor estimates: each segment keeps its own
% first and last point, so shared boundaries show both segments'
% independent CLF estimates instead of silently dropping one of them.
raw_segment_anchor_points = zeros(2 * n_generation_segments, ...
    n_second_level_eval, segment_feature_dim);
for traj_idx = 1:n_second_level_eval
    for segment_idx = 1:n_generation_segments
        sample_idx = (traj_idx - 1) * n_generation_segments + segment_idx;
        segment_curve = reshape(final_segment_endpoints(sample_idx, :), ...
            segment_feature_dim, [])';
        raw_segment_anchor_points(2 * segment_idx - 1, traj_idx, :) = ...
            segment_curve(1, :);
        raw_segment_anchor_points(2 * segment_idx, traj_idx, :) = ...
            segment_curve(end, :);
    end
end
% 每个内部锚点(相邻segment共享的边界),两段各自独立算出来的位置/切向
% 估计值之间的欧氏距离——这才是真正的"贴合差距"，不是QP的slack值。
n_internal_anchors = n_generation_segments - 1;
if n_internal_anchors > 0
    internal_gap_position = zeros(n_internal_anchors, n_second_level_eval);
    internal_gap_full = zeros(n_internal_anchors, n_second_level_eval);
    for traj_idx = 1:n_second_level_eval
        for boundary_idx = 1:n_internal_anchors
            left_point = squeeze(raw_segment_anchor_points( ...
                2 * boundary_idx, traj_idx, :));
            right_point = squeeze(raw_segment_anchor_points( ...
                2 * boundary_idx + 1, traj_idx, :));
            internal_gap_full(boundary_idx, traj_idx) = ...
                norm(left_point - right_point);
            internal_gap_position(boundary_idx, traj_idx) = ...
                norm(left_point(1:2) - right_point(1:2));
        end
    end
    disp([uncertainty_level_label, ...
        ' internal anchor gap (position, x-y) mean/max: ', ...
        num2str(mean(internal_gap_position(:))), ' / ', ...
        num2str(max(internal_gap_position(:)))]);
    disp([uncertainty_level_label, ...
        ' internal anchor gap (full feature vector) mean/max: ', ...
        num2str(mean(internal_gap_full(:))), ' / ', ...
        num2str(max(internal_gap_full(:)))]);
end
plot_cfg.anchor_points = raw_segment_anchor_points;
plot_cfg.anchor_target_points = second_level_anchor_points;
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
    % 锚点目标改画第三层 PTCLF 的目标端点(第二层每段的相邻两点)，
    % 布局与 raw_segment_anchor_points 一致: 每段 [首点; 末点]。
    if exist('third_anchor_clf_targets', 'var')
        third_anchor_target_points = zeros(2 * n_third_generation_segments, ...
            n_third_level_eval, segment_feature_dim);
        for traj_idx = 1:n_third_level_eval
            for segment_idx = 1:n_third_generation_segments
                target_row = (traj_idx - 1) * n_third_generation_segments + ...
                    segment_idx;
                third_anchor_target_points(2 * segment_idx - 1, traj_idx, :) = ...
                    third_anchor_clf_targets(target_row, 1:segment_feature_dim);
                third_anchor_target_points(2 * segment_idx, traj_idx, :) = ...
                    third_anchor_clf_targets(target_row, ...
                    segment_feature_dim + (1:segment_feature_dim));
            end
        end
        plot_cfg.anchor_target_points = third_anchor_target_points;
    end
end
plot_results(plot_cfg, target_points_plot, source_points_plot, ...
    reconstructed_points_plot);
if ~isempty(uncertainty_values)
    plot_sigma_vs_time(cfg, uncertainty_times, uncertainty_values, ...
        uncertainty_level_label, uncertainty_rollout_diagnostics);
    plot_anchor_clf_g_vs_time(cfg, uncertainty_level_label, ...
        uncertainty_rollout_diagnostics);
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
    % source 散点取解码轨迹的第一帧：randn 初值经反标准化(+ 第三层的
    % 增量累加)映射到物理坐标，与 rollout 轨迹严格同一坐标系。
    source_segment_rows = squeeze(animation_traj_path( ...
        1, segment_animation_idx, :));
    if size(source_segment_rows, 1) == 1
        source_segment_rows = reshape(source_segment_rows, 1, []);
    end
    % 每段的 source 起点是独立采样的随机点，和上一段终点并不重合，
    % 把所有段的点原样展开，保留每段真实的首尾点。
    source_feature_dim = size(source_segment_rows, 2) / ...
        cfg.segment_points_per_segment;
    animation_source_points = zeros( ...
        animation_segment_count * cfg.segment_points_per_segment, ...
        source_feature_dim);
    for segment_idx = 1:animation_segment_count
        segment_curve = reshape(source_segment_rows(segment_idx, :), ...
            source_feature_dim, [])';
        row_start = (segment_idx - 1) * cfg.segment_points_per_segment + 1;
        row_end = row_start + cfg.segment_points_per_segment - 1;
        animation_source_points(row_start:row_end, :) = segment_curve;
    end
    animation_cfg = cfg;
    animation_cfg.animation.segment_plot_path = ...
        animation_traj_path(:, segment_animation_idx, :);
    animation_cfg.animation.segment_plot_count = animation_segment_count;
    animation_cfg.animation.segment_plot_points_per_segment = ...
        cfg.segment_points_per_segment;
    if animation_use_third_global_rollout
        animation_cfg.animation.space_label = 'Original Global Space';
    else
        % 第二层的端点在 snap_path_index 时刚被钉住；之后 5 个尾段 RK4
        % 步只让段内点继续 FM。强制逐帧保留这一整段，不能被 frame_stride
        % 跳过，方便动画直观看到“先 snap、后 FM”。
        snap_flow_steps = struct_field_default(segment_rollout_constraint, ...
            'anchor_snap_flow_steps', 0);
        if snap_flow_steps > 0
            snap_path_index = numel(animation_times) - snap_flow_steps;
            animation_cfg.animation.forced_frame_indices = ...
                snap_path_index:numel(animation_times);
        end
    end
    animation_path = animate_single_trajectory(animation_cfg, animation_times, ...
        traj_path_single, animation_source_points);
end


%% Console Summary
disp(['Training trajectories loaded: ', num2str(size(target_data, 2))]);
disp(['Trajectory points per sample: ', num2str(numel(trajectory_t_slices))]);
disp(['First-level feature mode: ', first_level_feature_mode]);
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
if strlength(animation_path) > 0
    disp(['Single-trajectory animation: ', char(animation_path)]);
end
