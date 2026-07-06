% 绘制三列摘要图：目标轨迹、source 轨迹、roll out 轨迹
function plot_results(cfg, target_points, source_points, reconstructed_points)
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
if cfg.output.enabled && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_path = fullfile(output_dir, 'gp_flow_matching_demo_matlab.emf');

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.08, 0.18, 0.84, 0.48]);
movegui(fig, 'center');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% 计算目标和rollout的联合坐标范围，使三列图共享轴限
use_segment_plot = isfield(cfg, 'segment_plot_data') && ...
    ~isempty(cfg.segment_plot_data) && ...
    isfield(cfg, 'segment_plot_count') && cfg.segment_plot_count > 0 && ...
    isfield(cfg, 'segment_plot_points_per_segment');
if use_segment_plot
    segment_data = cfg.segment_plot_data;
    n_segments = cfg.segment_plot_count;
    n_points = cfg.segment_plot_points_per_segment;
    feature_dim = size(segment_data, 2) / n_points;
    n_segment_samples = floor(size(segment_data, 1) / n_segments);
else
    segment_data = [];
    n_segments = 0;
    n_points = 0;
    feature_dim = 0;
    n_segment_samples = 0;
end
target_xy = reshape(target_points(:, :, 1:2), [], 2);
if isempty(reconstructed_points)
    rollout_xy = zeros(0, 2);
else
    rollout_xy = reshape(reconstructed_points(:, :, 1:2), [], 2);
end
if use_segment_plot
    seg_curves = reshape(segment_data', feature_dim, n_points, []);
    seg_xy = permute(seg_curves(1:2, :, :), [2, 3, 1]);
    rollout_xy = [rollout_xy; reshape(seg_xy, [], 2)];
end
comparison_xy = [target_xy; rollout_xy];
comparison_xy = comparison_xy(all(isfinite(comparison_xy), 2), :);
x_limits = [min(comparison_xy(:, 1)), max(comparison_xy(:, 1))];
y_limits = [min(comparison_xy(:, 2)), max(comparison_xy(:, 2))];
x_padding = 0.05 * max(diff(x_limits), eps);
y_padding = 0.05 * max(diff(y_limits), eps);
x_limits = x_limits + [-x_padding, x_padding];
y_limits = y_limits + [-y_padding, y_padding];

%% Target Trajectories
nexttile;
hold on;
n_training_curves = size(target_points, 2);
for idx = 1:n_training_curves
    training_curve = squeeze(target_points(:, idx, 1:2));
    plot(training_curve(:, 1), training_curve(:, 2), '.-', 'LineWidth', 0.8, ...
        'HandleVisibility', 'off');
end
grid on;
axis equal;
xlim(x_limits);
ylim(y_limits);
xlabel('x');
ylabel('y');
title(sprintf('Target Trajectory Data (2D, %d Points)', size(target_points, 1)));

%% Source Trajectories
nexttile;
hold on;
max_curves = min(size(source_points, 2), cfg.first_level_generation_samples);
for idx = 1:max_curves
    source_curve = squeeze(source_points(:, idx, 1:2));
    plot(source_curve(:, 1), source_curve(:, 2), '--', 'LineWidth', 0.9);
end
grid on;
axis equal;
xlabel('x');
ylabel('y');
title('ODE Source Trajectories');

%% Rollout Trajectories
nexttile;
hold on;
if use_segment_plot
    max_curves = min(n_segment_samples, cfg.first_level_generation_samples);
    total_rollout_curves = n_segment_samples;
else
    max_curves = min(size(reconstructed_points, 2), ...
        cfg.first_level_generation_samples);
    total_rollout_curves = size(reconstructed_points, 2);
end
fprintf('Plotting rollout sample curves: %d / %d\n', ...
    max_curves, total_rollout_curves);
rollout_colors = lines(max(max_curves, 1));
if use_segment_plot
    curve_label = struct_field_default(cfg, 'sample_curve_label', 'Stage 2 sample curves');
    marker_label = struct_field_default(cfg, 'generated_point_label', 'Stage 2 generated points');
    for traj_idx = 1:max_curves
        for segment_idx = 1:n_segments
            sample_idx = (traj_idx - 1) * n_segments + segment_idx;
            if sample_idx > size(segment_data, 1)
                continue;
            end
            segment_curve = reshape(segment_data(sample_idx, :), feature_dim, [])';
            segment_curve = segment_curve(:, 1:2);
            color = rollout_colors(traj_idx, :);
            if traj_idx == 1 && segment_idx == 1
                plot(segment_curve(:, 1), segment_curve(:, 2), ...
                    'Color', color, 'LineWidth', 1.1, 'DisplayName', curve_label);
                plot(segment_curve(:, 1), segment_curve(:, 2), 'o', ...
                    'Color', color, 'MarkerFaceColor', color, ...
                    'MarkerSize', 4, 'LineStyle', 'none', 'DisplayName', marker_label);
            else
                plot(segment_curve(:, 1), segment_curve(:, 2), ...
                    'Color', color, 'LineWidth', 1.1, 'HandleVisibility', 'off');
                plot(segment_curve(:, 1), segment_curve(:, 2), 'o', ...
                    'Color', color, 'MarkerFaceColor', color, ...
                    'MarkerSize', 4, 'LineStyle', 'none', 'HandleVisibility', 'off');
            end
        end
    end
else
    for idx = 1:max_curves
        reconstructed_curve = squeeze(reconstructed_points(:, idx, 1:2));
        if idx == 1
            plot(reconstructed_curve(:, 1), reconstructed_curve(:, 2), ...
                'Color', rollout_colors(idx, :), 'LineWidth', 1.1, ...
                'DisplayName', struct_field_default(cfg, ...
                'sample_curve_label', 'Stage 2 sample curves'));
        else
            plot(reconstructed_curve(:, 1), reconstructed_curve(:, 2), ...
                'Color', rollout_colors(idx, :), 'LineWidth', 1.1, ...
                'HandleVisibility', 'off');
        end
    end
end
if isfield(cfg, 'reference_points') && ~isempty(cfg.reference_points)
    reference_points = cfg.reference_points;
    n_reference_curves = min(size(reference_points, 2), max_curves);
    for idx = 1:n_reference_curves
        reference_curve = squeeze(reference_points(:, idx, :));
        reference_curve = reference_curve(:, 1:2);
        if idx == 1
            plot(reference_curve(:, 1), reference_curve(:, 2), ...
                'Color', [0.9, 0.0, 0.0], 'LineStyle', '-', ...
                'LineWidth', 3.0, 'DisplayName', 'Training reference curve');
        else
            plot(reference_curve(:, 1), reference_curve(:, 2), ...
                'Color', [0.9, 0.0, 0.0], 'LineStyle', '-', ...
                'LineWidth', 2.0, 'HandleVisibility', 'off');
        end
    end
end
if isfield(cfg, 'anchor_points') && ~isempty(cfg.anchor_points)
    anchor_points = cfg.anchor_points;
    n_anchor_curves = min(size(anchor_points, 2), max_curves);
    for idx = 1:n_anchor_curves
        anchor_curve = squeeze(anchor_points(:, idx, :));
        anchor_curve = anchor_curve(:, 1:2);
        if idx == 1
            plot(anchor_curve(:, 1), anchor_curve(:, 2), 'kd', ...
                'MarkerFaceColor', 'none', 'MarkerSize', 7, ...
                'LineWidth', 1.2, 'LineStyle', 'none', ...
                'DisplayName', 'First-level anchor points');
        else
            plot(anchor_curve(:, 1), anchor_curve(:, 2), 'kd', ...
                'MarkerFaceColor', 'none', 'MarkerSize', 6, ...
                'LineWidth', 1.0, 'LineStyle', 'none', ...
                'HandleVisibility', 'off');
        end
    end
end
if isfield(cfg, 'anchor_target_points') && ~isempty(cfg.anchor_target_points)
    anchor_target_points = cfg.anchor_target_points;
    n_anchor_target_curves = min(size(anchor_target_points, 2), max_curves);
    for idx = 1:n_anchor_target_curves
        anchor_target_curve = squeeze(anchor_target_points(:, idx, :));
        anchor_target_curve = anchor_target_curve(:, 1:2);
        if idx == 1
            plot(anchor_target_curve(:, 1), anchor_target_curve(:, 2), 'ks', ...
                'MarkerFaceColor', 'none', 'MarkerSize', 7, ...
                'LineWidth', 1.2, 'LineStyle', 'none', ...
                'DisplayName', 'First-level rollout anchors');
        else
            plot(anchor_target_curve(:, 1), anchor_target_curve(:, 2), 'ks', ...
                'MarkerFaceColor', 'none', 'MarkerSize', 6, ...
                'LineWidth', 1.0, 'LineStyle', 'none', ...
                'HandleVisibility', 'off');
        end
    end
end
grid on;
axis equal;
xlim(x_limits);
ylim(y_limits);
xlabel('x');
ylabel('y');
if use_segment_plot
    title(sprintf('ODE Rollout Trajectories (%d Segments x %d Points, %d Samples)', ...
        cfg.segment_plot_count, cfg.segment_plot_points_per_segment, ...
        max_curves));
else
    title(sprintf('ODE Rollout Trajectories (%d Points, %d Samples)', ...
        size(reconstructed_points, 1), max_curves));
end
lgd = legend('Location', 'northeastoutside');
lgd.FontSize = 8;
lgd.ItemTokenSize = [14, 8];

if cfg.output.enabled
    export_graphics_compat(fig, output_path);
end
end
