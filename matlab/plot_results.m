% 绘制三列摘要图：目标轨迹、source 轨迹、roll out 轨迹
function plot_results(cfg, target_points, source_points, reconstructed_points)
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
if cfg.output.enabled && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
if strcmpi(struct_field_default(cfg, 'scenario', ''), 'racing') && ...
        struct_field_default(cfg, 'enable_third_level', false)
    output_filename = 'Racing_ThirdLevel_ThreePanel.emf';
else
    output_filename = 'gp_flow_matching_demo_matlab.emf';
end
output_path = fullfile(output_dir, output_filename);

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
% 赛车场景下把赛道边界也纳入范围计算：走廊在弯道外侧比轨迹更宽，只按轨迹
% 定范围会把赛道边缘切掉。
if isfield(cfg, 'track_segment') && ~isempty(cfg.track_segment)
    comparison_xy = [comparison_xy; cfg.track_segment.left; ...
        cfg.track_segment.right];
end
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
if isfield(cfg, 'track_segment')
    draw_track_segment(cfg.track_segment, 'HandleVisibility', 'off');
end
if isfield(cfg, 'obstacle')
    draw_obstacles(cfg.obstacle, 'HandleVisibility', 'off');
end
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
if isfield(cfg, 'track_segment')
    rollout_track = cfg.track_segment;
    % Keep the corridor and its boundaries in the rollout panel, but omit
    % the reference raceline so it cannot be confused with generated paths.
    rollout_track.raceline(:) = nan;
    draw_track_segment(rollout_track, 'HandleVisibility', 'off');
end
if isfield(cfg, 'obstacle')
    draw_obstacles(cfg.obstacle, 'HandleVisibility', 'off');
end
if use_segment_plot
    max_curves = min(n_segment_samples, cfg.first_level_generation_samples);
    total_rollout_curves = n_segment_samples;
else
    max_curves = min(size(reconstructed_points, 2), ...
        cfg.first_level_generation_samples);
    total_rollout_curves = size(reconstructed_points, 2);
end
% Report geometry violations independently of the display filter. Shared
% segment endpoints are counted once, so this is the number of distinct
% generated points represented by the plotted trajectories.
generated_xy = collect_generated_xy(reconstructed_points, use_segment_plot, ...
    segment_data, n_segments, n_points, feature_dim, max_curves);
report_generated_point_geometry(cfg, generated_xy);
% 只展示"整条 65 点折线都不穿障碍"的轨迹。判据用 cfg.obstacle 的物理几何
% (和 draw_obstacles 画的是同一套)，并在每条弦上稠密采样，所以点安全但弦
% 穿过去的那些会被剔除。关掉时展示全部，保持原行为。
curve_ok = true(1, max_curves);
obstacle_filter_mode = lower(string(struct_field_default(cfg, ...
    'obstacle_filter_mode', 'polyline')));
if struct_field_default(cfg.output, 'plot_only_obstacle_free_curves', false) ...
        && isfield(cfg, 'obstacle') && use_segment_plot
    curve_ok = obstacle_free_curves(segment_data, n_segments, n_points, ...
        feature_dim, max_curves, cfg.obstacle, obstacle_filter_mode);
end
kept_curves = find(curve_ok);
if isempty(kept_curves)
    warning('plot_results:NoObstacleFreeCurves', ...
        ['No rollout curve is obstacle-free; plotting all %d so the panel ', ...
        'is not empty.'], max_curves);
    kept_curves = 1:max_curves;
end
fprintf('Plotting rollout sample curves: %d / %d', ...
    numel(kept_curves), total_rollout_curves);
if numel(kept_curves) < max_curves
    if obstacle_filter_mode == "points"
        fprintf('  (%d of %d hidden: generated point inside an obstacle)', ...
            max_curves - numel(kept_curves), max_curves);
    else
        fprintf('  (%d of %d hidden: polyline crosses an obstacle)', ...
            max_curves - numel(kept_curves), max_curves);
    end
end
fprintf('\n');
% Colour stays tied to the trajectory index, so hiding curves never repaints
% the survivors.
rollout_colors = lines(max(max_curves, 1));
if use_segment_plot
    curve_label = struct_field_default(cfg, 'sample_curve_label', 'Stage 2 sample curves');
    marker_label = struct_field_default(cfg, 'generated_point_label', 'Stage 2 generated points');
    show_generated_points = struct_field_default(cfg, 'show_generated_points', true);
    for kept_idx = 1:numel(kept_curves)
        traj_idx = kept_curves(kept_idx);
        for segment_idx = 1:n_segments
            sample_idx = (traj_idx - 1) * n_segments + segment_idx;
            if sample_idx > size(segment_data, 1)
                continue;
            end
            segment_curve = reshape(segment_data(sample_idx, :), feature_dim, [])';
            segment_curve = segment_curve(:, 1:2);
            color = rollout_colors(traj_idx, :);
            if kept_idx == 1 && segment_idx == 1
                plot(segment_curve(:, 1), segment_curve(:, 2), ...
                    'Color', color, 'LineWidth', 1.1, 'DisplayName', curve_label);
                if show_generated_points
                    plot(segment_curve(:, 1), segment_curve(:, 2), 'o', ...
                        'Color', color, 'MarkerFaceColor', color, ...
                        'MarkerSize', 4, 'LineStyle', 'none', 'DisplayName', marker_label);
                end
            else
                plot(segment_curve(:, 1), segment_curve(:, 2), ...
                    'Color', color, 'LineWidth', 1.1, 'HandleVisibility', 'off');
                if show_generated_points
                    plot(segment_curve(:, 1), segment_curve(:, 2), 'o', ...
                        'Color', color, 'MarkerFaceColor', color, ...
                        'MarkerSize', 4, 'LineStyle', 'none', 'HandleVisibility', 'off');
                end
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
show_anchor_points = struct_field_default(cfg, 'show_anchor_points', true);
if show_anchor_points && isfield(cfg, 'anchor_points') && ~isempty(cfg.anchor_points)
    anchor_points = cfg.anchor_points;
    n_anchor_curves = min(size(anchor_points, 2), max_curves);
    anchor_idx_list = kept_curves(kept_curves <= n_anchor_curves);
    for k = 1:numel(anchor_idx_list)
        idx = anchor_idx_list(k);
        anchor_curve = squeeze(anchor_points(:, idx, :));
        anchor_curve = anchor_curve(:, 1:2);
        color = rollout_colors(idx, :);
        if k == 1
            plot(anchor_curve(:, 1), anchor_curve(:, 2), 'd', ...
                'Color', color, 'MarkerFaceColor', 'none', 'MarkerSize', 7, ...
                'LineWidth', 1.2, 'LineStyle', 'none', ...
                'DisplayName', struct_field_default(cfg, ...
                'anchor_points_label', 'First-level anchor points'));
        else
            plot(anchor_curve(:, 1), anchor_curve(:, 2), 'd', ...
                'Color', color, 'MarkerFaceColor', 'none', 'MarkerSize', 6, ...
                'LineWidth', 1.0, 'LineStyle', 'none', ...
                'HandleVisibility', 'off');
        end
    end
end
show_anchor_target_points = struct_field_default(cfg, 'show_anchor_target_points', true);
if show_anchor_target_points && isfield(cfg, 'anchor_target_points') && ...
        ~isempty(cfg.anchor_target_points)
    anchor_target_points = cfg.anchor_target_points;
    n_anchor_target_curves = min(size(anchor_target_points, 2), max_curves);
    target_idx_list = kept_curves(kept_curves <= n_anchor_target_curves);
    for k = 1:numel(target_idx_list)
        idx = target_idx_list(k);
        anchor_target_curve = squeeze(anchor_target_points(:, idx, :));
        anchor_target_curve = anchor_target_curve(:, 1:2);
        color = rollout_colors(idx, :);
        if k == 1
            plot(anchor_target_curve(:, 1), anchor_target_curve(:, 2), 's', ...
                'Color', color, 'MarkerFaceColor', 'none', 'MarkerSize', 7, ...
                'LineWidth', 1.2, 'LineStyle', 'none', ...
                'DisplayName', struct_field_default(cfg, ...
                'anchor_target_points_label', 'First-level rollout anchors'));
        else
            plot(anchor_target_curve(:, 1), anchor_target_curve(:, 2), 's', ...
                'Color', color, 'MarkerFaceColor', 'none', 'MarkerSize', 6, ...
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
    if numel(kept_curves) < max_curves
        title(sprintf('ODE Rollout Trajectories (%d Segments x %d Points)', ...
            cfg.segment_plot_count, ...
            cfg.segment_plot_points_per_segment));
    else
        title(sprintf('ODE Rollout Trajectories (%d Segments x %d Points, %d Samples)', ...
            cfg.segment_plot_count, cfg.segment_plot_points_per_segment, ...
            max_curves));
    end
else
    title(sprintf('ODE Rollout Trajectories (%d Points, %d Samples)', ...
        size(reconstructed_points, 1), max_curves));
end
% 图例放在坐标区下方而不是右侧：northeastoutside 会把整个 tiledlayout 往左挤，
% 三栏宽度变得不一致。
lgd = legend('Location', 'southoutside');
lgd.FontSize = 8;
lgd.ItemTokenSize = [14, 8];

if cfg.output.enabled
    export_graphics_compat(fig, output_path);
end
end

function is_free = obstacle_free_curves(segment_data, n_segments, n_points, ...
    feature_dim, n_curves, obstacle, filter_mode)
%OBSTACLE_FREE_CURVES Apply the safety test appropriate to the plotted level.
% Segments share endpoints, so the trajectory is n_segments*(n_points-1)+1
% distinct points. Stage 2 checks those generated points only. Stage 3 samples
% every chord densely and also rejects the "safe endpoints, unsafe chord" case.
is_free = true(1, n_curves);
if ~struct_field_default(obstacle, 'enabled', false) || ...
        ~isfield(obstacle, 'centers') || isempty(obstacle.centers)
    return;
end
centers = obstacle.centers;
semi_axes = obstacle.semi_axes;
if size(centers, 1) ~= 2
    centers = centers';
end
if size(semi_axes, 1) ~= 2
    semi_axes = semi_axes';
end
n_obstacles = size(centers, 2);
lambda = linspace(0, 1, 101);
for curve_idx = 1:n_curves
    polyline = zeros(n_segments * (n_points - 1) + 1, 2);
    for segment_idx = 1:n_segments
        sample_idx = (curve_idx - 1) * n_segments + segment_idx;
        if sample_idx > size(segment_data, 1)
            continue;
        end
        curve = reshape(segment_data(sample_idx, :), feature_dim, [])';
        first_row = (segment_idx - 1) * (n_points - 1) + 1;
        polyline(first_row:first_row + n_points - 1, :) = curve(:, 1:2);
    end
    if filter_mode == "points"
        test_sets = {polyline'};
    elseif filter_mode == "polyline"
        test_sets = cell(1, size(polyline, 1) - 1);
        for point_idx = 1:size(polyline, 1) - 1
            test_sets{point_idx} = polyline(point_idx, :)' .* (1 - lambda) + ...
                polyline(point_idx + 1, :)' .* lambda;
        end
    else
        error('obstacle_filter_mode must be ''points'' or ''polyline''.');
    end
    for test_idx = 1:numel(test_sets)
        dense = test_sets{test_idx};
        for obstacle_idx = 1:n_obstacles
            angle = geometry_scalar(obstacle, 'angles', obstacle_idx, 0.0);
            exponent = geometry_scalar(obstacle, 'exponents', obstacle_idx, 2.0);
            rotation = [cos(angle), -sin(angle); sin(angle), cos(angle)];
            scaled = (rotation' * (dense - centers(:, obstacle_idx))) ./ ...
                semi_axes(:, obstacle_idx);
            power_sum = sum(abs(scaled) .^ exponent, 1);
            if exponent > 2
                h_now = power_sum .^ (1 / exponent) - 1.0;
            else
                h_now = power_sum - 1.0;
            end
            if any(h_now < 0)
                is_free(curve_idx) = false;
                break;
            end
        end
        if ~is_free(curve_idx)
            break;
        end
    end
end
end

function value = geometry_scalar(geometry, field_name, obstacle_idx, default_value)
if ~isfield(geometry, field_name) || isempty(geometry.(field_name))
    value = default_value;
elseif isscalar(geometry.(field_name))
    value = geometry.(field_name);
else
    value = geometry.(field_name)(obstacle_idx);
end
end

function xy = collect_generated_xy(reconstructed_points, use_segment_plot, ...
    segment_data, n_segments, n_points, feature_dim, n_curves)
if ~use_segment_plot
    if isempty(reconstructed_points) || n_curves == 0
        xy = zeros(0, 2);
    else
        xy = reshape(reconstructed_points(:, 1:n_curves, 1:2), [], 2);
    end
    return;
end

n_distinct_points = n_segments * (n_points - 1) + 1;
xy = nan(n_curves * n_distinct_points, 2);
for curve_idx = 1:n_curves
    polyline = nan(n_distinct_points, 2);
    for segment_idx = 1:n_segments
        sample_idx = (curve_idx - 1) * n_segments + segment_idx;
        if sample_idx > size(segment_data, 1)
            continue;
        end
        curve = reshape(segment_data(sample_idx, :), feature_dim, [])';
        first_row = (segment_idx - 1) * (n_points - 1) + 1;
        polyline(first_row:first_row + n_points - 1, :) = curve(:, 1:2);
    end
    rows = (curve_idx - 1) * n_distinct_points + (1:n_distinct_points);
    xy(rows, :) = polyline;
end
end

function report_generated_point_geometry(cfg, xy)
n_total = size(xy, 1);
finite_mask = all(isfinite(xy), 2);
inside_obstacle = false(n_total, 1);
outside_boundary = false(n_total, 1);

if isfield(cfg, 'obstacle') && ...
        struct_field_default(cfg.obstacle, 'enabled', false) && ...
        isfield(cfg.obstacle, 'centers')
    for point_idx = find(finite_mask)'
        p = xy(point_idx, :)';
        for obstacle_idx = 1:size(cfg.obstacle.centers, 2)
            h = obstacle_level_and_gradient(p, cfg.obstacle, obstacle_idx);
            if h < 0
                inside_obstacle(point_idx) = true;
                break;
            end
        end
    end
end

has_boundary_geometry = isfield(cfg, 'track_boundary') && ...
    struct_field_default(cfg.track_boundary, 'enabled', false) && ...
    isfield(cfg.track_boundary, 'geometry') && ...
    isfield(cfg.track_boundary.geometry, 'implicit_fields');
if has_boundary_geometry
    fields = cfg.track_boundary.geometry.implicit_fields;
    for point_idx = find(finite_mask)'
        p = xy(point_idx, :)';
        h_left = evaluate_track_implicit_field(fields, 1, p);
        h_right = evaluate_track_implicit_field(fields, 2, p);
        outside_boundary(point_idx) = h_left < 0 || h_right < 0;
    end
end

either_violation = inside_obstacle | outside_boundary;
fprintf(['Generated-point geometry: %d distinct points; ', ...
    'inside physical obstacle %d; '], n_total, nnz(inside_obstacle));
if has_boundary_geometry
    fprintf('outside physical track boundary %d; either %d', ...
        nnz(outside_boundary), nnz(either_violation));
else
    fprintf('track-boundary count unavailable');
end
if nnz(~finite_mask) > 0
    fprintf('; nonfinite %d', nnz(~finite_mask));
end
fprintf('.\n');
end
