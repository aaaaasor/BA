%% demo_track_dataset
% 生成并预览赛车训练轨迹。默认只出图窗；save_figures = true 时把四张图
% 存到 outputs/ 下（文件名 track_dataset_fig1..4）。
clc; clear; close all;

%% Settings
s_range_m = [250, 1150];
n_points = 65;
n_trajectories = 200;
n_preview_polyline = 30;   % 单独画出来看折线形状的轨迹条数
% 生成参数。
corridor_inset_ratio = 0.02;
gate_spacing_m = 20;
% 图片保存。EMF 矢量图，和仓库其它输出一致。
save_figures = true;

this_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(this_dir, 'trajectory_data');
track_csv = fullfile(data_dir, 'Nuerburgring.csv');
raceline_csv = fullfile(data_dir, 'Nuerburgring_raceline.csv');

%% Extract Segment
segment = extract_track_segment(track_csv, raceline_csv, ...
    struct('s_range_m', s_range_m));

fprintf('segment      : s = %.0f .. %.0f m  (%.0f m)\n', ...
    segment.meta.s_range_m(1), segment.meta.s_range_m(2), segment.meta.length_m);
fprintf('corridor     : half width %.4f .. %.4f (normalized) = %.1f .. %.1f m\n', ...
    min(segment.half_width_left), max(segment.half_width_left), ...
    min(segment.half_width_left) / segment.transform.scale, ...
    max(segment.half_width_left) / segment.transform.scale);

%% Generate Trajectories
tic;
[points, gates] = generate_track_training_points_2d(n_points, ...
    n_trajectories, segment, ...
    struct('corridor_inset_ratio', corridor_inset_ratio, ...
    'gate_spacing_m', gate_spacing_m));
elapsed = toc;

% 图中显示米制长度。
to_m = 1.0 / segment.transform.scale;

fprintf('generator    : %d gates, %d x %d trajectories in %.1f s\n', ...
    numel(gates.s), n_points, n_trajectories, elapsed);
fprintf('flattened dim: %d  (%d points x %d features)\n', ...
    n_points * size(points, 3), n_points, size(points, 3));
fprintf('tangent check: max |1 - |t|| = %.2e\n', ...
    max(abs(1 - sqrt(points(:, :, 3) .^ 2 + points(:, :, 4) .^ 2)), [], 'all'));

%% Figure 1: Segment Geometry And Gates
figure('Name', 'segment geometry', 'Color', 'w'); hold on;
[h_fill, h_boundary] = draw_corridor(segment);
h_center = plot(segment.center(:, 1), segment.center(:, 2), 'k--', ...
    'LineWidth', 0.5);
h_raceline = plot(segment.raceline(:, 1), segment.raceline(:, 2), 'r-', ...
    'LineWidth', 1.5);
% 绘制 gate 横线。
gate_left = gates.center + gates.normal_left .* gates.extent_left;
gate_right = gates.center - gates.normal_left .* gates.extent_right;
h_gate = gobjects(1);
for gate_nr = 1:numel(gates.s)
    h = plot([gate_left(1, gate_nr), gate_right(1, gate_nr)], ...
        [gate_left(2, gate_nr), gate_right(2, gate_nr)], 'm-', 'LineWidth', 1);
    if gate_nr == 1
        h_gate = h;
    end
end
% 绘制 gate 上的赛车线位置。
rho = gates.rho_raceline;
center_lateral = rho .* gates.extent_left;
right_side = rho < 0;
center_lateral(right_side) = rho(right_side) .* gates.extent_right(right_side);
gate_mid = gates.center + gates.normal_left .* center_lateral;
h_heading = plot(gate_mid(1, :), gate_mid(2, :), 'o', 'MarkerSize', 3, ...
    'MarkerEdgeColor', [0.4 0 0.6], 'MarkerFaceColor', [0.4 0 0.6]);
axis equal; grid on;
xlim([0, 1]); ylim([0, 1]);
title(sprintf('%d perpendicular gates, spacing %g m along centerline', ...
    numel(gates.s), gate_spacing_m));
legend([h_fill, h_boundary, h_center, h_raceline, h_gate, h_heading], ...
    {'corridor', 'track boundary', 'centerline', 'raceline', ...
    sprintf('gate span (%.1f .. %.1f m across)', ...
    min(gates.extent_left + gates.extent_right) * to_m, ...
    max(gates.extent_left + gates.extent_right) * to_m), ...
    'sampling centre (raceline)'}, ...
    'Location', 'southoutside', 'NumColumns', 3);

%% Figure 2: Trajectory Family
figure('Name', 'trajectory family', 'Color', 'w'); hold on;
[h_fill, h_boundary] = draw_corridor(segment);
h_traj = gobjects(1);
for trajectory_nr = 1:n_trajectories
    h = plot(points(:, trajectory_nr, 1), points(:, trajectory_nr, 2), '-', ...
        'Color', [0 0.45 0.85 0.12]);
    if trajectory_nr == 1
        h_traj = h;
    end
end
h_raceline = plot(segment.raceline(:, 1), segment.raceline(:, 2), 'r-', ...
    'LineWidth', 1.5);
axis equal; grid on;
xlim([0, 1]); ylim([0, 1]);
title(sprintf('%d trajectories, segment %.0f m', n_trajectories, ...
    segment.meta.length_m));
legend([h_fill, h_boundary, h_traj, h_raceline], ...
    {sprintf('corridor (half width %.1f .. %.1f m)', ...
    min(segment.half_width_left) * to_m, max(segment.half_width_left) * to_m), ...
    'track boundary', ...
    sprintf('trajectory (%d shown)', n_trajectories), 'raceline'}, ...
    'Location', 'southoutside', 'NumColumns', 2);

%% Figure 3: Sample-Point Polyline
figure('Name', 'sample points', 'Color', 'w'); hold on;
[h_fill, h_boundary] = draw_corridor(segment);
n_shown = min(n_preview_polyline, n_trajectories);
h_poly = gobjects(1);
for trajectory_nr = 1:n_shown
    h = plot(points(:, trajectory_nr, 1), points(:, trajectory_nr, 2), '.-', ...
        'Color', [0 0.45 0.85 0.5], 'MarkerSize', 8);
    if trajectory_nr == 1
        h_poly = h;
    end
end
h_raceline = plot(segment.raceline(:, 1), segment.raceline(:, 2), 'r-', ...
    'LineWidth', 1.5);
axis equal; grid on;
xlim([0, 1]); ylim([0, 1]);
% 模型使用的离散轨迹点。
title(sprintf('sample-point polyline, %d points per trajectory', n_points));
legend([h_fill, h_boundary, h_poly, h_raceline], ...
    {'corridor', 'track boundary', ...
    sprintf('%d sample points, %d trajectories shown', n_points, n_shown), ...
    'raceline'}, 'Location', 'southoutside', 'NumColumns', 2);

%% Figure 4: Corridor Occupancy Ratio
% rho=0 为中心线，rho=±1 为左右边界。
[lateral, nearest_idx] = signed_lateral_offset(segment, points);
occupancy = corridor_occupancy_ratio(segment, lateral, nearest_idx);

figure('Name', 'corridor occupancy', 'Color', 'w');
n_curves = min(50, n_trajectories);
subplot(2, 1, 1);
h_rho = plot(1:n_points, occupancy(:, 1:n_curves), '-', ...
    'Color', [0 0.45 0.85 0.3]); hold on;
h_edge = yline(1, 'k-', 'LineWidth', 1.2);
yline(-1, 'k-', 'LineWidth', 1.2);
h_thr = yline(1 - corridor_inset_ratio, 'r--', 'LineWidth', 1);
yline(-(1 - corridor_inset_ratio), 'r--', 'LineWidth', 1);
grid on; ylim([-1.1, 1.1]);
xlabel('point index'); ylabel('occupancy ratio \rho');
title('corridor occupancy');
legend([h_rho(1), h_edge, h_thr], ...
    {sprintf('\\rho per trajectory (%d shown, >0 = left of centerline)', ...
    n_curves), 'track boundary \pm1', ...
    sprintf('rejection threshold \\pm%.2f', 1 - corridor_inset_ratio)}, ...
    'Location', 'southoutside', 'NumColumns', 3);

subplot(2, 1, 2);
h_hist = histogram(occupancy(:), 60); grid on; hold on;
h_edge = xline(1, 'k-', 'LineWidth', 1.2);
xline(-1, 'k-', 'LineWidth', 1.2);
xlabel('occupancy ratio \rho'); ylabel('count');
title('occupancy distribution over all points');
legend([h_hist, h_edge], ...
    {sprintf('%d points (%d trajectories x %d)', numel(occupancy), ...
    n_trajectories, n_points), 'track boundary \pm1'}, ...
    'Location', 'southoutside', 'NumColumns', 2);

fprintf('occupancy    : max |rho| = %.3f  (rejection threshold %.2f)\n', ...
    max(abs(occupancy), [], 'all'), 1 - corridor_inset_ratio);

%% Save Figures
% 用仓库统一的 export_graphics_compat（EMF 多重 fallback，并且先写
% 临时文件再替换，避免 OneDrive 目录下覆盖已有文件失败）。
if save_figures
    output_dir = fullfile(this_dir, 'outputs');
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    figure_names = {'segment geometry', 'trajectory family', ...
        'sample points', 'corridor occupancy'};
    for figure_nr = 1:numel(figure_names)
        fig = findobj('Type', 'figure', 'Name', figure_names{figure_nr});
        if isempty(fig)
            continue;
        end
        output_path = fullfile(output_dir, ...
            sprintf('track_dataset_fig%d.emf', figure_nr));
        export_graphics_compat(fig(1), output_path);
        disp(['saved ', output_path]);
    end
end

%% ------------------------------------------------------------------------
function [h_fill, h_boundary] = draw_corridor(segment)
% 绘制赛道走廊和边界。
h_fill = fill([segment.left(:, 1); flipud(segment.right(:, 1))], ...
    [segment.left(:, 2); flipud(segment.right(:, 2))], [0.90 0.90 0.90], ...
    'EdgeColor', 'none');
h_boundary = plot(segment.left(:, 1), segment.left(:, 2), 'k-', 'LineWidth', 1);
plot(segment.right(:, 1), segment.right(:, 2), 'k-', 'LineWidth', 1);
end

function [lateral, nearest_idx] = signed_lateral_offset(segment, points)
% 计算每条轨迹的横向偏移。
[n_points, n_trajectories, ~] = size(points);
lateral = zeros(n_points, n_trajectories);
nearest_idx = zeros(n_points, n_trajectories);
for trajectory_nr = 1:n_trajectories
    [lateral(:, trajectory_nr), nearest_idx(:, trajectory_nr)] = ...
        track_lateral_offset(segment, ...
        [points(:, trajectory_nr, 1), points(:, trajectory_nr, 2)]);
end
end

function occupancy = corridor_occupancy_ratio(segment, lateral, nearest_idx)
% 将横向偏移归一化为走廊占用率。
half_width_left = segment.half_width_left(nearest_idx);
half_width_right = segment.half_width_right(nearest_idx);
occupancy = lateral ./ max(half_width_left, eps);
right_side = lateral < 0;
occupancy(right_side) = lateral(right_side) ./ ...
    max(half_width_right(right_side), eps);
end
