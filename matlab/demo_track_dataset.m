%% demo_track_dataset
% 赛道段训练数据集的生成与可视化脚本。
% 本脚本用于在正式生成数据集前检查：赛段截取是否正确、gate 是否合理、
% 轨迹是否位于走廊内、稀疏采样折线是否平滑，以及横向偏移分布是否过于贴边。
% 直接运行只生成内存变量和图窗，不写文件；需要保存时调用 build_track_dataset。
clc; clear; close all;

%% Settings
s_range_m = [250, 1150];   % Mercedes-Arena + 发夹弯，弯序列 右-左-右
n_points = 65;             % 与 build_track_dataset 默认一致（三层阶梯 5/17/65）
n_trajectories = 200;      % 预览用，正式数据集用 cfg.n_train = 30
n_preview_polyline = 30;   % 单独画出来看折线形状的轨迹条数
% 显式传给生成器，保证 Figure 4 画的筛选阈值和实际拒绝采样用的是同一个值。
corridor_inset_ratio = 0.02;

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
    struct('corridor_inset_ratio', corridor_inset_ratio));
elapsed = toc;

fprintf('generator    : %d gates, %d x %d trajectories in %.1f s\n', ...
    numel(gates.radius), n_points, n_trajectories, elapsed);
fprintf('flattened dim: %d  (%d points x %d features)\n', ...
    n_points * size(points, 3), n_points, size(points, 3));
fprintf('tangent check: max |1 - |t|| = %.2e\n', ...
    max(abs(1 - sqrt(points(:, :, 3) .^ 2 + points(:, :, 4) .^ 2)), [], 'all'));

%% Figure 1: Segment Geometry And Gates
figure('Name', 'segment geometry', 'Color', 'w'); hold on;
draw_corridor(segment);
plot(segment.center(:, 1), segment.center(:, 2), 'k--', 'LineWidth', 0.5);
plot(segment.raceline(:, 1), segment.raceline(:, 2), 'r-', 'LineWidth', 1.5);
theta = linspace(0, 2 * pi, 60);
for gate_nr = 1:numel(gates.radius)
    plot(gates.center(1, gate_nr) + gates.radius(gate_nr) * cos(theta), ...
        gates.center(2, gate_nr) + gates.radius(gate_nr) * sin(theta), ...
        'm-', 'LineWidth', 1);
end
quiver(gates.center(1, :), gates.center(2, :), ...
    0.03 * cos(gates.heading), 0.03 * sin(gates.heading), 0, ...
    'Color', [0.4 0 0.6], 'LineWidth', 1);
axis equal; grid on;
title(sprintf('%d control-point gates (magenta) + heading range', ...
    numel(gates.radius)));
legend({'corridor', 'boundary', 'boundary', 'centerline', 'raceline'}, ...
    'Location', 'best');

%% Figure 2: Trajectory Family
figure('Name', 'trajectory family', 'Color', 'w'); hold on;
draw_corridor(segment);
for trajectory_nr = 1:n_trajectories
    plot(points(:, trajectory_nr, 1), points(:, trajectory_nr, 2), '-', ...
        'Color', [0 0.45 0.85 0.12]);
end
plot(segment.raceline(:, 1), segment.raceline(:, 2), 'r-', 'LineWidth', 1.5);
axis equal; grid on;
title(sprintf('%d trajectories, n\\_points = %d', n_trajectories, n_points));

%% Figure 3: Sample-Point Polyline (这才是喂给 FM 的东西)
figure('Name', 'sample points', 'Color', 'w'); hold on;
draw_corridor(segment);
for trajectory_nr = 1:min(n_preview_polyline, n_trajectories)
    plot(points(:, trajectory_nr, 1), points(:, trajectory_nr, 2), '.-', ...
        'Color', [0 0.45 0.85 0.5], 'MarkerSize', 8);
end
plot(segment.raceline(:, 1), segment.raceline(:, 2), 'r-', 'LineWidth', 1.5);
axis equal; grid on;
title(sprintf('sample-point polyline, %d trajectories', ...
    min(n_preview_polyline, n_trajectories)));

%% Figure 4: Corridor Occupancy Ratio
% 不画绝对横向偏移，因为每个轨迹点的最近中心线点不同，50 条轨迹没有共同的
% 边界曲线可画（按点序号均匀映射到中心线是错的：轨迹按自身弧长采样，弯道
% 切内侧时和中心线弧长不等价）。改画归一化占用率
%   rho = e_lat / w_L  (e_lat >= 0),   rho = e_lat / w_R  (e_lat < 0)
% 这样 rho = ±1 就是左右边界、0 是中心线，所有轨迹共享同一对边界线。
[lateral, nearest_idx] = signed_lateral_offset(segment, points);
occupancy = corridor_occupancy_ratio(segment, lateral, nearest_idx);

figure('Name', 'corridor occupancy', 'Color', 'w');
subplot(2, 1, 1);
plot(1:n_points, occupancy(:, 1:min(50, n_trajectories)), '-', ...
    'Color', [0 0.45 0.85 0.3]); hold on;
yline(1, 'k-', 'LineWidth', 1.2);
yline(-1, 'k-', 'LineWidth', 1.2);
yline(1 - corridor_inset_ratio, 'r--', 'LineWidth', 1);
yline(-(1 - corridor_inset_ratio), 'r--', 'LineWidth', 1);
grid on; ylim([-1.1, 1.1]);
xlabel('point index'); ylabel('occupancy ratio \rho');
title(sprintf(['corridor occupancy (\\pm1 = boundary, red = rejection ' ...
    'threshold \\pm%.2f)'], 1 - corridor_inset_ratio));

subplot(2, 1, 2);
histogram(occupancy(:), 60); grid on; hold on;
xline(1, 'k-', 'LineWidth', 1.2);
xline(-1, 'k-', 'LineWidth', 1.2);
xlabel('occupancy ratio \rho'); ylabel('count');
title('occupancy distribution over all points');

fprintf('occupancy    : max |rho| = %.3f  (rejection threshold %.2f)\n', ...
    max(abs(occupancy), [], 'all'), 1 - corridor_inset_ratio);

%% ------------------------------------------------------------------------
function draw_corridor(segment)
%DRAW_CORRIDOR 绘制灰色可行走廊及黑色左右边界。
% fill 的顶点顺序为“左边界正向 + 右边界反向”，从而形成闭合多边形。
fill([segment.left(:, 1); flipud(segment.right(:, 1))], ...
    [segment.left(:, 2); flipud(segment.right(:, 2))], [0.90 0.90 0.90], ...
    'EdgeColor', 'none');
plot(segment.left(:, 1), segment.left(:, 2), 'k-', 'LineWidth', 1);
plot(segment.right(:, 1), segment.right(:, 2), 'k-', 'LineWidth', 1);
end

function [lateral, nearest_idx] = signed_lateral_offset(segment, points)
%SIGNED_LATERAL_OFFSET 计算轨迹点相对最近中心线的带符号横向偏移。
% 输入 points 的布局为 n_points × n_trajectories × feature_dim，仅使用前两维
% 位置；输出 lateral 为 n_points × n_trajectories，左侧为正、右侧为负。
% nearest_idx 是同尺寸的中心线索引，调用方据此取该点真正对应的左右半宽——
% 每条轨迹的对应关系都不同，不能按点序号均匀映射。
center = segment.center;
normal_left = [-segment.tangent(:, 2), segment.tangent(:, 1)];
[n_points, n_trajectories, ~] = size(points);
lateral = zeros(n_points, n_trajectories);
nearest_idx = zeros(n_points, n_trajectories);
for trajectory_nr = 1:n_trajectories
    query = [points(:, trajectory_nr, 1), points(:, trajectory_nr, 2)];
    d2 = sum((permute(query, [1, 3, 2]) - permute(center, [3, 1, 2])) .^ 2, 3);
    [~, idx] = min(d2, [], 2);
    nearest_idx(:, trajectory_nr) = idx;
    offset = query - center(idx, :);
    lateral(:, trajectory_nr) = sum(offset .* normal_left(idx, :), 2);
end
end

function occupancy = corridor_occupancy_ratio(segment, lateral, nearest_idx)
%CORRIDOR_OCCUPANCY_RATIO 把带符号横向偏移归一化成走廊占用率。
% 左偏除以该点的左半宽、右偏除以右半宽，于是 ±1 恒为左右边界、0 为中心线，
% 与具体在赛道哪一段无关，所有轨迹可以共用同一对边界线。
half_width_left = segment.half_width_left(nearest_idx);
half_width_right = segment.half_width_right(nearest_idx);
occupancy = lateral ./ max(half_width_left, eps);
right_side = lateral < 0;
occupancy(right_side) = lateral(right_side) ./ ...
    max(half_width_right(right_side), eps);
end
