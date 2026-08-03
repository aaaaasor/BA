%GENERATE_TRACK_TRAINING_POINTS_2D 在给定赛段内生成二维几何轨迹训练样本。
%
%   [points,gates] = GENERATE_TRACK_TRAINING_POINTS_2D(...
%       n_points, n_trajectories, segment, opts)
%
% 输入：
%   n_points       - 每条轨迹最终保留的等弧长采样点数。
%   n_trajectories - 需要接受的有效轨迹数量。
%   segment        - extract_track_segment 返回的归一化赛段结构体。
%   opts           - 可选生成参数结构体；未提供字段使用下方默认值。
%
% 输出：
%   points - n_points × n_trajectories × 4 数组，特征顺序为
%            [x, y, dx/ds, dy/ds]。前两维是位置，后两维是单位切向量。
%   gates  - 自动生成的控制点采样区域及参考航向。
%
% 与原始版本的区别：
%   1. control point gate 不再手写，而是由赛道段几何自动放置：弧长位置取
%      段起终点 + 各弯 apex + 等间距填充，圆心挂到最近的赛车线点上，朝向
%      取赛车线局部切向，半径按该点到最近边界的剩余余量截断；
%   2. Hermite 切向量按控制点间距缩放（原版固定单位长度，换到别的尺度会失真）；
%   3. 采样出来的样条做赛道走廊内的拒绝采样。
%
% 本函数只生成几何路径，不计算速度、转向角、加速度或车辆动力学一致性。
function [points, gates] = generate_track_training_points_2d(n_points, ...
    n_trajectories, segment, opts)
this_dir = fileparts(mfilename('fullpath'));
trajectory_data_dir = fullfile(this_dir, 'trajectory_data');
if exist(trajectory_data_dir, 'dir') && ~contains(path, trajectory_data_dir)
    addpath(trajectory_data_dir);
end
if nargin < 4 || isempty(opts)
    opts = struct();
end
% 控制点沿弧长的标称间距（米）。赛道是一条很细的带子（半宽 ~8 m），控制点
% 太稀时三次样条无论如何都待不住走廊，所以间距必须和赛道宽度同量级。
% gate 挂在赛车线上，而赛车线在弯心离边界最近只有 ~0.5 m，样条在 gate 之间
% 的外扩会直接顶出走廊，所以间距要比挂中心线时密一倍。
gate_spacing_m = struct_field_default(opts, 'gate_spacing_m', 20.0);
% gate 半径 = 该处赛道半宽 * 该系数
gate_radius_ratio = struct_field_default(opts, 'gate_radius_ratio', 0.45);
% gate 切向区间 = 赛道切向 ± 该角度
gate_heading_spread_deg = struct_field_default(opts, ...
    'gate_heading_spread_deg', 8.0);
% 入口/出口 gate 收紧（起终点应当比中间弯道更确定）
gate_radius_ratio_ends = struct_field_default(opts, ...
    'gate_radius_ratio_ends', 0.15);
gate_heading_spread_deg_ends = struct_field_default(opts, ...
    'gate_heading_spread_deg_ends', 4.0);
% 走廊拒绝采样时相对半宽留出的安全内缩比例
corridor_inset_ratio = struct_field_default(opts, 'corridor_inset_ratio', 0.02);
max_attempts_per_trajectory = struct_field_default(opts, ...
    'max_attempts_per_trajectory', 200);
n_dense = struct_field_default(opts, 'n_dense', 5000);
rng_seed = struct_field_default(opts, 'rng_seed', 0);

previous_rng_state = rng;
% onCleanup 保证函数退出（包括报错退出）后恢复调用者的随机数状态。
restore_rng_state = onCleanup(@() rng(previous_rng_state));
rng(rng_seed);

% gate 决定各个 Hermite 控制点允许出现的位置和方向范围。
gates = build_segment_gates(segment, gate_spacing_m, gate_radius_ratio, ...
    gate_heading_spread_deg, gate_radius_ratio_ends, ...
    gate_heading_spread_deg_ends);
p_quantity = numel(gates.radius);

feature_dim = 4;
points = nan(n_points, n_trajectories, feature_dim);
t_set = (0:(p_quantity - 1)) / (p_quantity - 1);
t_dense_set = linspace(0, 1, n_dense);
delta_t = t_set(2) - t_set(1);

for trajectory_nr = 1:n_trajectories
    accepted = false;
    % 一次 attempt 会重新采样该轨迹的全部控制点；任一密集点越界则整条拒绝。
    for attempt = 1:max_attempts_per_trajectory
        p_set = nan(2, p_quantity);
        heading_set = nan(1, p_quantity);
        for p_nr = 1:p_quantity
            % sqrt(rand) 使位置在圆盘面积上均匀，而不是沿半径均匀并偏向圆心。
            alpha_i = 2 * pi * rand(1);
            p_set(:, p_nr) = gates.radius(p_nr) * sqrt(rand(1)) * ...
                [cos(alpha_i); sin(alpha_i)] + gates.center(:, p_nr);
            heading_set(p_nr) = gates.heading(p_nr) + ...
                (rand(1) - 0.5) * gates.heading_spread(p_nr);
        end

        % Hermite 插值需要的是对全局参数 t 的导数。这里用相邻控制点距离除以
        % delta_t 作为导数模长，使曲线尺度变化时仍保持合理的弯曲程度。
        dp_set = [cos(heading_set); sin(heading_set)];
        dp_set = dp_set .* control_point_speed(p_set, delta_t);

        polynomial_coefficient_x = GPFM_DataGeneration_fit_Polynomial( ...
            t_set, p_set(1, :), dp_set(1, :));
        polynomial_coefficient_y = GPFM_DataGeneration_fit_Polynomial( ...
            t_set, p_set(2, :), dp_set(2, :));

        x_dense_set = ppval(polynomial_coefficient_x, t_dense_set);
        y_dense_set = ppval(polynomial_coefficient_y, t_dense_set);

        % 用密集曲线而不是最终稀疏点做约束检查，避免两个采样点之间穿出边界。
        if ~inside_corridor(segment, [x_dense_set(:), y_dense_set(:)], ...
                corridor_inset_ratio)
            continue;
        end

        ds_dense_set = sqrt((x_dense_set(2:end) - x_dense_set(1:(end - 1))) .^ 2 + ...
            (y_dense_set(2:end) - y_dense_set(1:(end - 1))) .^ 2);
        s_dense_set = [0, cumsum(ds_dense_set)];
        trajectory_length = s_dense_set(end);

        % 先在弧长轴上均匀取样，再反查多项式参数 t，得到近似等空间间距的点。
        s_set = linspace(0, trajectory_length, n_points);
        t_train_i = interp1(s_dense_set, t_dense_set, s_set);
        points(:, trajectory_nr, 1) = ppval(polynomial_coefficient_x, t_train_i);
        points(:, trajectory_nr, 2) = ppval(polynomial_coefficient_y, t_train_i);

        dx_dt_train_i = ppval(pp_derivative(polynomial_coefficient_x), t_train_i);
        dy_dt_train_i = ppval(pp_derivative(polynomial_coefficient_y), t_train_i);
        speed_train_i = max(sqrt(dx_dt_train_i .^ 2 + dy_dt_train_i .^ 2), eps);
        % 归一化参数导数得到对弧长的单位切向量；eps 防止退化点除零。
        points(:, trajectory_nr, 3) = dx_dt_train_i ./ speed_train_i;
        points(:, trajectory_nr, 4) = dy_dt_train_i ./ speed_train_i;

        accepted = true;
        break;
    end
    if ~accepted
        error('generate_track_training_points_2d:rejectionFailed', ...
            ['Trajectory %d not accepted after %d attempts. Shrink ' ...
            'gate_radius_ratio / gate_heading_spread_deg, or relax ' ...
            'corridor_inset_ratio.'], trajectory_nr, max_attempts_per_trajectory);
    end
end
end

%% ------------------------------------------------------------------------
function gates = build_segment_gates(segment, spacing_m, radius_ratio, ...
    spread_deg, radius_ratio_ends, spread_deg_ends)
%BUILD_SEGMENT_GATES 根据赛段几何自动构造 Hermite 控制点采样 gate。
% gate 的弧长位置由端点、弯道 apex 和等距填充点共同决定；位置中心随后从
% 中心线投影到最近赛车线点。半径由赛车线到较近边界的剩余空间限制，避免
% 控制点圆盘自身穿出走廊。端点使用更小半径和航向范围以稳定起终点分布。
% gate 沿弧长的位置：段起点/终点 + 每个弯的 apex + 等间距填充（apex 优先保留，
% 等间距点若离已有 gate 太近则丢弃）。位置定在中心线上算，但圆心最终挂到
% 赛车线上——训练分布要围绕赛车线，不是中心线。
center = segment.center;
n_center = size(center, 1);
s_center = segment.s_center(:);

kappa = discrete_curvature(center);
apex_idx = corner_apex_indices(kappa, s_center);

spacing = spacing_m * segment.transform.scale;   % 米 -> 归一化弧长
n_fill = max(round(s_center(end) / spacing), 1);
s_fill = linspace(0, s_center(end), n_fill + 1)';
fill_idx = arrayfun(@(s) nearest_scalar_index(s_center, s), s_fill);

anchor_idx = unique([1; apex_idx(:); n_center]);
min_gap = 0.5 * spacing;
keep_fill = true(size(fill_idx));
for k = 1:numel(fill_idx)
    if min(abs(s_center(fill_idx(k)) - s_center(anchor_idx))) < min_gap
        keep_fill(k) = false;
    end
end
gate_idx = sort(unique([anchor_idx; fill_idx(keep_fill)]));

% 把每个 gate 挂到最近的赛车线点上，朝向改用赛车线的局部切向。
raceline = segment.raceline;
raceline_tangent = gradient_rows(raceline);
raceline_tangent = raceline_tangent ./ max(vecnorm(raceline_tangent, 2, 2), eps);

n_gates = numel(gate_idx);
gate_center = zeros(2, n_gates);
heading = zeros(1, n_gates);
room = zeros(1, n_gates);
normal_left = [-segment.tangent(:, 2), segment.tangent(:, 1)];
for k = 1:n_gates
    idx = gate_idx(k);
    rl_idx = nearest_point_index(raceline, center(idx, :));
    gate_center(:, k) = raceline(rl_idx, :)';
    heading(k) = atan2(raceline_tangent(rl_idx, 2), raceline_tangent(rl_idx, 1));

    % 赛车线在弯心贴向内侧，圆盘半径不能再按中心线半宽给，要按该点到
    % 最近边界的剩余余量截断，否则 gate 直接压出走廊。
    lateral = (raceline(rl_idx, :) - center(idx, :)) * normal_left(idx, :)';
    room(k) = min(segment.half_width_left(idx) - lateral, ...
        segment.half_width_right(idx) + lateral);
end
room = max(room, eps);

is_end = false(1, n_gates);
is_end([1, end]) = true;

radius = room * radius_ratio;
radius(is_end) = room(is_end) * radius_ratio_ends;
spread = repmat(deg2rad(spread_deg), 1, n_gates);
spread(is_end) = deg2rad(spread_deg_ends);

gates.center = gate_center;
gates.heading = heading;
gates.heading_spread = spread;
gates.radius = radius;
gates.center_index = gate_idx(:)';
gates.corridor_room = room;
end

function apex_idx = corner_apex_indices(kappa, s_center)
%CORNER_APEX_INDICES 从曲率序列中提取每个连续弯道的 apex 索引。
% 先用基于曲率尺度的死区将序列符号化，再对每个非零符号游程选择 |kappa|
% 最大的点。s_center 用于排除零长度的退化游程。
% 用与 extract_track_segment 一致的死区符号化找弯，取每个弯内 |kappa| 最大处。
scale_hint = median(abs(kappa));
deadband = max(scale_hint, eps);
sign_seq = zeros(size(kappa));
sign_seq(kappa > deadband) = 1;
sign_seq(kappa < -deadband) = -1;

change_idx = [1; find(diff(sign_seq) ~= 0) + 1];
starts = change_idx;
stops = [change_idx(2:end) - 1; numel(sign_seq)];

apex_idx = [];
for run_idx = 1:numel(starts)
    if sign_seq(starts(run_idx)) == 0
        continue;
    end
    span = starts(run_idx):stops(run_idx);
    if s_center(stops(run_idx)) - s_center(starts(run_idx)) <= 0
        continue;
    end
    [~, local_idx] = max(abs(kappa(span)));
    apex_idx(end + 1) = span(local_idx); %#ok<AGROW>
end
end

function idx = nearest_scalar_index(values, query)
%NEAREST_SCALAR_INDEX 返回一维数组中与 query 绝对距离最小的元素索引。
[~, idx] = min(abs(values - query));
end

function idx = nearest_point_index(points, query)
%NEAREST_POINT_INDEX 返回二维点集中与 query 欧氏距离最小的行索引。
[~, idx] = min(sum((points - query) .^ 2, 2));
end

function g = gradient_rows(xy)
%GRADIENT_ROWS 分别沿行索引估计二维曲线的 x、y 一阶导数。
g = [gradient(xy(:, 1)), gradient(xy(:, 2))];
end

function kappa = discrete_curvature(xy)
%DISCRETE_CURVATURE 用一、二阶有限差分估计开放二维曲线的有符号曲率。
% 正负号用于区分左右弯，绝对值用于寻找弯道 apex。
dx = gradient(xy(:, 1));
dy = gradient(xy(:, 2));
ddx = gradient(dx);
ddy = gradient(dy);
denominator = max((dx .^ 2 + dy .^ 2) .^ 1.5, eps);
kappa = (dx .* ddy - dy .* ddx) ./ denominator;
end

function speed = control_point_speed(p_set, delta_t)
%CONTROL_POINT_SPEED 为每个 Hermite 控制点估计参数导数的模长。
% 端点采用单侧控制点距离，内部点采用左右距离平均，最后除以相邻参数间隔
% delta_t，将空间距离换算为 dp/dt 的尺度。
% 每个控制点的切向量模长 ~ 相邻控制点间距 / delta_t（端点取单侧）。
gaps = vecnorm(diff(p_set, 1, 2), 2, 1);
speed = nan(1, size(p_set, 2));
speed(1) = gaps(1);
speed(end) = gaps(end);
if numel(speed) > 2
    speed(2:(end - 1)) = (gaps(1:(end - 1)) + gaps(2:end)) / 2.0;
end
speed = speed / delta_t;
end

function ok = inside_corridor(segment, query_xy, inset_ratio)
%INSIDE_CORRIDOR 判断一组二维查询点是否全部位于赛道走廊内。
% 对每个点找到最近中心线点，并投影到该处左法向，得到带符号横向偏移：
%   lateral > 0 表示中心线左侧，lateral < 0 表示右侧。
% 左右允许宽度分别乘 (1-inset_ratio)，从而在真实边界内预留安全内缩。
% 该判断是几何拒绝采样，不涉及车辆动力学或推理阶段的 QP/CBF guidance。
% 对每个查询点取最近的中心线点，用左法向算带符号横向偏移，判断是否在走廊内。
center = segment.center;
normal_left = [-segment.tangent(:, 2), segment.tangent(:, 1)];

% 分块计算，避免 n_dense * n_center 的距离矩阵吃满内存。
block_size = 512;
n_query = size(query_xy, 1);
ok = true;
for block_start = 1:block_size:n_query
    block_stop = min(block_start + block_size - 1, n_query);
    block = query_xy(block_start:block_stop, :);
    d2 = sum((permute(block, [1, 3, 2]) - permute(center, [3, 1, 2])) .^ 2, 3);
    [~, nearest_idx] = min(d2, [], 2);

    offset = block - center(nearest_idx, :);
    lateral = sum(offset .* normal_left(nearest_idx, :), 2);

    limit_left = segment.half_width_left(nearest_idx) * (1.0 - inset_ratio);
    limit_right = segment.half_width_right(nearest_idx) * (1.0 - inset_ratio);
    if any(lateral > limit_left) || any(lateral < -limit_right)
        ok = false;
        return;
    end
end
end
