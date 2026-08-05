% 在赛道内生成二维 Hermite 轨迹。
% points 的特征为 [x, y, dx/ds, dy/ds]，gates 保存控制点采样区域。

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
% gate 间距和横向覆盖范围。
gate_spacing_m = struct_field_default(opts, 'gate_spacing_m', 20.0);
gate_fill_ratio = struct_field_default(opts, 'gate_fill_ratio', 0.92);
gate_fill_ratio_ends = struct_field_default(opts, 'gate_fill_ratio_ends', 0.55);

% 控制横向随机曲线的平滑程度和分布宽度。
lateral_correlation_m = struct_field_default(opts, 'lateral_correlation_m', 120.0);
lateral_target_std = struct_field_default(opts, 'lateral_target_std', 0.55);
corner_spread_boost = struct_field_default(opts, 'corner_spread_boost', 0.6);
% 1 表示围绕赛车线，0 表示围绕中心线。
raceline_bias = struct_field_default(opts, 'raceline_bias', 1.0);
gate_heading_jitter_deg = struct_field_default(opts, ...
    'gate_heading_jitter_deg', 3.0);

% 边界内缩和折线检查参数。
corridor_inset_ratio = struct_field_default(opts, 'corridor_inset_ratio', 0.02);
polyline_check_enabled = struct_field_default(opts, ...
    'polyline_check_enabled', true);
polyline_inset_ratio = struct_field_default(opts, 'polyline_inset_ratio', 0.0);

% curvature 模式在弯道放置更多采样点。
sampling_mode = struct_field_default(opts, 'sampling_mode', 'curvature');
curvature_floor = struct_field_default(opts, 'curvature_floor', 4.0);
max_attempts_per_trajectory = struct_field_default(opts, ...
    'max_attempts_per_trajectory', 200);
n_dense = struct_field_default(opts, 'n_dense', 5000);
rng_seed = struct_field_default(opts, 'rng_seed', 0);

previous_rng_state = rng;
restore_rng_state = onCleanup(@() rng(previous_rng_state));
rng(rng_seed);

% 构造 gate。
gates = build_segment_gates(segment, gate_spacing_m, gate_fill_ratio, ...
    gate_fill_ratio_ends);
p_quantity = numel(gates.s);
smooth_window = max(round(lateral_correlation_m * segment.transform.scale / ...
    (gate_spacing_m * segment.transform.scale)), 1);

% 构造折线检查使用的走廊多边形。
poly_scale = 1.0 - polyline_inset_ratio;
poly_normal_left = [-segment.tangent(:, 2), segment.tangent(:, 1)];
poly_left = segment.center + poly_normal_left .* ...
    (segment.half_width_left * poly_scale);
poly_right = segment.center - poly_normal_left .* ...
    (segment.half_width_right * poly_scale);
corridor_x = [poly_left(:, 1); flipud(poly_right(:, 1))];
corridor_y = [poly_left(:, 2); flipud(poly_right(:, 2))];
% 首尾封口不参与相交检查。
boundary_edges = [poly_left(1:(end - 1), :), poly_left(2:end, :); ...
    poly_right(1:(end - 1), :), poly_right(2:end, :)];

feature_dim = 4;
points = nan(n_points, n_trajectories, feature_dim);
t_set = (0:(p_quantity - 1)) / (p_quantity - 1);
t_dense_set = linspace(0, 1, n_dense);
delta_t = t_set(2) - t_set(1);

for trajectory_nr = 1:n_trajectories
    accepted = false;
    % 越界时重新生成整条轨迹。
    for attempt = 1:max_attempts_per_trajectory
        % 生成平滑的随机横向偏移。
        u = movmean(randn(1, p_quantity), smooth_window, 'Endpoints', 'shrink');
        u = u / max(std(u), eps);
        u = u .* (lateral_target_std * (1 + corner_spread_boost * gates.kappa_norm));
        u = max(min(raceline_bias * gates.rho_raceline + u, 1.0), -1.0);

        % 将归一化偏移换成实际横向位置。
        lateral = u .* gates.extent_left;
        right_side = u < 0;
        lateral(right_side) = u(right_side) .* gates.extent_right(right_side);

        p_set = gates.center + gates.normal_left .* lateral;

        % 根据横向变化计算控制点朝向。
        dlateral_ds = gradient(lateral) ./ gradient(gates.s);
        heading_set = gates.heading + atan(dlateral_ds) + ...
            (rand(1, p_quantity) - 0.5) * deg2rad(gate_heading_jitter_deg);

        % 构造 Hermite 切向量。
        dp_set = [cos(heading_set); sin(heading_set)];
        dp_set = dp_set .* control_point_speed(p_set, delta_t);

        polynomial_coefficient_x = GPFM_DataGeneration_fit_Polynomial( ...
            t_set, p_set(1, :), dp_set(1, :));
        polynomial_coefficient_y = GPFM_DataGeneration_fit_Polynomial( ...
            t_set, p_set(2, :), dp_set(2, :));

        x_dense_set = ppval(polynomial_coefficient_x, t_dense_set);
        y_dense_set = ppval(polynomial_coefficient_y, t_dense_set);

        % 检查密集 Hermite 曲线是否越界。
        [dense_lateral, dense_idx] = track_lateral_offset(segment, ...
            [x_dense_set(:), y_dense_set(:)]);
        limit_scale = 1.0 - corridor_inset_ratio;
        if any(dense_lateral > segment.half_width_left(dense_idx) * limit_scale) ...
                || any(dense_lateral < -segment.half_width_right(dense_idx) * limit_scale)
            continue;
        end

        ds_dense_set = sqrt((x_dense_set(2:end) - x_dense_set(1:(end - 1))) .^ 2 + ...
            (y_dense_set(2:end) - y_dense_set(1:(end - 1))) .^ 2);
        s_dense_set = [0, cumsum(ds_dense_set)];
        trajectory_length = s_dense_set(end);

        % 从密集曲线中选取最终训练点。
        if strcmp(sampling_mode, 'curvature')
            kappa_dense = parametric_curvature(x_dense_set, y_dense_set);
            weight = sqrt(max(abs(kappa_dense), curvature_floor));
            mu_dense = cumtrapz(s_dense_set, weight);
            mu_set = linspace(0, mu_dense(end), n_points);
            t_train_i = interp1(mu_dense, t_dense_set, mu_set);
        else
            s_set = linspace(0, trajectory_length, n_points);
            t_train_i = interp1(s_dense_set, t_dense_set, s_set);
        end
        x_train_i = ppval(polynomial_coefficient_x, t_train_i);
        y_train_i = ppval(polynomial_coefficient_y, t_train_i);

        % 检查训练点连成的折线是否越界。
        if polyline_check_enabled && ~polyline_inside_corridor(corridor_x, ...
                corridor_y, boundary_edges, x_train_i, y_train_i)
            continue;
        end

        points(:, trajectory_nr, 1) = x_train_i;
        points(:, trajectory_nr, 2) = y_train_i;

        dx_dt_train_i = ppval(pp_derivative(polynomial_coefficient_x), t_train_i);
        dy_dt_train_i = ppval(pp_derivative(polynomial_coefficient_y), t_train_i);
        speed_train_i = max(sqrt(dx_dt_train_i .^ 2 + dy_dt_train_i .^ 2), eps);
        % 保存单位切向量。
        points(:, trajectory_nr, 3) = dx_dt_train_i ./ speed_train_i;
        points(:, trajectory_nr, 4) = dy_dt_train_i ./ speed_train_i;

        accepted = true;
        break;
    end
    if ~accepted
        error('generate_track_training_points_2d:rejectionFailed', ...
            ['Trajectory %d not accepted after %d attempts. With a coarse ' ...
            'n_points the polyline check can never pass — take a subset of ' ...
            'the finest level instead, or set polyline_check_enabled=false. ' ...
            'Otherwise shrink lateral_target_std / gate_fill_ratio or raise ' ...
            'lateral_correlation_m.'], trajectory_nr, max_attempts_per_trajectory);
    end
end
end

%% ------------------------------------------------------------------------
function kappa = parametric_curvature(x_set, y_set)
% 计算二维曲线曲率。
dx = gradient(x_set(:));
dy = gradient(y_set(:));
ddx = gradient(dx);
ddy = gradient(dy);
kappa = (dx .* ddy - dy .* ddx) ./ max((dx .^ 2 + dy .^ 2) .^ 1.5, eps);
kappa = kappa';
end

%% ------------------------------------------------------------------------
function ok = polyline_inside_corridor(corridor_x, corridor_y, ...
    boundary_edges, x_set, y_set)
% 检查折线顶点和线段是否都在走廊内。
if numel(x_set) < 2
    ok = true;
    return;
end
if ~all(inpolygon(x_set(:), y_set(:), corridor_x, corridor_y))
    ok = false;
    return;
end

p1x = x_set(1:(end - 1))'; p1y = y_set(1:(end - 1))';
p2x = x_set(2:end)';       p2y = y_set(2:end)';
q1x = boundary_edges(:, 1)'; q1y = boundary_edges(:, 2)';
q2x = boundary_edges(:, 3)'; q2y = boundary_edges(:, 4)';

% 用叉积判断折线是否穿过边界。
ex = q2x - q1x; ey = q2y - q1y;
d1 = ex .* (p1y - q1y) - ey .* (p1x - q1x);
d2 = ex .* (p2y - q1y) - ey .* (p2x - q1x);
sx = p2x - p1x; sy = p2y - p1y;
d3 = sx .* (q1y - p1y) - sy .* (q1x - p1x);
d4 = sx .* (q2y - p1y) - sy .* (q2x - p1x);
ok = ~any((d1 .* d2 < 0) & (d3 .* d4 < 0), 'all');
end

%% ------------------------------------------------------------------------
function gates = build_segment_gates(segment, spacing_m, fill_ratio, ...
    fill_ratio_ends)
% 沿赛道构造横向线段 gate。
center = segment.center;
n_center = size(center, 1);
s_center = segment.s_center(:);

apex_idx = corner_apex_indices(discrete_curvature(center));

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
n_gates = numel(gate_idx);

normal_left = [-segment.tangent(:, 2), segment.tangent(:, 1)];
tangent = segment.tangent(gate_idx, :);

ratio = repmat(fill_ratio, 1, n_gates);
ratio([1, end]) = fill_ratio_ends;

% 计算赛车线在各 gate 上的位置。
raceline = segment.raceline;
rho_raceline = zeros(1, n_gates);
for k = 1:n_gates
    idx = gate_idx(k);
    rl_idx = nearest_point_index(raceline, center(idx, :));
    lateral = (raceline(rl_idx, :) - center(idx, :)) * normal_left(idx, :)';
    if lateral >= 0
        rho_raceline(k) = lateral / max(segment.half_width_left(idx), eps);
    else
        rho_raceline(k) = lateral / max(segment.half_width_right(idx), eps);
    end
end

gates.center = center(gate_idx, :)';
gates.normal_left = normal_left(gate_idx, :)';
gates.heading = atan2(tangent(:, 2), tangent(:, 1))';
gates.s = s_center(gate_idx)';
gates.extent_left = segment.half_width_left(gate_idx)' .* ratio;
gates.extent_right = segment.half_width_right(gate_idx)' .* ratio;
gates.rho_raceline = max(min(rho_raceline, 1.0), -1.0);
% 计算各 gate 的归一化曲率。
kappa_gate = abs(discrete_curvature(center));
kappa_gate = movmean(kappa_gate, 9);
kappa_gate = kappa_gate(gate_idx)';
gates.kappa_norm = min(kappa_gate / max(prctile(kappa_gate, 95), eps), 1.0);
gates.center_index = gate_idx(:)';
end

function apex_idx = corner_apex_indices(kappa)
% 取每个弯道中曲率最大的点。
deadband = max(median(abs(kappa)), eps);
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
    [~, local_idx] = max(abs(kappa(span)));
    apex_idx(end + 1) = span(local_idx); %#ok<AGROW>
end
end

function idx = nearest_scalar_index(values, query)
% 查找最近的一维索引。
[~, idx] = min(abs(values - query));
end

function kappa = discrete_curvature(xy)
% 计算离散二维曲线曲率。
dx = gradient(xy(:, 1));
dy = gradient(xy(:, 2));
ddx = gradient(dx);
ddy = gradient(dy);
denominator = max((dx .^ 2 + dy .^ 2) .^ 1.5, eps);
kappa = (dx .* ddy - dy .* ddx) ./ denominator;
end

function speed = control_point_speed(p_set, delta_t)
% 根据相邻控制点距离设置 Hermite 导数大小。
gaps = vecnorm(diff(p_set, 1, 2), 2, 1);
speed = nan(1, size(p_set, 2));
speed(1) = gaps(1);
speed(end) = gaps(end);
if numel(speed) > 2
    speed(2:(end - 1)) = (gaps(1:(end - 1)) + gaps(2:end)) / 2.0;
end
speed = speed / delta_t;
end
