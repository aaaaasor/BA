function info = track_boundary_cbf_info(stats, constraint_cfg, t)
%TRACK_BOUNDARY_CBF_INFO Linear CBF rows for a local track cross-section.
% Both boundary rows use the same centerline parameter.  This is essential
% on a folded track: finding the nearest point on the left and right curves
% independently can select two different branches and falsely classify the
% empty region between them as part of the track.

info.enabled = struct_field_default(constraint_cfg, ...
    'track_boundary_enabled', false);
n_u = numel(stats.mu);
info.A = zeros(0, n_u);
info.bounds = zeros(0, 1);
info.h_values = zeros(0, 1);
info.phi_values = zeros(0, 1);
info.s_values = zeros(0, 1);
info.row_point_indices = zeros(0, 1);
info.row_boundary_indices = zeros(0, 1);
if ~info.enabled
    return;
end
activation_time = struct_field_default(constraint_cfg, ...
    'track_boundary_activation_time', 0.0);
if t < activation_time
    info.enabled = false;
    return;
end
if ~isfield(constraint_cfg, 'track_boundary_geometry') || ...
        ~isfield(constraint_cfg, 'track_boundary_point_maps')
    error(['Track boundary constraint requires track_boundary_geometry and ', ...
        'track_boundary_point_maps.']);
end

geometry = constraint_cfg.track_boundary_geometry;
point_maps = constraint_cfg.track_boundary_point_maps;
reference_s_min = zeros(1, numel(point_maps));
reference_s_max = ones(1, numel(point_maps));
if isfield(constraint_cfg, 'track_boundary_reference_s_min') && ...
        isfield(constraint_cfg, 'track_boundary_reference_s_max')
    reference_s_min = constraint_cfg.track_boundary_reference_s_min(:)';
    reference_s_max = constraint_cfg.track_boundary_reference_s_max(:)';
    if numel(reference_s_min) ~= numel(point_maps) || ...
            numel(reference_s_max) ~= numel(point_maps)
        error('Track boundary reference window count must match point maps.');
    end
end
% 搜索窗口不只是为了排除折叠赛道的另一条分支 —— 实测它还是唯一的纵向回正
% 机制: 窗口夹住时 s_near 被钉在边沿, 由此算出的 h 会把沿赛道滑走的点推回
% 名义站点附近。改成全局搜索后 h 变准了(与真实横向余量之差 <= 1.2e-3), 但
% 这个回正力随之消失, 第二层立刻退化: 相邻点 s 间隔倒退比例 13.9% -> 36.9%,
% |s - s_ref| 最大 3.64 -> 13.48 个名义间距(全赛道只有 16 个), 折线变成跨越
% 全场的乱线。在有专门的纵向约束(比如对 s 的单调性 CBF)之前, 窗口必须保留。
phase_search_half_steps = struct_field_default(constraint_cfg, ...
    'track_boundary_phase_search_half_steps', 2.0);
reference_cache = struct_field_default(constraint_cfg, ...
    'track_boundary_reference_cache', struct([]));
if ~isempty(reference_cache) && numel(reference_cache) ~= numel(point_maps)
    error('Track boundary reference cache count must match point maps.');
end
if ~isscalar(phase_search_half_steps) || ~isfinite(phase_search_half_steps) || ...
        phase_search_half_steps <= 0
    error('track_boundary_phase_search_half_steps must be finite and positive.');
end
phi0 = struct_field_default(constraint_cfg, ...
    'track_boundary_phi0', 5.0);
phi1_early = struct_field_default(constraint_cfg, ...
    'track_boundary_phi1_early_gain', phi0);
phi1_switch_time = struct_field_default(constraint_cfg, ...
    'track_boundary_phi1_switch_time', 0.85);
omega = struct_field_default(constraint_cfg, ...
    'track_boundary_phi1_omega', 0.1);
phi1_max = struct_field_default(constraint_cfg, ...
    'track_boundary_phi1_max', inf);
tau_min = struct_field_default(constraint_cfg, ...
    'track_boundary_phi1_tau_min', 0.01);
margin = struct_field_default(constraint_cfg, 'track_boundary_margin', 0.0);
true_distance_enabled = struct_field_default(constraint_cfg, ...
    'track_boundary_true_distance_enabled', false);
true_distance_activation_time = struct_field_default(constraint_cfg, ...
    'track_boundary_true_distance_activation_time', 0.0);
true_distance_active = true_distance_enabled && ...
    t >= true_distance_activation_time;
grad_tol = struct_field_default(constraint_cfg, 'grad_tol', 1e-6);
if ~isscalar(phi0) || ~isfinite(phi0) || phi0 <= 0
    error('track_boundary_phi0 must be finite and positive.');
end
if ~isscalar(phi1_early) || ~isfinite(phi1_early) || phi1_early <= 0
    error('track_boundary_phi1_early_gain must be finite and positive.');
end
if ~isscalar(phi1_switch_time) || ~isfinite(phi1_switch_time) || ...
        phi1_switch_time < 0
    error('track_boundary_phi1_switch_time must be finite and nonnegative.');
end
if ~isscalar(omega) || ~isfinite(omega) || omega <= 0
    error('track_boundary_phi1_omega must be finite and positive.');
end
if ~isscalar(phi1_max) || isnan(phi1_max) || phi1_max <= 0
    error('track_boundary_phi1_max must be positive or Inf.');
end
if ~isscalar(tau_min) || ~isfinite(tau_min) || tau_min <= 0
    error('track_boundary_phi1_tau_min must be finite and positive.');
end
if ~isscalar(margin) || ~isfinite(margin) || margin < 0
    error('track_boundary_margin must be finite and nonnegative.');
end
boundary_slack_configured = struct_field_default(constraint_cfg, ...
    'track_boundary_slack_enabled', false);
boundary_hard_after_time = struct_field_default(constraint_cfg, ...
    'track_boundary_slack_hard_after_time', inf);
boundary_slack_active = boundary_slack_configured && ...
    t < boundary_hard_after_time;
if boundary_slack_active
    % Soft boundary: regularize RK4's exact endpoint evaluation.
    tau = max(1.0 - t, tau_min);
else
    % Hard boundary: retain the uncapped prescribed-time gain.  eps only
    % prevents division by literal zero and is not a practical gain cap.
    tau = max(1.0 - t, eps);
end
phi1 = min(omega / (tau ^ 2), phi1_max);

constraint_method = lower(string(struct_field_default(constraint_cfg, ...
    'track_boundary_constraint_method', 'paired_cross_section')));
if constraint_method == "global_implicit_fields"
    combine_method = lower(string(struct_field_default(constraint_cfg, ...
        'track_boundary_combine_method', 'separate')));
    softmin_kappa = struct_field_default(constraint_cfg, ...
        'track_boundary_softmin_kappa', 500.0);
    info = global_implicit_field_cbf_rows(info, stats, geometry, ...
        point_maps, margin, phi0, phi1_early, phi1_switch_time, phi1, ...
        t, grad_tol, combine_method, softmin_kappa);
    return;
elseif constraint_method == "independent_signed_distance"
    info = independent_signed_distance_cbf_rows(info, stats, geometry, ...
        point_maps, margin, phi0, phi1_early, phi1_switch_time, phi1, ...
        t, grad_tol);
    return;
elseif constraint_method ~= "paired_cross_section"
    error(['track_boundary_constraint_method must be ', ...
        '''paired_cross_section'', ''independent_signed_distance'', ', ...
        'or ''global_implicit_fields''.']);
end

for map_idx = 1:numel(point_maps)
    point_map = point_maps(map_idx);
    p = point_map.M * stats.x(:) + point_map.o;
    s_reference = 0.5 * (reference_s_min(map_idx) + ...
        reference_s_max(map_idx));
    reference_half_width = max(0.5 * (reference_s_max(map_idx) - ...
        reference_s_min(map_idx)), eps);
    search_half_width = phase_search_half_steps * 2.0 * ...
        reference_half_width;
    if isempty(reference_cache)
        [s_near, window_min, window_max] = branch_locked_center_parameter( ...
            p, geometry, s_reference, search_half_width);
    else
        [s_near, window_min, window_max, phase_curvature] = ...
            branch_locked_center_parameter_cached(p, geometry, ...
            reference_cache(map_idx));
    end
    if isempty(reference_cache)
        phase_curvature = 0.0;
    end
    q_left = evaluate_curve(geometry.curves(1), s_near);
    q_right = evaluate_curve(geometry.curves(2), s_near);
    left_to_right = q_right - q_left;
    corridor_width = norm(left_to_right);
    if corridor_width < grad_tol
        continue;
    end
    cross_track = left_to_right ./ corridor_width;
    boundary_points = [q_left, q_right];
    inward_normals = [cross_track, -cross_track];
    % 梯度里的链式项。h 通过 s_near 依赖 p, 而 s_near = argmin |p-c(s)|^2 只是
    % 中心线距离的驻点, 不是 h 的驻点(实测 dh/ds = 0.265), 所以包络定理在这里
    % 不适用, 这一项是一阶的。丢掉它实测每步造成 dh = -4.1e-5, 而 CBF 每步只
    % 预算 2.0e-6, 结果是前向不变性以一个与 dt 无关的量失效(加密时间步无效)。
    % 补上后残差降为二阶。ds*/dp 由驻点条件 (p-c(s))'*c'(s)=0 隐函数求导得到。
    [ds_dp, dcross_track_ds, q_left_prime, q_right_prime] = ...
        center_parameter_sensitivity(p, geometry, s_near, window_min, ...
        window_max, cross_track, corridor_width, grad_tol, phase_curvature);
    window_pair_h = zeros(2, 1);
    window_pair_grad = zeros(2, n_u);
    for boundary_idx = 1:2
        inward_normal = inward_normals(:, boundary_idx);
        h_window = inward_normal' * ...
            (p - boundary_points(:, boundary_idx)) - margin;
        if boundary_idx == 1
            dh_ds = dcross_track_ds' * (p - q_left) - ...
                cross_track' * q_left_prime;
        else
            dh_ds = -dcross_track_ds' * (p - q_right) + ...
                cross_track' * q_right_prime;
        end
        grad_window = (inward_normal' + dh_ds * ds_dp) * point_map.M;
        window_pair_h(boundary_idx) = h_window;
        window_pair_grad(boundary_idx, :) = grad_window;
    end
    true_pair_h = inf(2, 1);
    true_pair_grad = zeros(2, n_u);
    if true_distance_active && ~isempty(reference_cache)
        [true_pair_h, true_pair_grad] = ...
            true_branch_cross_section_pair(p, geometry, ...
            reference_cache(map_idx), margin, point_map.M, grad_tol);
    end
    % Never mix one rail from the phase window with the opposite rail from
    % the true-distance geometry.  Such a hybrid pair need not bound one
    % common corridor and can create contradictory hard inequalities.  Pick
    % the complete pair whose worst side is more restrictive instead.
    if true_distance_active && all(isfinite(true_pair_h)) && ...
            min(true_pair_h) < min(window_pair_h)
        pair_h = true_pair_h;
        pair_grad = true_pair_grad;
    else
        pair_h = window_pair_h;
        pair_grad = window_pair_grad;
    end
    % The two sides of one cross-section must share the recovery gain.  If
    % only the violated side used the prescribed-time gain while the safe
    % side retained phi0, the pair could demand a large inward velocity and
    % simultaneously impose a smaller opposite speed cap.  A common gain
    % preserves pair feasibility because h_left+h_right equals the corridor
    % width (minus the two margins).
    if all(pair_h >= 0)
        pair_phi = phi0;
    elseif t < phi1_switch_time
        pair_phi = phi1_early;
    else
        pair_phi = phi1;
    end
    for boundary_idx = 1:2
        h = pair_h(boundary_idx);
        grad = pair_grad(boundary_idx, :);
        if norm(grad) < grad_tol
            continue;
        end
        phi = pair_phi;
        % Prescribed-time CBF: hdot + phi(t,h)*h >= 0.
        info.A(end + 1, :) = -grad;
        info.bounds(end + 1, 1) = grad * stats.mu(:) + phi * h;
        info.h_values(end + 1, 1) = h;
        info.phi_values(end + 1, 1) = phi;
        info.s_values(end + 1, 1) = s_near;
        info.row_point_indices(end + 1, 1) = point_map.point_index;
        info.row_boundary_indices(end + 1, 1) = boundary_idx;
    end
end
end

function info = global_implicit_field_cbf_rows(info, stats, geometry, ...
    point_maps, margin, phi0, phi1_early, phi1_switch_time, phi1, t, ...
    grad_tol, combine_method, softmin_kappa)
% Evaluate the two preconstructed global functions directly.  No centerline
% point, phase window, branch label, or nearest-point solve is performed here.
if ~isfield(geometry, 'implicit_fields')
    error('Track geometry does not contain precomputed implicit fields.');
end
fields = geometry.implicit_fields;
if combine_method ~= "separate" && combine_method ~= "softmin"
    error(['track_boundary_combine_method must be ''separate'' or ', ...
        '''softmin''.']);
end
if combine_method == "softmin" && ...
        (~isscalar(softmin_kappa) || ~isfinite(softmin_kappa) || ...
        softmin_kappa <= 0)
    error('track_boundary_softmin_kappa must be finite and positive.');
end
for map_idx = 1:numel(point_maps)
    point_map = point_maps(map_idx);
    p = point_map.M * stats.x(:) + point_map.o;
    pair_h = zeros(2, 1);
    pair_grad = zeros(2, numel(stats.mu));
    for boundary_idx = 1:2
        [h_raw, grad_p] = evaluate_global_implicit_field( ...
            fields, boundary_idx, p);
        pair_h(boundary_idx) = h_raw - margin;
        pair_grad(boundary_idx, :) = grad_p * point_map.M;
    end
    if combine_method == "softmin"
        % Stable conservative soft minimum:
        %   h_soft = -log(exp(-k*h1)+exp(-k*h2))/k <= min(h1,h2).
        % Subtracting min(h) keeps the exponent arguments nonpositive and
        % avoids overflow even for a large kappa or far-off-track state.
        h_min = min(pair_h);
        exponentials = exp(-softmin_kappa .* (pair_h - h_min));
        weight_sum = sum(exponentials);
        weights = exponentials ./ weight_sum;
        h = h_min - log(weight_sum) / softmin_kappa;
        grad = weights' * pair_grad;
        if norm(grad) < grad_tol
            continue;
        end
        if h >= 0
            phi = phi0;
        elseif t < phi1_switch_time
            phi = phi1_early;
        else
            phi = phi1;
        end
        info.A(end + 1, :) = -grad;
        info.bounds(end + 1, 1) = grad * stats.mu(:) + phi * h;
        info.h_values(end + 1, 1) = h;
        info.phi_values(end + 1, 1) = phi;
        info.s_values(end + 1, 1) = nan;
        info.row_point_indices(end + 1, 1) = point_map.point_index;
        info.row_boundary_indices(end + 1, 1) = 0;
        continue;
    end
    for boundary_idx = 1:2
        h = pair_h(boundary_idx);
        grad = pair_grad(boundary_idx, :);
        if norm(grad) < grad_tol
            continue;
        end
        if h >= 0
            phi = phi0;
        elseif t < phi1_switch_time
            phi = phi1_early;
        else
            phi = phi1;
        end
        info.A(end + 1, :) = -grad;
        info.bounds(end + 1, 1) = grad * stats.mu(:) + phi * h;
        info.h_values(end + 1, 1) = h;
        info.phi_values(end + 1, 1) = phi;
        info.s_values(end + 1, 1) = nan;
        info.row_point_indices(end + 1, 1) = point_map.point_index;
        info.row_boundary_indices(end + 1, 1) = boundary_idx;
    end
end
end

function [h, grad] = evaluate_global_implicit_field(fields, boundary_idx, p)
% Single shared evaluator.  The terminal safety filter must classify points
% with exactly the equation the CBF rows enforce, so both call one function.
[h, grad] = evaluate_track_implicit_field(fields, boundary_idx, p);
end

function info = independent_signed_distance_cbf_rows(info, stats, geometry, ...
    point_maps, margin, phi0, phi1_early, phi1_switch_time, phi1, t, grad_tol)
% Two independent implicit functions h_left(p), h_right(p).  For each rail,
% h is the signed Euclidean distance to the globally nearest spline point,
% positive on the side facing the track center.  No common center parameter,
% nominal phase, branch lock, or longitudinal history enters these rows.
for map_idx = 1:numel(point_maps)
    point_map = point_maps(map_idx);
    p = point_map.M * stats.x(:) + point_map.o;
    for boundary_idx = 1:2
        curve = geometry.curves(boundary_idx);
        [s_near, q] = nearest_rail_point(p, curve);
        center_at = [ppval(geometry.center_pp_x, s_near); ...
            ppval(geometry.center_pp_y, s_near)];
        inward_hint = center_at - q;
        inward_norm = norm(inward_hint);
        if inward_norm < grad_tol
            continue;
        end
        inward_hint = inward_hint ./ inward_norm;
        offset = p - q;
        distance = norm(offset);
        if distance < grad_tol
            signed_distance = 0.0;
            grad_p = inward_hint';
        else
            side_sign = sign(offset' * inward_hint);
            if side_sign == 0
                side_sign = 1.0;
            end
            signed_distance = side_sign * distance;
            grad_p = side_sign * offset' / distance;
        end
        h = signed_distance - margin;
        grad = grad_p * point_map.M;
        if norm(grad) < grad_tol
            continue;
        end
        if h >= 0
            phi = phi0;
        elseif t < phi1_switch_time
            phi = phi1_early;
        else
            phi = phi1;
        end
        info.A(end + 1, :) = -grad;
        info.bounds(end + 1, 1) = grad * stats.mu(:) + phi * h;
        info.h_values(end + 1, 1) = h;
        info.phi_values(end + 1, 1) = phi;
        info.s_values(end + 1, 1) = s_near;
        info.row_point_indices(end + 1, 1) = point_map.point_index;
        info.row_boundary_indices(end + 1, 1) = boundary_idx;
    end
end
end

function [s_near, q] = nearest_rail_point(p, curve)
s_grid = curve.distance_s_grid;
points = curve.distance_points;
distance_grid = sum((points - p) .^ 2, 1);
[~, nearest_idx] = min(distance_grid);
lo = s_grid(max(nearest_idx - 1, 1));
hi = s_grid(min(nearest_idx + 1, numel(s_grid)));
distance_sq = @(s) (ppval(curve.pp_x, s) - p(1)) .^ 2 + ...
    (ppval(curve.pp_y, s) - p(2)) .^ 2;
if hi > lo
    s_near = fminbnd(distance_sq, lo, hi, center_search_options());
else
    s_near = lo;
end
q = evaluate_curve(curve, s_near);
end

function [pair_h, pair_grad] = true_branch_cross_section_pair( ...
    p, geometry, cache, margin, point_map_M, grad_tol)
%TRUE_BRANCH_CROSS_SECTION_PAIR Safety pair at the geometric nearest phase.
% The phase-locked pair above still supplies longitudinal ordering.  This
% second search ignores the soft phase score, but is restricted to the same
% topological branch (plus the configured overlap).  Both rails are then
% evaluated at one common center parameter, so the two hard rows always
% describe one real corridor cross-section and retain the smooth spline
% sensitivity used by the original boundary PTCBF.
pair_h = inf(2, 1);
pair_grad = zeros(2, size(point_map_M, 2));
s_grid = cache.safety_s_grid;
center_points = cache.safety_center_points;
if numel(s_grid) < 2 || size(center_points, 2) ~= numel(s_grid)
    return;
end
distance_grid = sum((center_points - p) .^ 2, 1);
[~, nearest_idx] = min(distance_grid);
lo = s_grid(max(nearest_idx - 1, 1));
hi = s_grid(min(nearest_idx + 1, numel(s_grid)));
distance_sq = @(s) (ppval(geometry.center_pp_x, s) - p(1)) .^ 2 + ...
    (ppval(geometry.center_pp_y, s) - p(2)) .^ 2;
if hi > lo
    s_near = fminbnd(distance_sq, lo, hi, center_search_options());
else
    s_near = lo;
end
q_left = evaluate_curve(geometry.curves(1), s_near);
q_right = evaluate_curve(geometry.curves(2), s_near);
left_to_right = q_right - q_left;
corridor_width = norm(left_to_right);
if corridor_width < grad_tol
    return;
end
cross_track = left_to_right ./ corridor_width;
[ds_dp, dcross_track_ds, q_left_prime, q_right_prime] = ...
    center_parameter_sensitivity(p, geometry, s_near, s_grid(1), ...
    s_grid(end), cross_track, corridor_width, grad_tol, 0.0);
pair_h(1) = cross_track' * (p - q_left) - margin;
dh_left_ds = dcross_track_ds' * (p - q_left) - ...
    cross_track' * q_left_prime;
pair_grad(1, :) = (cross_track' + dh_left_ds * ds_dp) * point_map_M;
pair_h(2) = -cross_track' * (p - q_right) - margin;
dh_right_ds = -dcross_track_ds' * (p - q_right) + ...
    cross_track' * q_right_prime;
pair_grad(2, :) = (-cross_track' + dh_right_ds * ds_dp) * point_map_M;
end

function options = center_search_options()
%CENTER_SEARCH_OPTIONS  最近点搜索的收敛容差。
% fminbnd 的默认 TolX = 1e-4 对这里远远不够: s_near 的误差通过 dh/ds(实测
% 约 0.26) 直接进入 h, 即 h 上有 2.6e-5 的数值噪声。而 rollout 末段 h 已经
% 收敛到 1e-6 量级 —— 噪声是信号的 39 倍, 足以让一个真实 h = +6.8e-7 的安全
% 点被算成 h = -3.6e-6。再乘上最后一格的 phi = 2.0e30, 这个纯噪声就变成
% b = -7.4e24, 超过 linprog 的 -1e20 下限, QP 直接不可行。
% 1e-12 让 h 的噪声降到 2.6e-13, 远低于 h 本身。
persistent cached_options
if isempty(cached_options)
    cached_options = optimset('TolX', 1e-12);
end
options = cached_options;
end

function [s_near, s_min, s_max, phase_curvature] = ...
    branch_locked_center_parameter_cached( ...
    p, geometry, cache)
s_min = cache.s_min;
s_max = cache.s_max;
reference_weight = struct_field_default(cache, 'reference_weight', 0.0);
previous_weight = struct_field_default(cache, 'previous_weight', 0.0);
phase_scale = max(struct_field_default(cache, 'phase_scale', 1.0), eps);
s_nominal = struct_field_default(cache, 's_nominal', cache.s_reference);
s_previous = struct_field_default(cache, 's_previous', s_nominal);
grid_objective = (cache.x_grid - p(1)) .^ 2 + ...
    (cache.y_grid - p(2)) .^ 2 + ...
    reference_weight .* ((cache.s_grid - s_nominal) ./ phase_scale) .^ 2 + ...
    previous_weight .* ((cache.s_grid - s_previous) ./ phase_scale) .^ 2;
[~, nearest_idx] = min(grid_objective);
lo = max(cache.s_min, cache.s_grid(max(nearest_idx - 1, 1)));
hi = min(cache.s_max, cache.s_grid(min(nearest_idx + 1, numel(cache.s_grid))));
distance_sq = @(s) (ppval(geometry.center_pp_x, s) - p(1)) .^ 2 + ...
    (ppval(geometry.center_pp_y, s) - p(2)) .^ 2 + ...
    reference_weight .* ((s - s_nominal) ./ phase_scale) .^ 2 + ...
    previous_weight .* ((s - s_previous) ./ phase_scale) .^ 2;
if hi > lo
    s_near = fminbnd(distance_sq, lo, hi, center_search_options());
else
    s_near = lo;
end
phase_curvature = (reference_weight + previous_weight) / (phase_scale ^ 2);
end

function [s_near, s_min, s_max] = branch_locked_center_parameter(p, ...
    geometry, s_reference, search_half_width)
% Locate the closest centerline point inside the inherited lookup window.
% The window ends are not physical corridor faces, but clamping there is
% what pushes a longitudinally drifting point back toward its nominal
% station -- see the note at the top of this file.
s_reference = max(0.0, min(1.0, s_reference));
s_min = max(0.0, s_reference - search_half_width);
s_max = min(1.0, s_reference + search_half_width);
all_s = geometry.s_control;
    in_window = all_s > s_min & all_s < s_max;
    % Include both window faces: the constrained nearest point may lie
    % exactly at either face when the point has moved longitudinally.
    s_grid = unique([s_min, all_s(in_window), s_max]);
    if numel(s_grid) < 3
        s_grid = unique([s_grid, 0.5 * (s_min + s_max)]);
    end
x_grid = ppval(geometry.center_pp_x, s_grid);
y_grid = ppval(geometry.center_pp_y, s_grid);
grid_objective = (x_grid - p(1)) .^ 2 + (y_grid - p(2)) .^ 2;
[~, nearest_idx] = min(grid_objective);
lo = max(s_min, s_grid(max(nearest_idx - 1, 1)));
hi = min(s_max, s_grid(min(nearest_idx + 1, numel(s_grid))));
distance_sq = @(s) (ppval(geometry.center_pp_x, s) - p(1)) .^ 2 + ...
    (ppval(geometry.center_pp_y, s) - p(2)) .^ 2;
if hi > lo
    s_near = fminbnd(distance_sq, lo, hi, center_search_options());
else
    s_near = lo;
end
end

function [ds_dp, dcross_track_ds, q_left_prime, q_right_prime] = ...
    center_parameter_sensitivity(p, geometry, s_near, window_min, ...
    window_max, cross_track, corridor_width, grad_tol, phase_curvature)
%CENTER_PARAMETER_SENSITIVITY  ds*/dp 以及横截面随 s 的变化率。
% s* 由驻点条件 g(s,p) = (p - c(s))' * c'(s) = 0 隐式定义, 于是
%   ds*/dp = c'(s*)' / (|c'(s*)|^2 - (p - c(s*))' * c''(s*)).
% 分母趋零处是曲率焦点(中轴), 导数发散; 该处退回冻结横截面梯度。
% PCHIP 在控制点处的二阶导不连续，因此不可靠位置同样由下面的保护退回。
ds_dp = zeros(1, 2);
dcross_track_ds = zeros(2, 1);
q_left_prime = zeros(2, 1);
q_right_prime = zeros(2, 1);
% 搜索窗口夹住 s_near 时它不再随 p 移动, 真实的 ds*/dp 就是 0。
window_span = max(window_max - window_min, eps);
if min(s_near - window_min, window_max - s_near) < 1e-6 * window_span
    return;
end
c_at = [ppval(geometry.center_pp_x, s_near); ...
    ppval(geometry.center_pp_y, s_near)];
c_prime = [ppval(geometry.center_dpp_x, s_near); ...
    ppval(geometry.center_dpp_y, s_near)];
c_second = [ppval(geometry.center_ddpp_x, s_near); ...
    ppval(geometry.center_ddpp_y, s_near)];
speed_sq = c_prime' * c_prime;
denominator = speed_sq - (p - c_at)' * c_second + phase_curvature;
% 焦点保护: 分母相对 |c'|^2 太小就放弃这一项, 退回原来的冻结梯度。
if speed_sq < grad_tol || denominator < 0.1 * speed_sq
    return;
end
ds_dp = c_prime' / denominator;
q_left_prime = [ppval(geometry.curves(1).dpp_x, s_near); ...
    ppval(geometry.curves(1).dpp_y, s_near)];
q_right_prime = [ppval(geometry.curves(2).dpp_x, s_near); ...
    ppval(geometry.curves(2).dpp_y, s_near)];
% ct = a/|a| with a = q_right - q_left, so dct/ds projects da/ds onto the
% subspace orthogonal to ct.
da_ds = q_right_prime - q_left_prime;
dcross_track_ds = (da_ds - cross_track * (cross_track' * da_ds)) / ...
    corridor_width;
end

function q = evaluate_curve(curve, s)
q = [ppval(curve.pp_x, s); ppval(curve.pp_y, s)];
end
