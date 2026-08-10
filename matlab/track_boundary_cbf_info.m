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
phase_search_half_steps = struct_field_default(constraint_cfg, ...
    'track_boundary_phase_search_half_steps', 2.0);
phase_lock_weight = struct_field_default(constraint_cfg, ...
    'track_boundary_phase_lock_weight', 1.0);
if ~isscalar(phase_search_half_steps) || ~isfinite(phase_search_half_steps) || ...
        phase_search_half_steps <= 0
    error('track_boundary_phase_search_half_steps must be finite and positive.');
end
if ~isscalar(phase_lock_weight) || ~isfinite(phase_lock_weight) || ...
        phase_lock_weight < 0
    error('track_boundary_phase_lock_weight must be finite and nonnegative.');
end
phi0 = struct_field_default(constraint_cfg, ...
    'track_boundary_phi0', 5.0);
phi1_early = struct_field_default(constraint_cfg, ...
    'track_boundary_phi1_early_gain', phi0);
phi1_switch_time = struct_field_default(constraint_cfg, ...
    'track_boundary_phi1_switch_time', 0.85);
omega = struct_field_default(constraint_cfg, ...
    'track_boundary_phi1_omega', 0.1);
tau_min = struct_field_default(constraint_cfg, ...
    'track_boundary_phi1_tau_min', 0.01);
margin = struct_field_default(constraint_cfg, 'track_boundary_margin', 0.0);
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
if ~isscalar(tau_min) || ~isfinite(tau_min) || tau_min <= 0
    error('track_boundary_phi1_tau_min must be finite and positive.');
end
if ~isscalar(margin) || ~isfinite(margin) || margin < 0
    error('track_boundary_margin must be finite and nonnegative.');
end
time_shift = struct_field_default(constraint_cfg, 'ptzf_time_shift', 0.0);
t_eff = t + time_shift;
boundary_slack_configured = struct_field_default(constraint_cfg, ...
    'track_boundary_slack_enabled', false);
boundary_hard_after_time = struct_field_default(constraint_cfg, ...
    'track_boundary_slack_hard_after_time', inf);
boundary_slack_active = boundary_slack_configured && ...
    t < boundary_hard_after_time;
if boundary_slack_active
    % Soft boundary: regularize RK4's exact endpoint evaluation.
    tau = max(1.0 - t_eff, tau_min);
else
    % Hard boundary: retain the uncapped prescribed-time gain.  eps only
    % prevents division by literal zero and is not a practical gain cap.
    tau = max(1.0 - t_eff, eps);
end
phi1 = omega / (tau ^ 2);

for map_idx = 1:numel(point_maps)
    point_map = point_maps(map_idx);
    p = point_map.M * stats.x(:) + point_map.o;
    s_reference = 0.5 * (reference_s_min(map_idx) + ...
        reference_s_max(map_idx));
    reference_half_width = max(0.5 * (reference_s_max(map_idx) - ...
        reference_s_min(map_idx)), eps);
    search_half_width = phase_search_half_steps * 2.0 * ...
        reference_half_width;
    s_near = branch_locked_center_parameter(p, geometry, s_reference, ...
        search_half_width, phase_lock_weight);
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
    pair_h = zeros(2, 1);
    for boundary_idx = 1:2
        pair_h(boundary_idx) = inward_normals(:, boundary_idx)' * ...
            (p - boundary_points(:, boundary_idx)) - margin;
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
        inward_normal = inward_normals(:, boundary_idx);
        h = pair_h(boundary_idx);
        grad = inward_normal' * point_map.M;
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

function s_near = branch_locked_center_parameter(p, geometry, s_reference, ...
    search_half_width, phase_lock_weight)
% Locate the closest point on the inherited branch.  Unlike a hard lookup
% window, the phase prior is part of the objective: its ends are not used as
% physical corridor faces, while nearby folded branches remain disfavored.
s_reference = max(0.0, min(1.0, s_reference));
s_min = max(0.0, s_reference - search_half_width);
s_max = min(1.0, s_reference + search_half_width);
all_s = geometry.s_control;
in_window = all_s >= s_min & all_s <= s_max;
s_grid = all_s(in_window);
if numel(s_grid) < 2
    s_grid = linspace(s_min, s_max, 3);
end
x_grid = ppval(geometry.center_pp_x, s_grid);
y_grid = ppval(geometry.center_pp_y, s_grid);
local_ds = max(min(1e-3, 0.25 * search_half_width), eps);
s_lo = max(0.0, s_reference - local_ds);
s_hi = min(1.0, s_reference + local_ds);
if s_hi > s_lo
    local_speed = norm([ ...
        ppval(geometry.center_pp_x, s_hi) - ...
            ppval(geometry.center_pp_x, s_lo); ...
        ppval(geometry.center_pp_y, s_hi) - ...
            ppval(geometry.center_pp_y, s_lo)]) / (s_hi - s_lo);
else
    local_speed = 1.0;
end
phase_scale_sq = max(local_speed ^ 2, eps);
grid_objective = (x_grid - p(1)) .^ 2 + (y_grid - p(2)) .^ 2 + ...
    phase_lock_weight * phase_scale_sq * (s_grid - s_reference) .^ 2;
[~, nearest_idx] = min(grid_objective);
lo = max(s_min, s_grid(max(nearest_idx - 1, 1)));
hi = min(s_max, s_grid(min(nearest_idx + 1, numel(s_grid))));
distance_sq = @(s) (ppval(geometry.center_pp_x, s) - p(1)) .^ 2 + ...
    (ppval(geometry.center_pp_y, s) - p(2)) .^ 2 + ...
    phase_lock_weight * phase_scale_sq * (s - s_reference) .^ 2;
if hi > lo
    s_near = fminbnd(distance_sq, lo, hi);
else
    s_near = lo;
end
end

function q = evaluate_curve(curve, s)
q = [ppval(curve.pp_x, s); ppval(curve.pp_y, s)];
end
