function [joint_info, boundary_info] = joint_safety_softmin_info( ...
    obstacle_info, boundary_info, stats, constraint_cfg, t)
%JOINT_SAFETY_SOFTMIN_INFO Combine obstacle and track-boundary CBFs.
% For each controlled physical point, form
%   h = -log(sum_i exp(-kappa*h_i))/kappa.
% Since h <= min_i h_i, h >= 0 implies every component h_i >= 0.

enabled = struct_field_default(constraint_cfg, ...
    'joint_safety_softmin_enabled', false);
if ~enabled
    joint_info = obstacle_info;
    return;
end

kappa = struct_field_default(constraint_cfg, ...
    'joint_safety_softmin_kappa', 2000.0);
if ~isscalar(kappa) || ~isfinite(kappa) || kappa <= 0
    error('joint_safety_softmin_kappa must be finite and positive.');
end

phi0 = struct_field_default(constraint_cfg, ...
    'joint_safety_phi0', 15.0);
phi1_early = struct_field_default(constraint_cfg, ...
    'joint_safety_phi1_early_gain', 30.0);
phi1_switch_time = struct_field_default(constraint_cfg, ...
    'joint_safety_phi1_switch_time', 1.0);
omega = struct_field_default(constraint_cfg, ...
    'joint_safety_phi1_omega', 0.1);
phi1_max = struct_field_default(constraint_cfg, ...
    'joint_safety_phi1_max', inf);
if ~isscalar(phi1_max) || isnan(phi1_max) || phi1_max <= 0
    error('joint_safety_phi1_max must be positive or Inf.');
end
recovery_margin = struct_field_default(constraint_cfg, ...
    'joint_safety_recovery_margin', 0.0);
if ~isscalar(recovery_margin) || ~isfinite(recovery_margin) || ...
        recovery_margin < 0
    error('joint_safety_recovery_margin must be finite and nonnegative.');
end
% Geometric safety uses physical time directly, with no PTZF shift.
tau = max(1.0 - t, eps);
phi1 = min(omega / (tau ^ 2), phi1_max);
grad_tol = struct_field_default(constraint_cfg, 'grad_tol', 1e-6);

joint_info = obstacle_info;
joint_info.enabled = obstacle_info.enabled || boundary_info.enabled;
joint_info.A = zeros(0, numel(stats.mu));
joint_info.bounds = zeros(0, 1);
joint_info.h_values = zeros(0, 1);
joint_info.row_point_indices = zeros(0, 1);
joint_info.row_is_escape = false(0, 1);
joint_info.n_escape_rows = 0;
joint_info.active_candidate_indices = zeros(0, 1);
if isfield(joint_info, 'candidate_row_active')
    joint_info.candidate_row_active(:) = false;
end

obstacle_points = struct_field_default(obstacle_info, ...
    'row_point_indices', zeros(0, 1));
boundary_points = struct_field_default(boundary_info, ...
    'row_point_indices', zeros(0, 1));
point_indices = unique([obstacle_points(:); boundary_points(:)]);

for point_idx = reshape(point_indices, 1, [])
    obstacle_rows = find(obstacle_points == point_idx);
    boundary_rows = find(boundary_points == point_idx);
    component_h = [obstacle_info.h_values(obstacle_rows); ...
        boundary_info.h_values(boundary_rows)];
    component_grad = [-obstacle_info.A(obstacle_rows, :); ...
        -boundary_info.A(boundary_rows, :)];
    if isempty(component_h)
        continue;
    end

    % Stable log-sum-exp evaluation.  The boundary component is already a
    % soft minimum of its two global fields with the same kappa, so this is
    % exactly the five-way soft minimum (three obstacles plus two rails).
    h_min = min(component_h);
    exponentials = exp(-kappa .* (component_h - h_min));
    weight_sum = sum(exponentials);
    weights = exponentials ./ weight_sum;
    h_soft = h_min - log(weight_sum) / kappa;
    % Recover to the interior h_soft >= recovery_margin.  This makes the
    % original component functions cross zero before the PTCBF singularity
    % instead of approaching zero only from the unsafe side.
    h = h_soft - recovery_margin;
    grad = weights' * component_grad;
    if norm(grad) < grad_tol
        continue;
    end

    % The finite-kappa soft minimum is conservatively below min(h_i), so it
    % can remain slightly negative after every original safety function is
    % already nonnegative.  In that case the physical conjunction is safe
    % and must leave the singular recovery branch immediately.
    components_safe = all(component_h >= 0);
    if components_safe
        phi = phi0;
    elseif t < phi1_switch_time
        phi = phi1_early;
    else
        phi = phi1;
    end
    joint_info.A(end + 1, :) = -grad;
    joint_info.bounds(end + 1, 1) = grad * stats.mu(:) + phi * h;
    joint_info.h_values(end + 1, 1) = h;
    joint_info.row_point_indices(end + 1, 1) = point_idx;
    joint_info.row_is_escape(end + 1, 1) = false;
end

joint_info.n_rows = size(joint_info.A, 1);
if isempty(joint_info.h_values)
    joint_info.h_min = inf;
    joint_info.max_residual_without_u = -inf;
else
    joint_info.h_min = min(joint_info.h_values);
    joint_info.max_residual_without_u = max(-joint_info.bounds);
end

% The joint rows replace, rather than supplement, the separate boundary
% rows.  This is what removes opposing hard half-spaces from the QP.
boundary_info.enabled = false;
boundary_info.A = zeros(0, numel(stats.mu));
boundary_info.bounds = zeros(0, 1);
boundary_info.h_values = zeros(0, 1);
end
