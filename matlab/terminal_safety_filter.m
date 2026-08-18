function [x_safe, diagnostics] = terminal_safety_filter(x_terminal, constraint_cfg)
%TERMINAL_SAFETY_FILTER Project an unsafe terminal state into the safe set.
% This implements the SafeFlow terminal filter
%   min_x 1/2 ||x-x_terminal||^2
% subject to every enabled terminal boundary/obstacle level h(x) >= 0.
% Safe inputs are returned exactly, so the filter does not perturb an
% already feasible rollout.

x_terminal = x_terminal(:);
tolerance = struct_field_default(constraint_cfg, ...
    'terminal_safety_filter_tolerance', 1e-8);
max_iterations = struct_field_default(constraint_cfg, ...
    'terminal_safety_filter_max_iterations', 200);
if ~isscalar(tolerance) || ~isfinite(tolerance) || tolerance <= 0
    error('terminal_safety_filter_tolerance must be finite and positive.');
end
if ~isscalar(max_iterations) || ~isfinite(max_iterations) || ...
        max_iterations < 1 || max_iterations ~= round(max_iterations)
    error(['terminal_safety_filter_max_iterations must be a positive ', ...
        'integer.']);
end

[initial_c, ~] = safety_constraints(x_terminal, constraint_cfg);
initial_violation = positive_max(initial_c);
diagnostics = struct( ...
    'applied', false, ...
    'initial_max_violation', initial_violation, ...
    'final_max_violation', initial_violation, ...
    'state_correction_norm', 0.0, ...
    'exitflag', 1, ...
    'iterations', 0);
x_safe = x_terminal;
if isempty(initial_c) || initial_violation <= tolerance
    return;
end
if exist('fmincon', 'file') ~= 2
    error('terminal_safety_filter:MissingFmincon', ...
        'Terminal safety projection requires Optimization Toolbox fmincon.');
end

projection_metric = build_projection_metric(constraint_cfg, numel(x_terminal));
objective = @(x) projection_objective(x, x_terminal, projection_metric);
nonlinear_constraints = @(x) safety_constraints(x, constraint_cfg);
x_start = build_solver_start(x_terminal, constraint_cfg, tolerance);
options = optimoptions('fmincon', ...
    'Algorithm', 'sqp', ...
    'Display', 'off', ...
    'SpecifyObjectiveGradient', true, ...
    'SpecifyConstraintGradient', true, ...
    'ConstraintTolerance', tolerance, ...
    'OptimalityTolerance', tolerance, ...
    'StepTolerance', min(tolerance, 1e-10), ...
    'MaxIterations', max_iterations, ...
    'MaxFunctionEvaluations', max(2000, 50 * numel(x_terminal)));
[candidate, ~, exitflag, output] = fmincon(objective, x_start, ...
    [], [], [], [], [], [], nonlinear_constraints, options);
[final_c, ~] = safety_constraints(candidate, constraint_cfg);
final_violation = positive_max(final_c);
if exitflag <= 0 || ~all(isfinite(candidate)) || ...
        final_violation > 10 * tolerance
    error('terminal_safety_filter:ProjectionFailed', ...
        ['Terminal projection failed: exitflag=%d, initial violation=%.6g, ', ...
        'final violation=%.6g.'], exitflag, initial_violation, final_violation);
end

x_safe = candidate;
diagnostics.applied = true;
diagnostics.final_max_violation = final_violation;
diagnostics.state_correction_norm = norm(candidate - x_terminal);
diagnostics.exitflag = exitflag;
diagnostics.iterations = output.iterations;
end

function x_start = build_solver_start(x_reference, cfg, tolerance)
% At the exact center of a superellipse its analytical gradient is zero.
% Move only the optimizer's initial guess along the configured escape axis;
% the projection objective is still measured from the original terminal
% state, so this does not change the optimization problem.
x_start = x_reference;
if ~struct_field_default(cfg, 'obstacle_enabled', false) || ...
        ~isfield(cfg, 'obstacle_geometry') || ...
        ~isfield(cfg, 'obstacle_point_maps')
    return;
end
obstacle = cfg.obstacle_geometry;
n_obstacles = size(obstacle.centers, 2);
for map_idx = 1:numel(cfg.obstacle_point_maps)
    point_map = cfg.obstacle_point_maps(map_idx);
    for obstacle_idx = 1:n_obstacles
        p = point_map.M * x_start + point_map.o;
        [h, grad_p, escape_direction] = obstacle_level_and_gradient( ...
            p, obstacle, obstacle_idx);
        if h >= 0 || norm(grad_p) > 1e-10
            continue;
        end
        semi_axes = obstacle.semi_axes(:, obstacle_idx);
        center = obstacle.centers(:, obstacle_idx);
        target_p = center + ...
            (min(semi_axes) + 10 * tolerance) * escape_direction;
        map_gram = point_map.M * point_map.M';
        if rcond(map_gram) > 1e-12
            x_start = x_start + point_map.M' * ...
                (map_gram \ (target_p - p));
        end
    end
end
end

function [value, gradient] = projection_objective(x, x_reference, metric)
delta = x(:) - x_reference;
value = 0.5 * (delta' * metric * delta);
gradient = metric * delta;
end

function metric = build_projection_metric(cfg, n_x)
% Minimize displacement of the physical trajectory points, rather than the
% standardized GP coordinates. Duplicate boundary/obstacle maps for the
% same point are counted only once. A tiny state penalty makes the metric
% positive definite and leaves unconstrained tangent features unchanged.
maps = struct('point_index', {}, 'M', {});
if struct_field_default(cfg, 'track_boundary_enabled', false) && ...
        isfield(cfg, 'track_boundary_point_maps')
    source = cfg.track_boundary_point_maps;
    for idx = 1:numel(source)
        maps(end + 1) = struct('point_index', source(idx).point_index, ...
            'M', source(idx).M); %#ok<AGROW>
    end
end
if struct_field_default(cfg, 'obstacle_enabled', false) && ...
        isfield(cfg, 'obstacle_point_maps')
    source = cfg.obstacle_point_maps;
    existing_indices = [maps.point_index];
    for idx = 1:numel(source)
        if ~ismember(source(idx).point_index, existing_indices)
            maps(end + 1) = struct( ...
                'point_index', source(idx).point_index, ...
                'M', source(idx).M); %#ok<AGROW>
            existing_indices(end + 1) = source(idx).point_index; %#ok<AGROW>
        end
    end
end
if isempty(maps)
    metric = eye(n_x);
    return;
end
physical_map = vertcat(maps.M);
metric = physical_map' * physical_map;
metric_scale = max(trace(metric) / max(n_x, 1), 1.0);
metric = metric + (1e-12 * metric_scale) * eye(n_x);
end

function [c, ceq, gradient_c, gradient_ceq] = safety_constraints(x, cfg)
% fmincon uses c <= 0, whereas every safety function uses h >= 0.
x = x(:);
n_x = numel(x);
c = zeros(0, 1);
gradient_c = zeros(n_x, 0);
ceq = zeros(0, 1);
gradient_ceq = zeros(n_x, 0);

if struct_field_default(cfg, 'track_boundary_enabled', false)
    if ~isfield(cfg, 'track_boundary_geometry') || ...
            ~isfield(cfg.track_boundary_geometry, 'implicit_fields') || ...
            ~isfield(cfg, 'track_boundary_point_maps')
        error(['Terminal boundary projection requires global implicit ', ...
            'fields and track_boundary_point_maps.']);
    end
    method = lower(string(struct_field_default(cfg, ...
        'track_boundary_constraint_method', 'paired_cross_section')));
    if method ~= "global_implicit_fields"
        error(['Terminal safety projection currently requires ', ...
            'track_boundary_constraint_method=''global_implicit_fields''.']);
    end
    fields = cfg.track_boundary_geometry.implicit_fields;
    margin = struct_field_default(cfg, 'track_boundary_margin', 0.0);
    point_maps = cfg.track_boundary_point_maps;
    for map_idx = 1:numel(point_maps)
        point_map = point_maps(map_idx);
        p = point_map.M * x + point_map.o;
        for boundary_idx = 1:2
            [h_raw, grad_p] = evaluate_track_implicit_field( ...
                fields, boundary_idx, p);
            h = h_raw - margin;
            c(end + 1, 1) = -h; %#ok<AGROW>
            gradient_c(:, end + 1) = ...
                -(grad_p * point_map.M)'; %#ok<AGROW>
        end
    end
end

if struct_field_default(cfg, 'obstacle_enabled', false)
    if ~isfield(cfg, 'obstacle_geometry') || ...
            ~isfield(cfg, 'obstacle_point_maps')
        error(['Terminal obstacle projection requires obstacle_geometry ', ...
            'and obstacle_point_maps.']);
    end
    obstacle = cfg.obstacle_geometry;
    point_maps = cfg.obstacle_point_maps;
    n_obstacles = size(obstacle.centers, 2);
    for map_idx = 1:numel(point_maps)
        point_map = point_maps(map_idx);
        p = point_map.M * x + point_map.o;
        for obstacle_idx = 1:n_obstacles
            [h, grad_p] = obstacle_level_and_gradient( ...
                p, obstacle, obstacle_idx);
            c(end + 1, 1) = -h; %#ok<AGROW>
            gradient_c(:, end + 1) = ...
                -(grad_p' * point_map.M)'; %#ok<AGROW>
        end
    end
end
end

function value = positive_max(values)
if isempty(values)
    value = 0.0;
else
    value = max(max(values), 0.0);
end
end
