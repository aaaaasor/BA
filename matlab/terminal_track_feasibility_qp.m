function [segment_data, diagnostics] = terminal_track_feasibility_qp( ...
    segment_data, feature_dim, n_points, constraint_cfg)
%TERMINAL_TRACK_FEASIBILITY_QP Correct only truly unsafe terminal points.
% P1/P5 and tangent features remain unchanged. For each unsafe P2/P3/P4,
% enumerate fine convex cells on its inherited branch, solve a minimum-move
% QP in every cell, and retain the closest candidate that passes independent
% dense-PCHIP track and exact obstacle checks. No safety margin is used.

required = {'track_boundary_geometry', 'track_boundary_point_maps', ...
    'track_boundary_reference_s_min_targets', ...
    'track_boundary_reference_s_max_targets'};
if ~all(isfield(constraint_cfg, required))
    error('Terminal feasibility QP requires track geometry and reference windows.');
end
point_indices = [constraint_cfg.track_boundary_point_maps.point_index];
if ~isequal(point_indices, 2:(n_points - 1))
    error('Terminal feasibility QP expects internal points P2..P%d.', n_points - 1);
end
if size(segment_data, 2) ~= feature_dim * n_points
    error('Terminal feasibility QP segment width is inconsistent.');
end

geometry = constraint_cfg.track_boundary_geometry;
s_min_targets = constraint_cfg.track_boundary_reference_s_min_targets;
s_max_targets = constraint_cfg.track_boundary_reference_s_max_targets;
n_segments = size(segment_data, 1);
n_internal = numel(point_indices);
if size(s_min_targets, 1) ~= n_segments || size(s_min_targets, 2) ~= n_internal
    error('Terminal feasibility QP reference-window size is inconsistent.');
end

obstacle_enabled = struct_field_default(constraint_cfg, ...
    'obstacle_enabled', false) && isfield(constraint_cfg, 'obstacle_geometry');
if obstacle_enabled
    obstacle = constraint_cfg.obstacle_geometry;
    n_obstacles = size(obstacle.centers, 2);
else
    obstacle = struct();
    n_obstacles = 0;
end

% Independent acceptance polygon; much denser than the QP cell grid.
s_dense = linspace(0.0, 1.0, 20000);
left_dense = evaluate_boundary_many(geometry.curves(1), s_dense);
right_dense = evaluate_boundary_many(geometry.curves(2), s_dense);
track_x = [left_dense(:, 1); flipud(right_dense(:, 1)); left_dense(1, 1)];
track_y = [left_dense(:, 2); flipud(right_dense(:, 2)); left_dense(1, 2)];

options = optimoptions('quadprog', 'Display', 'off', ...
    'Algorithm', 'active-set', 'ConstraintTolerance', 1e-9, ...
    'OptimalityTolerance', 1e-10, 'MaxIterations', 500);
n_subcells = 64;
search_half_steps = struct_field_default(constraint_cfg, ...
    'track_boundary_phase_search_half_steps', 2.0);
diagnostics.corrected_segments = 0;
diagnostics.corrected_points = 0;
diagnostics.max_point_move = 0.0;
diagnostics.max_endpoint_change = 0.0;

for sample_idx = 1:n_segments
    curve_before = reshape(segment_data(sample_idx, :), feature_dim, [])';
    curve_after = curve_before;
    segment_changed = false;
    for internal_idx = 1:n_internal
        point_idx = point_indices(internal_idx);
        p0 = curve_before(point_idx, 1:2)';
        if point_is_safe(p0, track_x, track_y, obstacle, n_obstacles)
            continue;
        end

        reference_min = s_min_targets(sample_idx, internal_idx);
        reference_max = s_max_targets(sample_idx, internal_idx);
        s_reference = 0.5 * (reference_min + reference_max);
        reference_half_width = max(0.5 * (reference_max - reference_min), eps);
        search_half_width = search_half_steps * 2.0 * reference_half_width;
        search_min = max(0.0, s_reference - search_half_width);
        search_max = min(1.0, s_reference + search_half_width);
        s_edges = linspace(search_min, search_max, n_subcells + 1);

        best_point = [];
        best_distance_sq = inf;
        for cell_idx = 1:n_subcells
            s0 = s_edges(cell_idx);
            s1 = s_edges(cell_idx + 1);
            vertices = [ ...
                evaluate_boundary(geometry.curves(1), s0)'; ...
                evaluate_boundary(geometry.curves(2), s0)'; ...
                evaluate_boundary(geometry.curves(2), s1)'; ...
                evaluate_boundary(geometry.curves(1), s1)'];
            [A_cell, b_cell] = convex_polygon_halfspaces(vertices);
            [candidate, exitflag] = project_point_qp( ...
                p0, A_cell, b_cell, obstacle, n_obstacles, options);
            if exitflag <= 0 || isempty(candidate) || ...
                    ~point_is_safe(candidate, track_x, track_y, ...
                    obstacle, n_obstacles)
                continue;
            end
            distance_sq = sum((candidate - p0) .^ 2);
            if distance_sq < best_distance_sq
                best_distance_sq = distance_sq;
                best_point = candidate;
            end
        end
        if isempty(best_point)
            error(['Terminal feasibility QP found no safe local-cell ', ...
                'candidate for segment %d P%d.'], sample_idx, point_idx);
        end
        curve_after(point_idx, 1:2) = best_point';
        move = norm(best_point - p0);
        diagnostics.corrected_points = diagnostics.corrected_points + 1;
        diagnostics.max_point_move = max(diagnostics.max_point_move, move);
        segment_changed = true;
    end
    if segment_changed
        diagnostics.corrected_segments = diagnostics.corrected_segments + 1;
    end
    diagnostics.max_endpoint_change = max(diagnostics.max_endpoint_change, ...
        max(abs([curve_after(1, 1:2) - curve_before(1, 1:2), ...
        curve_after(end, 1:2) - curve_before(end, 1:2)])));
    segment_data(sample_idx, :) = reshape(curve_after', 1, []);
end
end

function [p, exitflag] = project_point_qp(p0, A, b, obstacle, n_obstacles, options)
[p, ~, exitflag] = quadprog(eye(2), -p0, A, b, [], [], [], [], p0, options);
if exitflag <= 0 || isempty(p)
    return;
end
for cut_iteration = 1:10
    added_cut = false;
    for obstacle_idx = 1:n_obstacles
        [h_now, grad_now, escape_direction] = ...
            obstacle_level_and_gradient(p, obstacle, obstacle_idx);
        if h_now >= -1e-10
            continue;
        end
        if norm(grad_now) < 1e-12
            grad_now = escape_direction;
        end
        A = [A; -grad_now']; %#ok<AGROW>
        b = [b; h_now - grad_now' * p]; %#ok<AGROW>
        added_cut = true;
    end
    if ~added_cut
        return;
    end
    [p, ~, exitflag] = quadprog(eye(2), -p0, A, b, [], [], [], [], p, options);
    if exitflag <= 0 || isempty(p)
        return;
    end
end
end

function safe = point_is_safe(p, track_x, track_y, obstacle, n_obstacles)
[inside, on] = inpolygon(p(1), p(2), track_x, track_y);
safe = inside || on;
if ~safe
    return;
end
for obstacle_idx = 1:n_obstacles
    if obstacle_level_and_gradient(p, obstacle, obstacle_idx) < -1e-9
        safe = false;
        return;
    end
end
end

function p = evaluate_boundary(curve, s)
p = [ppval(curve.pp_x, s); ppval(curve.pp_y, s)];
end

function points = evaluate_boundary_many(curve, s)
points = [ppval(curve.pp_x, s(:)')', ppval(curve.pp_y, s(:)')'];
end

function [A, b] = convex_polygon_halfspaces(vertices)
hull = convhull(vertices(:, 1), vertices(:, 2));
vertices = vertices(hull, :);
n_edges = size(vertices, 1) - 1;
A = zeros(n_edges, 2);
b = zeros(n_edges, 1);
for edge_idx = 1:n_edges
    v0 = vertices(edge_idx, :)';
    v1 = vertices(edge_idx + 1, :)';
    edge = v1 - v0;
    A(edge_idx, :) = [edge(2), -edge(1)];
    b(edge_idx) = A(edge_idx, :) * v0;
end
end
