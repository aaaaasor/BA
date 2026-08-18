function h_soft = draw_joint_softmin_safe_set_background(ax, constraint_cfg, track)
%DRAW_JOINT_SOFTMIN_SAFE_SET_BACKGROUND Draw static red/green h_soft sets.
% Safe: h_soft >= 0 (transparent light green). Unsafe: h_soft < 0 (red).

if isempty(track) || ~isfield(constraint_cfg, 'obstacle_geometry')
    draw_track_segment(track, 'HandleVisibility', 'off');
    h_soft = [];
    return;
end
if isfield(constraint_cfg, 'track_boundary_geometry') && ...
        isfield(constraint_cfg.track_boundary_geometry, 'implicit_fields')
    boundary_geometry = constraint_cfg.track_boundary_geometry;
else
    % Compatibility for rollout caches created before the fixed implicit
    % boundary geometry was stored in the constraint structure.
    boundary_geometry = build_track_boundary_geometry(track, 400, 'spline');
end

all_xy = [track.left; track.right; track.raceline];
x_pad = max(0.025, 0.04 * (max(all_xy(:, 1)) - min(all_xy(:, 1))));
y_pad = max(0.025, 0.04 * (max(all_xy(:, 2)) - min(all_xy(:, 2))));
x_limits = [min(all_xy(:, 1)) - x_pad, max(all_xy(:, 1)) + x_pad];
y_limits = [min(all_xy(:, 2)) - y_pad, max(all_xy(:, 2)) + y_pad];
x_grid = linspace(x_limits(1), x_limits(2), 500);
y_grid = linspace(y_limits(1), y_limits(2), 500);
[x_mesh, y_mesh] = meshgrid(x_grid, y_grid);

obstacle = constraint_cfg.obstacle_geometry;
n_obstacles = size(obstacle.semi_axes, 2);
component_h = inf([size(x_mesh), n_obstacles + 2]);
for obstacle_idx = 1:n_obstacles
    component_h(:, :, obstacle_idx) = obstacle_level_grid( ...
        x_mesh, y_mesh, obstacle, obstacle_idx);
end
fields = boundary_geometry.implicit_fields;
margin = struct_field_default(constraint_cfg, 'track_boundary_margin', 0.0);
for boundary_idx = 1:2
    component_h(:, :, n_obstacles + boundary_idx) = ...
        implicit_field_grid(fields, boundary_idx, x_mesh, y_mesh) - margin;
end
kappa = struct_field_default(constraint_cfg, ...
    'joint_safety_softmin_kappa', 2000.0);
h_min = min(component_h, [], 3);
h_soft = h_min - log(sum(exp(-kappa .* (component_h - h_min)), 3)) ./ kappa;
safe_mask = h_soft >= 0;

draw_track_segment(track, 'HandleVisibility', 'off');
safe_color = [0.62, 0.88, 0.66];
unsafe_color = [0.88, 0.16, 0.16];
safe_rgb = repmat(reshape(safe_color, 1, 1, 3), size(x_mesh, 1), size(x_mesh, 2));
unsafe_rgb = repmat(reshape(unsafe_color, 1, 1, 3), size(x_mesh, 1), size(x_mesh, 2));
image(ax, x_grid, y_grid, safe_rgb, 'AlphaData', 0.38 .* safe_mask, ...
    'HandleVisibility', 'off');
image(ax, x_grid, y_grid, unsafe_rgb, 'AlphaData', 0.52 .* ~safe_mask, ...
    'HandleVisibility', 'off');
set(ax, 'YDir', 'normal');
contour(ax, x_mesh, y_mesh, h_soft, [0, 0], ...
    'Color', [0.10, 0.10, 0.10], 'LineWidth', 1.35, ...
    'HandleVisibility', 'off');

physical_obstacle = struct_field_default(constraint_cfg, ...
    'obstacle_physical_geometry', obstacle);
for obstacle_idx = 1:size(physical_obstacle.centers, 2)
    xy = obstacle_outline_points(physical_obstacle, obstacle_idx);
    plot(ax, xy(:, 1), xy(:, 2), '--', 'Color', [0.30, 0.30, 0.30], ...
        'LineWidth', 1.0, 'HandleVisibility', 'off');
end
axis(ax, 'equal');
xlim(ax, x_limits);
ylim(ax, y_limits);
end

function h = obstacle_level_grid(x, y, obstacle, obstacle_idx)
c = obstacle.centers(:, obstacle_idx);
a = obstacle.semi_axes(:, obstacle_idx);
angle = geometry_value(obstacle, 'angles', obstacle_idx, 0.0);
exponent = geometry_value(obstacle, 'exponents', obstacle_idx, 2.0);
dx = x - c(1);
dy = y - c(2);
q_x = cos(angle) .* dx + sin(angle) .* dy;
q_y = -sin(angle) .* dx + cos(angle) .* dy;
power_sum = abs(q_x ./ a(1)) .^ exponent + ...
    abs(q_y ./ a(2)) .^ exponent;
if exponent > 2
    h = power_sum .^ (1 / exponent) - 1.0;
else
    h = power_sum - 1.0;
end
end

function value = geometry_value(geometry, field_name, idx, default_value)
if ~isfield(geometry, field_name) || isempty(geometry.(field_name))
    value = default_value;
elseif isscalar(geometry.(field_name))
    value = geometry.(field_name);
else
    value = geometry.(field_name)(idx);
end
end

function h = implicit_field_grid(fields, boundary_idx, x, y)
x_max = fields.x_min + fields.dx * (fields.n_x - 1);
y_max = fields.y_min + fields.dy * (fields.n_y - 1);
x_clamped = min(max(x, fields.x_min), x_max);
y_clamped = min(max(y, fields.y_min), y_max);
x_coordinate = (x_clamped - fields.x_min) ./ fields.dx + 1;
y_coordinate = (y_clamped - fields.y_min) ./ fields.dy + 1;
x_idx = min(max(floor(x_coordinate), 1), fields.n_x - 1);
y_idx = min(max(floor(y_coordinate), 1), fields.n_y - 1);
x_fraction = x_coordinate - x_idx;
y_fraction = y_coordinate - y_idx;
grid_values = fields.h{boundary_idx};
idx_00 = sub2ind(size(grid_values), y_idx, x_idx);
idx_10 = sub2ind(size(grid_values), y_idx, x_idx + 1);
idx_01 = sub2ind(size(grid_values), y_idx + 1, x_idx);
idx_11 = sub2ind(size(grid_values), y_idx + 1, x_idx + 1);
h_00 = grid_values(idx_00);
h_10 = grid_values(idx_10);
h_01 = grid_values(idx_01);
h_11 = grid_values(idx_11);
h = (1 - x_fraction) .* (1 - y_fraction) .* h_00 + ...
    x_fraction .* (1 - y_fraction) .* h_10 + ...
    (1 - x_fraction) .* y_fraction .* h_01 + ...
    x_fraction .* y_fraction .* h_11;
grad_x = ((1 - y_fraction) .* (h_10 - h_00) + ...
    y_fraction .* (h_11 - h_01)) ./ fields.dx;
grad_y = ((1 - x_fraction) .* (h_01 - h_00) + ...
    x_fraction .* (h_11 - h_10)) ./ fields.dy;
h = h + grad_x .* (x - x_clamped) + grad_y .* (y - y_clamped);
end
