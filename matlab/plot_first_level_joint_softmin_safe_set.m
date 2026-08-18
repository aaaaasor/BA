function output_paths = plot_first_level_joint_softmin_safe_set(cfg)
%PLOT_FIRST_LEVEL_JOINT_SOFTMIN_SAFE_SET Plot h_soft >= 0 and h_soft < 0.
% The plotted h_soft uses the same first-level obstacle inflation, track
% margin, and soft-min kappa as the joint obstacle/boundary QP constraint.

output_paths = strings(0, 1);
if ~isfield(cfg, 'track_segment') || isempty(cfg.track_segment) || ...
        ~isfield(cfg, 'obstacle') || ...
        ~struct_field_default(cfg.obstacle, 'enabled', false) || ...
        ~isfield(cfg, 'track_boundary') || ...
        ~isfield(cfg.track_boundary, 'geometry') || ...
        ~isfield(cfg.track_boundary.geometry, 'implicit_fields')
    return;
end

variance_cfg = cfg.variance_constraint;
if ~struct_field_default(variance_cfg, ...
        'first_level_joint_safety_softmin_enabled', false)
    return;
end
kappa = struct_field_default(variance_cfg, ...
    'first_level_joint_safety_softmin_kappa', 2000.0);
if ~isscalar(kappa) || ~isfinite(kappa) || kappa <= 0
    error('First-level joint safety soft-min kappa must be positive and finite.');
end

% Use the exact obstacle geometry seen by the first-level QP.
obstacle = cfg.obstacle;
n_obstacles = size(obstacle.semi_axes, 2);
inflations = struct_field_default(obstacle, ...
    'constraint_inflations', zeros(1, n_obstacles));
if isscalar(inflations)
    inflations = repmat(inflations, 1, n_obstacles);
end
inflation_scale = struct_field_default(variance_cfg, ...
    'first_level_obstacle_constraint_inflation_scale', 0.0);
obstacle.semi_axes = obstacle.semi_axes + ...
    inflation_scale .* reshape(inflations, 1, []);

track = cfg.track_segment;
all_xy = [track.left; track.right; track.raceline];
x_pad = max(0.025, 0.04 * range(all_xy(:, 1)));
y_pad = max(0.025, 0.04 * range(all_xy(:, 2)));
x_limits = [min(all_xy(:, 1)) - x_pad, max(all_xy(:, 1)) + x_pad];
y_limits = [min(all_xy(:, 2)) - y_pad, max(all_xy(:, 2)) + y_pad];
x_grid = linspace(x_limits(1), x_limits(2), 500);
y_grid = linspace(y_limits(1), y_limits(2), 500);
[x_mesh, y_mesh] = meshgrid(x_grid, y_grid);

component_h = inf([size(x_mesh), n_obstacles + 2]);
for obstacle_idx = 1:n_obstacles
    component_h(:, :, obstacle_idx) = obstacle_level_grid( ...
        x_mesh, y_mesh, obstacle, obstacle_idx);
end

margin = struct_field_default(variance_cfg, ...
    'first_level_track_boundary_margin', ...
    struct_field_default(cfg.track_boundary, 'margin', 0.0));
fields = cfg.track_boundary.geometry.implicit_fields;
for boundary_idx = 1:2
    component_h(:, :, n_obstacles + boundary_idx) = ...
        implicit_field_grid(fields, boundary_idx, x_mesh, y_mesh) - margin;
end

% Stable soft minimum: -log(sum(exp(-kappa*h_i)))/kappa.
h_min = min(component_h, [], 3);
weight_sum = sum(exp(-kappa .* (component_h - h_min)), 3);
h_soft = h_min - log(weight_sum) ./ kappa;
safe_mask = h_soft >= 0;
unsafe_mask = ~safe_mask;

fig = figure('Name', 'First-level h_soft safe and unsafe sets', ...
    'Color', 'w', 'WindowStyle', 'normal', 'Units', 'normalized', ...
    'Position', [0.15, 0.10, 0.62, 0.78]);
movegui(fig, 'center');
ax = axes(fig);
hold(ax, 'on');
set(fig, 'CurrentAxes', ax);
draw_track_segment(track, 'HandleVisibility', 'off');

safe_color = [0.62, 0.88, 0.66];
unsafe_color = [0.88, 0.16, 0.16];
safe_rgb = repmat(reshape(safe_color, 1, 1, 3), size(x_mesh, 1), size(x_mesh, 2));
unsafe_rgb = repmat(reshape(unsafe_color, 1, 1, 3), size(x_mesh, 1), size(x_mesh, 2));
image(ax, x_grid, y_grid, safe_rgb, 'AlphaData', 0.38 .* safe_mask, ...
    'HandleVisibility', 'off');
image(ax, x_grid, y_grid, unsafe_rgb, 'AlphaData', 0.52 .* unsafe_mask, ...
    'HandleVisibility', 'off');
set(ax, 'YDir', 'normal');
[~, boundary_handle] = contour(ax, x_mesh, y_mesh, h_soft, [0, 0], ...
    'Color', [0.10, 0.10, 0.10], 'LineWidth', 1.5, ...
    'DisplayName', 'h_{soft}=0');
draw_obstacle_outlines(cfg.obstacle, ax);
plot(ax, nan, nan, 's', 'MarkerSize', 11, 'MarkerFaceColor', safe_color, ...
    'MarkerEdgeColor', safe_color, 'DisplayName', 'safe: h_{soft} \geq 0');
plot(ax, nan, nan, 's', 'MarkerSize', 11, 'MarkerFaceColor', unsafe_color, ...
    'MarkerEdgeColor', unsafe_color, 'DisplayName', 'unsafe: h_{soft} < 0');
uistack(boundary_handle, 'top');

axis(ax, 'equal');
xlim(ax, x_limits);
ylim(ax, y_limits);
xlabel(ax, 'x');
ylabel(ax, 'y');
title(ax, sprintf('First-level joint soft-min safety set (\\kappa = %.4g)', kappa));
legend(ax, 'Location', 'best');
grid(ax, 'off');

if struct_field_default(cfg.output, 'enabled', false)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    png_path = fullfile(output_dir, ...
        'FirstLevel_Joint_SoftMin_Safe_Unsafe_Set.png');
    exportgraphics(fig, png_path, 'Resolution', 220);
    output_paths = string(png_path);
    disp(['Saved first-level h_soft safe/unsafe set: ', png_path]);
end
end

function h = obstacle_level_grid(x, y, obstacle, obstacle_idx)
c = obstacle.centers(:, obstacle_idx);
a = obstacle.semi_axes(:, obstacle_idx);
angle = obstacle_value_grid(obstacle, 'angles', obstacle_idx, 0.0);
exponent = obstacle_value_grid(obstacle, 'exponents', obstacle_idx, 2.0);
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

function value = obstacle_value_grid(obstacle, field_name, obstacle_idx, default_value)
if ~isfield(obstacle, field_name) || isempty(obstacle.(field_name))
    value = default_value;
elseif isscalar(obstacle.(field_name))
    value = obstacle.(field_name);
else
    value = obstacle.(field_name)(obstacle_idx);
end
end

function h = implicit_field_grid(fields, boundary_idx, x, y)
% Vectorized form of evaluate_track_implicit_field, including off-grid
% linear extension from the nearest boundary cell.
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

function draw_obstacle_outlines(obstacle, ax)
% Keep obstacle interiors red so the unsafe-set color is not obscured.
for obstacle_idx = 1:size(obstacle.centers, 2)
    xy = obstacle_outline_points(obstacle, obstacle_idx);
    plot(ax, xy(:, 1), xy(:, 2), '--', 'Color', [0.30, 0.30, 0.30], ...
        'LineWidth', 1.0, 'HandleVisibility', 'off');
end
end
