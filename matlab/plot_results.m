function plot_results(cfg, traj_path, x_train, y_train)
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_path = fullfile(output_dir, 'gp_flow_matching_demo_matlab.png');

axis_grid = linspace(-5.0, 5.0, 180);
[grid_x, grid_y] = meshgrid(axis_grid, axis_grid);
source_density = source_pdf(grid_x, grid_y);
target_density = target_pdf(grid_x, grid_y, cfg.mixture);
source_levels = linspace(0.05 * max(source_density(:)), 0.95 * max(source_density(:)), 8);
target_levels = linspace(0.05 * max(target_density(:)), 0.95 * max(target_density(:)), 8);

fig = figure('Color', 'w', 'WindowStyle', 'normal', 'Units', 'normalized', 'Position', [0.10, 0.10, 0.75, 0.78]);
movegui(fig, 'center');

tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile([1 2]);
contour(grid_x, grid_y, source_density, source_levels, 'LineWidth', 1.3, 'LineColor', [0.45, 0.70, 0.98]);
hold on;
contour(grid_x, grid_y, target_density, target_levels, 'LineWidth', 1.1, 'LineColor', [0.98, 0.68, 0.40]);
grid on;
axis equal;
xlabel('x');
ylabel('y');
title('Source and Target Densities');
legend('Source Density', 'Target Density', 'Location', 'northwest');

nexttile;
max_curves = min(size(traj_path, 2), 40);
hold on;
for idx = 1:max_curves
    plot(traj_path(:, idx, 1), traj_path(:, idx, 2), 'LineWidth', 0.9);
end
grid on;
axis equal;
xlabel('x(t)');
ylabel('y(t)');
title('Sample Trajectories in 2D');

nexttile;
rng(cfg.random_seed);
n_quiver = min(450, size(x_train, 1));
quiver_idx = randperm(size(x_train, 1), n_quiver);
x_quiver = x_train(quiver_idx, :);
y_quiver = y_train(quiver_idx, :);
quiver(x_quiver(:, 2), x_quiver(:, 3), y_quiver(:, 1), y_quiver(:, 2), 3.5, ...
    'Color', [0.20, 0.45, 0.85], 'LineWidth', 0.8, 'MaxHeadSize', 0.6);
grid on;
axis equal;
x_pad = 0.6;
y_pad = 0.6;
xlim([min(x_train(:, 2)) - x_pad, max(x_train(:, 2)) + x_pad]);
ylim([min(x_train(:, 3)) - y_pad, max(x_train(:, 3)) + y_pad]);
xlabel('x_t');
ylabel('y_t');
title('Training Velocity Field Arrows');

exportgraphics(fig, output_path, 'Resolution', 180);
end
