function plot_results(cfg, generated, traj_path, x_train, y_train)
axis_grid = linspace(-5.0, 5.0, 180);
[grid_x, grid_y] = meshgrid(axis_grid, axis_grid);
source_density = source_pdf(grid_x, grid_y);
target_density = target_pdf(grid_x, grid_y, cfg.mixture);
source_levels = linspace(0.05 * max(source_density(:)), 0.95 * max(source_density(:)), 8);
target_levels = linspace(0.05 * max(target_density(:)), 0.95 * max(target_density(:)), 8);

fig = figure('Color', 'w', 'WindowStyle', 'normal', 'Units', 'normalized', 'Position', [0.10, 0.10, 0.75, 0.78]);
movegui(fig, 'center');

subplot(2, 2, 1);
contour(grid_x, grid_y, source_density, source_levels, 'LineWidth', 1.3, 'LineColor', [0.45, 0.70, 0.98]);
hold on;
contour(grid_x, grid_y, target_density, target_levels, 'LineWidth', 1.1, 'LineColor', [0.98, 0.68, 0.40]);
grid on;
axis equal;
xlabel('x');
ylabel('y');
title('Source and Target Densities');
legend('Source Density', 'Target Density', 'Location', 'northwest');

subplot(2, 2, 2);
scatter(generated(:, 1), generated(:, 2), 18, 'filled', 'MarkerFaceColor', [0.35, 0.62, 0.90], 'MarkerFaceAlpha', 0.35, 'MarkerEdgeAlpha', 0.20);
hold on;
contour(grid_x, grid_y, target_density, target_levels, 'LineWidth', 1.1, 'LineColor', [0.98, 0.68, 0.40]);
grid on;
axis equal;
xlabel('x');
ylabel('y');
title('Generated Samples vs Target');
legend('Generated', 'Target Density', 'Location', 'northwest');

subplot(2, 2, 3);
max_curves = min(size(traj_path, 2), 20);
hold on;
for idx = 1:max_curves
    plot(traj_path(:, idx, 1), traj_path(:, idx, 2), 'LineWidth', 0.9);
end
grid on;
axis equal;
xlabel('x(t)');
ylabel('y(t)');
title('Sample Trajectories in 2D');

subplot(2, 2, 4);
rng(cfg.random_seed);
n_plot = min(3500, size(x_train, 1));
sample_idx = randperm(size(x_train, 1), n_plot);
x_plot = x_train(sample_idx, :);
y_plot = y_train(sample_idx, :);
speed = sqrt(sum(y_plot .^ 2, 2));
scatter(x_plot(:, 2), x_plot(:, 3), 12, speed, 'filled', 'MarkerFaceAlpha', 0.38, 'MarkerEdgeAlpha', 0.18);
grid on;
axis equal;
xlabel('x_t');
ylabel('y_t');
title('Training States Colored by |v|');
colorbar;
end
