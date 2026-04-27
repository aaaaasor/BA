function plot_results(cfg, generated, traj_times, traj_path, x_train, y_train)
x_grid = linspace(-5.0, 5.0, 500)';

fig = figure('Color', 'w', 'WindowStyle', 'normal', 'Units', 'normalized', 'Position', [0.12, 0.12, 0.72, 0.72]);
movegui(fig, 'center');

subplot(2, 2, 1);
plot(x_grid, source_pdf(x_grid), 'LineWidth', 2);
hold on;
plot(x_grid, target_pdf(x_grid, cfg.mixture), 'LineWidth', 2);
grid on;
legend('Source PDF', 'Target PDF', 'Location', 'best');
title('Source and Target Distributions');

subplot(2, 2, 2);
histogram(generated, 40, 'Normalization', 'pdf', 'FaceAlpha', 0.65);
hold on;
plot(x_grid, target_pdf(x_grid, cfg.mixture), 'LineWidth', 2);
grid on;
legend('Generated', 'Target PDF', 'Location', 'best');
title('Generated vs Target');

subplot(2, 2, 3);
max_curves = min(size(traj_path, 2), 20);
plot(traj_times, traj_path(:, 1:max_curves), 'LineWidth', 1.0);
grid on;
xlabel('t');
ylabel('x(t)');
title('Sample Trajectories');
y_min = min(min(traj_path(:, 1:max_curves)));
y_max = max(max(traj_path(:, 1:max_curves)));
ylim([y_min - 0.35, y_max + 0.35]);

subplot(2, 2, 4);
rng(7);
n_plot = min(3500, size(x_train, 1));
sample_idx = randperm(size(x_train, 1), n_plot);
x_plot = x_train(sample_idx, :);
y_plot = y_train(sample_idx);
scatter(x_plot(:, 1), x_plot(:, 2), 12, y_plot, 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeAlpha', 0.5);
grid on;
xlabel('t');
ylabel('x_t');
title('Training Pairs: (t, x_t) to v');
colorbar;
end
