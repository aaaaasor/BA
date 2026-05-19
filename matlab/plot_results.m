% Plot the main MATLAB summary figure:
% - all training trajectories,
% - rollout trajectories,
% - training velocity vectors.
function plot_results(cfg, trajectory_points, traj_path, x_train, y_train)
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_path = fullfile(output_dir, 'gp_flow_matching_demo_matlab.png');

%% Create figure layout
fig = figure('Color', 'w', 'WindowStyle', 'normal', 'Units', 'normalized', 'Position', [0.10, 0.10, 0.75, 0.78]);
movegui(fig, 'center');

tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Training trajectory data
nexttile([1 2]);
hold on;
n_training_curves = size(trajectory_points, 2);
for idx = 1:n_training_curves
    training_curve = squeeze(trajectory_points(:, idx, :));
    plot(training_curve(:, 1), training_curve(:, 2), '.-', 'LineWidth', 0.8, ...
        'Color', [0.35, 0.55, 0.85], 'HandleVisibility', 'off');
end
grid on;
axis equal;
xlabel('x');
ylabel('y');
title(sprintf('Training Trajectory Data (2D, %d Points)', size(trajectory_points, 1)));

%% Rollout trajectories
nexttile;
hold on;
max_curves = min(size(traj_path, 2), 40);
for idx = 1:max_curves
    plot(traj_path(:, idx, 1), traj_path(:, idx, 2), 'LineWidth', 0.9);
end
grid on;
axis equal;
xlabel('x(t)');
ylabel('y(t)');
title('Sample Trajectories in 2D');

%% Training velocity quiver plot
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
