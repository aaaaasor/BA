% Plot the main MATLAB summary figure:
function plot_results(cfg, target_points, source_points, reconstructed_points)
%% Output Path
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_path = fullfile(output_dir, 'gp_flow_matching_demo_matlab.png');

%% Figure Layout
fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.08, 0.18, 0.84, 0.48]);
movegui(fig, 'center');

tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Target Trajectories
nexttile;
hold on;
n_training_curves = size(target_points, 2);
for idx = 1:n_training_curves
    training_curve = squeeze(target_points(:, idx, :));
    plot(training_curve(:, 1), training_curve(:, 2), '.-', 'LineWidth', 0.8, ...
        'Color', [0.35, 0.55, 0.85], 'HandleVisibility', 'off');
end
grid on;
axis equal;
xlabel('x');
ylabel('y');
title(sprintf('Target Trajectory Data (2D, %d Points)', size(target_points, 1)));

%% Source Trajectories
nexttile;
hold on;
max_curves = min(size(source_points, 2), cfg.n_trajectories);
for idx = 1:max_curves
    source_curve = squeeze(source_points(:, idx, :));
    plot(source_curve(:, 1), source_curve(:, 2), '--', 'LineWidth', 0.9);
end
grid on;
axis equal;
xlabel('x');
ylabel('y');
title('10D ODE Source Trajectories');

%% Rollout Trajectories
nexttile;
hold on;
max_curves = min(size(reconstructed_points, 2), cfg.n_trajectories);
for idx = 1:max_curves
    reconstructed_curve = squeeze(reconstructed_points(:, idx, :));
    plot(reconstructed_curve(:, 1), reconstructed_curve(:, 2), 'LineWidth', 0.9);
end
grid on;
axis equal;
xlabel('x');
ylabel('y');
title('10D ODE Rollout Trajectories');

exportgraphics(fig, output_path, 'Resolution', 180);
end
