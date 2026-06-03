% Plot LoG-GP predictive uncertainty along an ODE rollout.
function plot_uncertainty_vs_time(cfg, traj_times, traj_gp_vars, plot_title, ...
    threshold)
if nargin < 4 || isempty(plot_title)
    plot_title = 'LoG-GP Predictive Uncertainty Along ODE Rollout';
end
if nargin < 5 || isempty(threshold)
    threshold = cfg.variance_constraint.sigma2_max;
end
%% Output Path
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_path = fullfile(output_dir, ...
    'trajectory_gp_uncertainty_vs_time_matlab.emf');

%% Figure Layout
fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.18, 0.18, 0.55, 0.48]);
movegui(fig, 'center');
hold on;

%% Uncertainty Curves
scalar_vars = traj_gp_vars(:, :, 1);
for idx = 1:size(scalar_vars, 2)
    plot(traj_times, scalar_vars(:, idx), 'Color', [0.55, 0.72, 0.92], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
end
yline(threshold, '--', ...
    'Color', [0.85, 0.20, 0.20], 'LineWidth', 1.4, ...
    'DisplayName', 'uncertainty threshold');
set(gca, 'YScale', 'log');
grid on;
xlabel('s');
ylabel('LoG-GP predictive uncertainty');
title(plot_title);
legend('Location', 'best');
exportgraphics(fig, output_path, 'ContentType', 'vector');
end
