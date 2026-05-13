function plot_variance_vs_time(cfg, traj_times, traj_gp_vars)
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_path = fullfile(output_dir, 'trajectory_gp_variance_vs_time_matlab.png');

fig = figure('Color', 'w', 'WindowStyle', 'normal', 'Units', 'normalized', 'Position', [0.18, 0.18, 0.55, 0.48]);
movegui(fig, 'center');
hold on;

scalar_vars = traj_gp_vars(:, :, 1);
for idx = 1:size(scalar_vars, 2)
    plot(traj_times, scalar_vars(:, idx), 'Color', [0.55, 0.72, 0.92], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
end
yline(cfg.variance_constraint.sigma2_max, '--', 'Color', [0.85, 0.20, 0.20], 'LineWidth', 1.4, 'DisplayName', '\sigma^2 threshold');
set(gca, 'YScale', 'log');
grid on;
xlabel('t');
ylabel('posterior variance');
title('Posterior Variance Along Rollout Trajectories');
legend('Location', 'best');
exportgraphics(fig, output_path, 'Resolution', 180);
end
