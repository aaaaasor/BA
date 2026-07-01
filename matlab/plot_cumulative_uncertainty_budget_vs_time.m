function plot_cumulative_uncertainty_budget_vs_time(cfg, traj_times, ...
    uncertainty_values, budget, plot_title)
total_sigma2 = aggregate_variance_values(uncertainty_values);
cumulative_sigma2 = cumtrapz(traj_times(:), total_sigma2);
budget_line = budget * traj_times(:);
barrier_b = budget_line - cumulative_sigma2;

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.12, 0.14, 0.70, 0.56]);
movegui(fig, 'center');
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
for path_idx = 1:size(cumulative_sigma2, 2)
    plot(traj_times, cumulative_sigma2(:, path_idx), ...
        'Color', [0.20, 0.45, 0.85], 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
end
plot(traj_times, budget_line, '--', 'Color', [0.85, 0.20, 0.20], ...
    'LineWidth', 1.2, 'DisplayName', 'Bt');
grid on;
xlabel('s');
ylabel('\Sigma_i \int_0^s \sigma_i^2 d\tau');
title('cumulative variance with linear budget');
legend('Location', 'best');

nexttile;
hold on;
for path_idx = 1:size(barrier_b, 2)
    plot(traj_times, barrier_b(:, path_idx), ...
        'Color', [0.10, 0.55, 0.25], 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
end
yline(0, '--', 'Color', [0.85, 0.20, 0.20], ...
    'LineWidth', 1.2, 'DisplayName', 'b = 0');
grid on;
xlabel('s');
ylabel('b(s)');
title('barrier b = Bt - cumulative variance');
legend('Location', 'best');
sgtitle(plot_title);

if cfg.output.enabled
    this_file = mfilename('fullpath');
    output_dir = fullfile(fileparts(this_file), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    output_path = fullfile(output_dir, ...
        'trajectory_gp_hocbf_integral_budget_matlab.emf');
    export_graphics_compat(fig, output_path);
end
end
