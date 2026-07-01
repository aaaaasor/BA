function plot_sigma_vs_time(cfg, traj_times, uncertainty_values, level_label)
beta_by_sample = aggregate_variance_values(uncertainty_values);
sigma_by_sample = sqrt(beta_by_sample);

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.14, 0.18, 0.60, 0.46]);
movegui(fig, 'center');
hold on;
for sample_idx = 1:size(sigma_by_sample, 2)
    plot(traj_times, sigma_by_sample(:, sample_idx), ...
        'Color', [0.20, 0.45, 0.85], 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
end
grid on;
xlabel('s');
ylabel('\sigma(s)');
title([level_label, ': GP predictive \sigma along rollout']);

if cfg.output.enabled
    this_file = mfilename('fullpath');
    output_dir = fullfile(fileparts(this_file), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    output_path = fullfile(output_dir, ...
        ['trajectory_gp_sigma_vs_time_', strrep(level_label, ' ', '_'), '_matlab.emf']);
    export_graphics_compat(fig, output_path);
end
end
