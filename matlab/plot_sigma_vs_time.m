function plot_sigma_vs_time(cfg, traj_times, uncertainty_values, level_label, ...
    rollout_diagnostics)
if nargin < 5
    rollout_diagnostics = [];
end
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
plot(nan, nan, '-', 'Color', [0.20, 0.45, 0.85], ...
    'LineWidth', 1.2, 'DisplayName', 'actual sqrt(beta)');
if ~isempty(rollout_diagnostics) && isfield(rollout_diagnostics, 'hocbf')
    trace = rollout_diagnostics.hocbf;
    has_terminal_cap = isfield(trace, 'trace_terminal_beta_cap') && ...
        ~isempty(trace.trace_terminal_beta_cap) && ...
        any(isfinite(trace.trace_terminal_beta_cap));
    if has_terminal_cap
        terminal_t = trace.trace_t(:);
        terminal_sigma_cap = sqrt(max(trace.trace_terminal_beta_cap(:), 0.0));
        rows_now = isfinite(terminal_sigma_cap) & isfinite(terminal_t);
        if any(rows_now)
            [t_unique, ~, group_idx] = unique(terminal_t(rows_now));
            cap_line = accumarray(group_idx, terminal_sigma_cap(rows_now), ...
                [], @max);
            [t_unique, sort_idx] = sort(t_unique);
            cap_line = cap_line(sort_idx);
            plot(t_unique, cap_line, '--', 'Color', [0.85, 0.20, 0.20], ...
                'LineWidth', 1.4, 'DisplayName', ...
                'PTCBF cap sqrt(beta final + hbar)');
        end
    end
end
grid on;
xlabel('s');
ylabel('\sigma(s)');
title([level_label, ': GP predictive \sigma along rollout']);
legend('Location', 'best', 'Interpreter', 'none');

if cfg.output.enabled
    this_file = mfilename('fullpath');
    output_dir = fullfile(fileparts(this_file), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    output_path = fullfile(output_dir, ...
        ['trajectory_gp_sigma_vs_time_', strrep(level_label, ' ', '_'), '_matlab.emf']);
    export_graphics_compat(fig, output_path);
end
end
