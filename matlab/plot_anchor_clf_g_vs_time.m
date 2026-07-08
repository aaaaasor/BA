function plot_anchor_clf_g_vs_time(cfg, level_label, rollout_diagnostics)
if isempty(rollout_diagnostics) || ~isfield(rollout_diagnostics, 'hocbf')
    return;
end
trace = rollout_diagnostics.hocbf;
if isempty(trace.trace_t) || ~isfield(trace, 'trace_anchor_clf_v') || ...
        isempty(trace.trace_anchor_clf_v)
    return;
end

sample_idx = trace.trace_sample_idx(:);
step_idx = trace.trace_step_idx(:);
if isfield(trace, 'trace_stage_idx') && ~isempty(trace.trace_stage_idx)
    stage_idx = trace.trace_stage_idx(:);
else
    stage_idx = ones(size(sample_idx));
end
trace_t = trace.trace_t(:);
g_now = trace.trace_anchor_clf_v(:);
gbar_now = trace.trace_anchor_clf_ptzf_bound(:);

rows_now = isfinite(g_now) | isfinite(gbar_now);
if ~any(rows_now)
    return;
end

[~, sort_idx] = sortrows([sample_idx, step_idx, stage_idx]);
trace_t = trace_t(sort_idx);
g_now = g_now(sort_idx);
gbar_now = gbar_now(sort_idx);
sample_idx = sample_idx(sort_idx);
unique_samples = unique(sample_idx)';

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.14, 0.18, 0.60, 0.46]);
movegui(fig, 'center');
hold on;
for sample_now = unique_samples
    rows_sample = sample_idx == sample_now;
    plot(trace_t(rows_sample), g_now(rows_sample), '-', ...
        'Color', [0.20, 0.45, 0.85], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
    plot(trace_t(rows_sample), gbar_now(rows_sample), '--', ...
        'Color', [0.85, 0.20, 0.20], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
end
plot(nan, nan, '-', 'Color', [0.20, 0.45, 0.85], ...
    'LineWidth', 1.2, 'DisplayName', 'g(s) = ||anchor error||^2');
plot(nan, nan, '--', 'Color', [0.85, 0.20, 0.20], ...
    'LineWidth', 1.2, 'DisplayName', 'gbar(s) PTZF envelope');
grid on;
xlabel('s');
ylabel('g, gbar');
title([level_label, ': anchor CLF g(s) vs PTZF gbar(s)']);
legend('Location', 'best', 'Interpreter', 'none');

if cfg.output.enabled
    this_file = mfilename('fullpath');
    output_dir = fullfile(fileparts(this_file), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    output_path = fullfile(output_dir, ...
        ['trajectory_anchor_clf_g_vs_gbar_', strrep(level_label, ' ', '_'), '_matlab.emf']);
    export_graphics_compat(fig, output_path);
end
end
