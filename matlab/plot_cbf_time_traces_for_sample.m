function diagnostics = plot_cbf_time_traces_for_sample(cfg, model_collection, ...
    x_init, t_min, t_max, n_steps, constraint_cfg, refine_cfg, sample_idx)
%PLOT_CBF_TIME_TRACES_FOR_SAMPLE Plot psi1, psi2, and terminal PTCBF h(t).
% A single selected sample is re-run with full RK4-stage diagnostics.  At
% duplicate RK4 stage times, the minimum active value is plotted so each
% curve shows the conservative safety margin at that instant.

n_samples = size(x_init, 1);
if sample_idx < 1 || sample_idx > n_samples || sample_idx ~= round(sample_idx)
    error('Sample index %g is outside the valid range 1:%d.', ...
        sample_idx, n_samples);
end

sample_constraint = constraint_cfg;
sample_constraint.diagnostics = true;
sample_constraint.control_trace_enabled = false;
sample_fields = {'anchor_clf_targets', ...
    'track_boundary_reference_s_min_targets', ...
    'track_boundary_reference_s_max_targets'};
for field_idx = 1:numel(sample_fields)
    field_name = sample_fields{field_idx};
    if isfield(sample_constraint, field_name) && ...
            ~isempty(sample_constraint.(field_name))
        sample_constraint.(field_name) = ...
            sample_constraint.(field_name)(sample_idx, :);
    end
end

serial_cfg.enabled = false;
serial_cfg.num_workers = 1;
serial_cfg.fallback_to_serial = true;
fprintf(['Generating HOCBF/PTCBF time traces from third-level ', ...
    'sample %03d...\n'], sample_idx);
[~, ~, diagnostics] = rk4_rollout(model_collection, ...
    x_init(sample_idx, :), t_min, t_max, n_steps, ...
    sample_constraint, refine_cfg, serial_cfg);

if ~isfield(diagnostics, 'hocbf') || ...
        isempty(diagnostics.hocbf.trace_t)
    error('The diagnostic re-run did not return an HOCBF/PTCBF trace.');
end
trace = diagnostics.hocbf;
t = trace.trace_t(:);

hocbf_active = true(size(t));
if isfield(trace, 'trace_hocbf_filter_active') && ...
        ~isempty(trace.trace_hocbf_filter_active)
    hocbf_active = logical(trace.trace_hocbf_filter_active(:));
elseif isfield(trace, 'trace_hocbf_enabled') && ...
        ~isempty(trace.trace_hocbf_enabled)
    hocbf_active = logical(trace.trace_hocbf_enabled(:));
end
ptcbf_active = true(size(t));
if isfield(trace, 'trace_ptcbf_enabled') && ...
        ~isempty(trace.trace_ptcbf_enabled)
    ptcbf_active = logical(trace.trace_ptcbf_enabled(:));
end

[time_values, psi1_values] = conservative_time_trace( ...
    t, trace.trace_psi1(:), hocbf_active);
[~, psi2_values] = conservative_time_trace( ...
    t, trace.trace_psi2(:), hocbf_active);
[ptcbf_time, ptcbf_values] = conservative_time_trace( ...
    t, trace.trace_terminal_h(:), ptcbf_active);

output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
if ~exist(output_dir, 'dir'); mkdir(output_dir); end
sample_tag = sprintf('Sample%03d', sample_idx);

fig_psi1 = make_trace_figure(time_values, psi1_values, ...
    '$\psi_1$', ['Third-level HOCBF $\psi_1(t)$, ', sample_tag]);
fig_psi2 = make_trace_figure(time_values, psi2_values, ...
    '$\psi_2$', ['Third-level HOCBF $\psi_2(t)$, ', sample_tag]);
fig_ptcbf = make_trace_figure(ptcbf_time, ptcbf_values, ...
    '$h_{\mathrm{PTCBF}}$', ...
    ['Third-level PTCBF $h_{\mathrm{PTCBF}}(t)$, ', sample_tag]);

if struct_field_default(cfg.output, 'enabled', true)
    export_graphics_compat(fig_psi1, fullfile(output_dir, ...
        ['ThirdLevel_HOCBF_Psi1_Over_Time_', sample_tag, '.emf']));
    export_graphics_compat(fig_psi2, fullfile(output_dir, ...
        ['ThirdLevel_HOCBF_Psi2_Over_Time_', sample_tag, '.emf']));
    export_graphics_compat(fig_ptcbf, fullfile(output_dir, ...
        ['ThirdLevel_PTCBF_h_Over_Time_', sample_tag, '.emf']));
end
end
%%
function [unique_t, min_values] = conservative_time_trace(t, values, active)
valid = active & isfinite(t) & isfinite(values);
if ~any(valid)
    error('No finite active values were recorded for a requested CBF trace.');
end
t = t(valid);
values = values(valid);
[unique_t, ~, group_idx] = unique(t, 'sorted');
min_values = accumarray(group_idx, values, [], @min);
end
%%
function fig = make_trace_figure(t, values, y_label, figure_title)
fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.16, 0.16, 0.70, 0.50]);
movegui(fig, 'center');
plot(t, values, 'Color', [0.10, 0.38, 0.82], 'LineWidth', 1.6);
hold on;
yline(0, '--', 'Color', [0.80, 0.18, 0.18], ...
    'LineWidth', 1.1, 'DisplayName', 'safety boundary');
grid on;
xlabel('time $t$ (s)', 'Interpreter', 'latex');
ylabel(y_label, 'Interpreter', 'latex');
title(figure_title, 'Interpreter', 'latex');
legend('CBF value', 'safety boundary', 'Location', 'best');
end
