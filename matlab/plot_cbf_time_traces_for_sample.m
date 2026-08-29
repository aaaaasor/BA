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
[node_times, node_path, diagnostics] = rk4_rollout(model_collection, ...
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

[ptcbf_time, ptcbf_values] = conservative_time_trace( ...
    t, trace.trace_terminal_h(:), ptcbf_active);

output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
if ~exist(output_dir, 'dir'); mkdir(output_dir); end
sample_tag = sprintf('Sample%03d', sample_idx);

fig_psi1 = [];
fig_psi2 = [];
hocbf_trace_active = hocbf_active & isfinite(t) & ...
    isfinite(trace.trace_psi1(:)) & isfinite(trace.trace_psi2(:));
if any(hocbf_trace_active)
    [time_values, psi1_values] = conservative_time_trace( ...
        t, trace.trace_psi1(:), hocbf_trace_active);
    [~, psi2_values] = conservative_time_trace( ...
        t, trace.trace_psi2(:), hocbf_trace_active);
    fig_psi1 = make_trace_figure(time_values, psi1_values, ...
        '$\psi_1$', ['Third-level HOCBF $\psi_1(t)$, ', sample_tag]);
    fig_psi2 = make_trace_figure(time_values, psi2_values, ...
        '$\psi_2$', ['Third-level HOCBF $\psi_2(t)$, ', sample_tag]);
else
    fprintf(['Skipping HOCBF psi1/psi2 figures for sample %03d ', ...
        'because HOCBF is disabled or has no active finite trace.\n'], ...
        sample_idx);
end
fig_ptcbf = make_trace_figure(ptcbf_time, ptcbf_values, ...
    '$h_{\mathrm{PTCBF}}$', ...
    ['Third-level PTCBF $h_{\mathrm{PTCBF}}(t)$, ', sample_tag]);

beta = trace.trace_sigma2(:);
beta_cap = trace.trace_terminal_beta_cap(:);
state_excess = beta - beta_cap;
active_rows = ptcbf_active & isfinite(state_excess);
violating_rows = find(active_rows & state_excess > 1e-8);
if isempty(violating_rows)
    fprintf(['Third-level variance mechanism sample %03d: no RK4-stage ', ...
        'beta-cap violation.\n'], sample_idx);
else
    first_row = violating_rows(1);
    [worst_excess, local_worst_idx] = max(state_excess(violating_rows));
    worst_row = violating_rows(local_worst_idx);
    print_variance_violation_row('first', trace, first_row, state_excess(first_row));
    print_variance_violation_row('worst', trace, worst_row, worst_excess);
end

fig_mechanism = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.08, 0.08, 0.82, 0.78]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot(t, beta, '-', 'LineWidth', 1.2, 'DisplayName', '$\beta$'); hold on;
plot(t, beta_cap, '--', 'LineWidth', 1.5, 'DisplayName', '$\beta_{cap}$');
grid on; xlabel('$t$', 'Interpreter', 'latex'); ylabel('variance');
legend('Interpreter', 'latex', 'Location', 'best');
nexttile;
semilogy(t, max(trace.trace_grad_norm(:), realmin), 'LineWidth', 1.2);
grid on; xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\|\nabla_x\beta\|_2$', 'Interpreter', 'latex');
nexttile;
plot(t, trace.trace_terminal_constraint_residual(:), 'LineWidth', 1.2); hold on;
yline(0, '--', 'Color', [0.80, 0.18, 0.18]); grid on;
xlabel('$t$', 'Interpreter', 'latex'); ylabel('QP terminal residual');
nexttile;
plot(t, trace.trace_max_abs_u(:), 'LineWidth', 1.2); grid on;
xlabel('$t$', 'Interpreter', 'latex'); ylabel('$\max |u|$', 'Interpreter', 'latex');
sgtitle(['Third-level variance violation mechanism, ', sample_tag], ...
    'Interpreter', 'none');

% Compare the chain-rule variance derivative used by the QP with the
% finite change of the GP variance along the committed RK4 trajectory.
% For each integration step, average the four stage derivatives with the
% classical RK4 weights, then compare against (beta_{k+1}-beta_k)/dt.
node_beta = zeros(numel(node_times), 1);
for node_idx = 1:numel(node_times)
    x_node = reshape(node_path(node_idx, 1, :), [], 1);
    gp_input = [node_times(node_idx); x_node];
    for output_idx = 1:numel(model_collection.model.output_models)
        node_beta(node_idx) = node_beta(node_idx) + ...
            model_collection.model.output_models{output_idx}. ...
            predict_variance(gp_input);
    end
end
dt_nodes = diff(node_times(:));
actual_beta_dot = diff(node_beta) ./ dt_nodes;
model_beta_dot_stage = trace.trace_rho_components(:, 1) + ...
    trace.trace_rho_components(:, 2) + ...
    trace.trace_terminal_constraint_residual(:) + ...
    trace.trace_terminal_bound(:);
model_beta_dot = nan(size(actual_beta_dot));
rk4_weights = [1; 2; 2; 1] ./ 6;
for step_idx = 1:numel(actual_beta_dot)
    stage_rows = find(trace.trace_step_idx(:) == step_idx & ...
        ismember(trace.trace_stage_idx(:), 1:4));
    if numel(stage_rows) == 4
        [~, stage_order] = sort(trace.trace_stage_idx(stage_rows));
        stage_rows = stage_rows(stage_order);
        model_beta_dot(step_idx) = ...
            rk4_weights' * model_beta_dot_stage(stage_rows);
    end
end
derivative_time = 0.5 .* (node_times(1:end-1) + node_times(2:end));
derivative_error = actual_beta_dot - model_beta_dot;
finite_derivative = isfinite(actual_beta_dot) & isfinite(model_beta_dot);
if any(finite_derivative)
    derivative_rmse = sqrt(mean(derivative_error(finite_derivative) .^ 2));
    [derivative_max_error, derivative_worst_local] = max( ...
        abs(derivative_error(finite_derivative)));
    derivative_finite_rows = find(finite_derivative);
    derivative_worst_idx = derivative_finite_rows(derivative_worst_local);
    fprintf(['Third-level variance derivative consistency sample %03d: ', ...
        'RMSE=%.6g, max|actual-model|=%.6g at t=%.6g ', ...
        '(actual=%.6g, model=%.6g).\n'], sample_idx, derivative_rmse, ...
        derivative_max_error, derivative_time(derivative_worst_idx), ...
        actual_beta_dot(derivative_worst_idx), ...
        model_beta_dot(derivative_worst_idx));
else
    derivative_rmse = nan;
    derivative_max_error = nan;
end
diagnostics.variance_derivative_consistency = struct( ...
    'time', derivative_time, 'node_beta', node_beta, ...
    'actual_beta_dot', actual_beta_dot, ...
    'model_beta_dot', model_beta_dot, ...
    'error', derivative_error, 'rmse', derivative_rmse, ...
    'max_abs_error', derivative_max_error);

fig_derivative = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.12, 0.12, 0.76, 0.68]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot(derivative_time, actual_beta_dot, '-', 'LineWidth', 1.3, ...
    'DisplayName', 'actual finite difference'); hold on;
plot(derivative_time, model_beta_dot, '--', 'LineWidth', 1.3, ...
    'DisplayName', 'QP model (RK4 stage average)');
grid on; ylabel('$\dot{\beta}$', 'Interpreter', 'latex');
legend('Location', 'best');
nexttile;
plot(derivative_time, derivative_error, 'LineWidth', 1.2); hold on;
yline(0, '--', 'Color', [0.80, 0.18, 0.18]); grid on;
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$\dot{\beta}_{actual}-\dot{\beta}_{model}$', ...
    'Interpreter', 'latex');
sgtitle(['Third-level variance derivative consistency, ', sample_tag], ...
    'Interpreter', 'none');

if struct_field_default(cfg.output, 'enabled', true)
    if ~isempty(fig_psi1) && isgraphics(fig_psi1)
        export_graphics_compat(fig_psi1, fullfile(output_dir, ...
            ['ThirdLevel_HOCBF_Psi1_Over_Time_', sample_tag, '.emf']));
    end
    if ~isempty(fig_psi2) && isgraphics(fig_psi2)
        export_graphics_compat(fig_psi2, fullfile(output_dir, ...
            ['ThirdLevel_HOCBF_Psi2_Over_Time_', sample_tag, '.emf']));
    end
    export_graphics_compat(fig_ptcbf, fullfile(output_dir, ...
        ['ThirdLevel_PTCBF_h_Over_Time_', sample_tag, '.emf']));
    export_graphics_compat(fig_mechanism, fullfile(output_dir, ...
        ['ThirdLevel_Variance_Violation_Mechanism_', sample_tag, '.png']));
    export_graphics_compat(fig_derivative, fullfile(output_dir, ...
        ['ThirdLevel_Variance_Derivative_Consistency_', sample_tag, '.png']));
end
end
%%
function print_variance_violation_row(label, trace, row_idx, excess)
fprintf(['Third-level variance %s violation: t=%.6g, beta=%.6g, ', ...
    'cap=%.6g, excess=%.6g, grad_norm=%.6g, terminal_bound=%.6g, ', ...
    'QP_residual=%.6g, max|u|=%.6g, exitflag=%g.\n'], ...
    label, trace.trace_t(row_idx), trace.trace_sigma2(row_idx), ...
    trace.trace_terminal_beta_cap(row_idx), excess, ...
    trace.trace_grad_norm(row_idx), trace.trace_terminal_bound(row_idx), ...
    trace.trace_terminal_constraint_residual(row_idx), ...
    trace.trace_max_abs_u(row_idx), trace.trace_qp_exitflag(row_idx));
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
