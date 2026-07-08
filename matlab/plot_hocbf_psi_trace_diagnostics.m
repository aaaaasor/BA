function plot_hocbf_psi_trace_diagnostics(cfg, rollout_diagnostics)
trace = rollout_diagnostics.hocbf;
if isempty(trace.trace_t)
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
psi_values = [trace.trace_psi0(:), trace.trace_psi1(:), trace.trace_psi2(:)];
has_rho_trace = isfield(trace, 'trace_rho') && ...
    ~isempty(trace.trace_rho);
if has_rho_trace
    rho_trace = trace.trace_rho(:);
else
    rho_trace = [];
end
has_rho_components = isfield(trace, 'trace_rho_components') && ...
    ~isempty(trace.trace_rho_components);
if has_rho_components
    rho_components = trace.trace_rho_components;
else
    rho_components = [];
end
has_psi_decomposition = isfield(trace, 'trace_psi0_dot') && ...
    isfield(trace, 'trace_b_dot') && isfield(trace, 'trace_h_bar_dot') && ...
    isfield(trace, 'trace_alpha1_psi0') && ...
    ~isempty(trace.trace_psi0_dot);
if has_psi_decomposition
    psi0_dot_trace = trace.trace_psi0_dot(:);
    b_dot_trace = trace.trace_b_dot(:);
    h_bar_dot_trace = trace.trace_h_bar_dot(:);
    alpha1_psi0_trace = trace.trace_alpha1_psi0(:);
else
    psi0_dot_trace = [];
    b_dot_trace = [];
    h_bar_dot_trace = [];
    alpha1_psi0_trace = [];
end
has_constraint_traces = isfield(trace, 'trace_integral_bound') && ...
    isfield(trace, 'trace_terminal_bound') && ...
    isfield(trace, 'trace_hocbf_constraint_residual') && ...
    isfield(trace, 'trace_terminal_constraint_residual') && ...
    ~isempty(trace.trace_integral_bound) && ...
    ~isempty(trace.trace_terminal_bound);
if has_constraint_traces
    integral_bound = trace.trace_integral_bound(:);
    terminal_bound = trace.trace_terminal_bound(:);
    if isfield(trace, 'trace_hocbf_relaxed_constraint_residual') && ...
            ~isempty(trace.trace_hocbf_relaxed_constraint_residual)
        hocbf_residual = trace.trace_hocbf_relaxed_constraint_residual(:);
    else
        hocbf_residual = trace.trace_hocbf_constraint_residual(:);
    end
    if isfield(trace, 'trace_terminal_relaxed_constraint_residual') && ...
            ~isempty(trace.trace_terminal_relaxed_constraint_residual)
        terminal_residual = trace.trace_terminal_relaxed_constraint_residual(:);
    else
        terminal_residual = trace.trace_terminal_constraint_residual(:);
    end
else
    integral_bound = [];
    terminal_bound = [];
    hocbf_residual = [];
    terminal_residual = [];
end
if has_constraint_traces
    if isfield(trace, 'trace_hocbf_enabled') && ...
            ~isempty(trace.trace_hocbf_enabled)
        hocbf_enabled_trace = logical(trace.trace_hocbf_enabled(:));
    else
        hocbf_enabled_trace = true(size(trace_t));
    end
    integral_bound(~hocbf_enabled_trace) = nan;
    hocbf_residual(~hocbf_enabled_trace) = nan;
    if isfield(trace, 'trace_ptcbf_enabled') && ...
            ~isempty(trace.trace_ptcbf_enabled)
        terminal_enabled_trace = logical(trace.trace_ptcbf_enabled(:));
        terminal_bound(~terminal_enabled_trace) = nan;
        terminal_residual(~terminal_enabled_trace) = nan;
    end
end
has_anchor_clf_traces = isfield(trace, 'trace_anchor_clf_bound') && ...
    isfield(trace, 'trace_anchor_clf_residual') && ...
    ~isempty(trace.trace_anchor_clf_bound) && ...
    ~isempty(trace.trace_anchor_clf_residual) && ...
    any(isfinite(trace.trace_anchor_clf_bound));
if has_anchor_clf_traces
    anchor_clf_bound = trace.trace_anchor_clf_bound(:);
    anchor_clf_residual = trace.trace_anchor_clf_residual(:);
else
    anchor_clf_bound = [];
    anchor_clf_residual = [];
end
barrier_b = trace.trace_barrier_b(:);
relaxed_barrier_b = trace.trace_relaxed_barrier_b(:);
h_bar = trace.trace_h_bar(:);
cumulative_variance = trace.trace_cumulative_variance(:);
[~, sort_idx] = sortrows([sample_idx, step_idx, stage_idx]);
trace_t = trace_t(sort_idx);
psi_values = psi_values(sort_idx, :);
if has_rho_trace
    rho_trace = rho_trace(sort_idx);
end
if has_rho_components
    rho_components = rho_components(sort_idx, :);
end
if has_psi_decomposition
    psi0_dot_trace = psi0_dot_trace(sort_idx);
    b_dot_trace = b_dot_trace(sort_idx);
    h_bar_dot_trace = h_bar_dot_trace(sort_idx);
    alpha1_psi0_trace = alpha1_psi0_trace(sort_idx);
end
if has_constraint_traces
    integral_bound = integral_bound(sort_idx);
    terminal_bound = terminal_bound(sort_idx);
    hocbf_residual = hocbf_residual(sort_idx);
    terminal_residual = terminal_residual(sort_idx);
end
if has_anchor_clf_traces
    anchor_clf_bound = anchor_clf_bound(sort_idx);
    anchor_clf_residual = anchor_clf_residual(sort_idx);
end
barrier_b = barrier_b(sort_idx);
relaxed_barrier_b = relaxed_barrier_b(sort_idx);
h_bar = h_bar(sort_idx);
cumulative_variance = cumulative_variance(sort_idx);
sample_idx = sample_idx(sort_idx);
stage_idx = stage_idx(sort_idx);
unique_samples = unique(sample_idx)';

psi_names = {'$\psi_0$', '$\psi_1$', '$\psi_2$'};
fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.12, 0.12, 0.72, 0.64]);
movegui(fig, 'center');
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for psi_idx = 1:3
    nexttile;
    hold on;
    for sample_now = unique_samples
        rows_now = sample_idx == sample_now;
        plot(trace_t(rows_now), psi_values(rows_now, psi_idx), ...
            'Color', [0.20, 0.45, 0.85], 'LineWidth', 1.0, ...
            'HandleVisibility', 'off');
    end
    yline(0, '--', 'Color', [0.85, 0.20, 0.20], ...
        'LineWidth', 1.1, 'DisplayName', [psi_names{psi_idx}, ' = 0']);
    grid on;
    xlabel('s');
    ylabel(psi_names{psi_idx}, 'Interpreter', 'latex');
    title([psi_names{psi_idx}, ' trace over RK4 stages'], 'Interpreter', 'latex');
    legend('Location', 'best', 'Interpreter', 'latex');
end
sgtitle('Integral HOCBF \psi diagnostics', 'Interpreter', 'tex');

fig_integral = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.16, 0.16, 0.70, 0.50]);
movegui(fig_integral, 'center');
hold on;
for sample_now = unique_samples
    rows_now = sample_idx == sample_now;
    upper_now = cumulative_variance(rows_now) + ...
        relaxed_barrier_b(rows_now);
    plot(trace_t(rows_now), cumulative_variance(rows_now), ...
        'Color', [0.20, 0.45, 0.85], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
    plot(trace_t(rows_now), upper_now, '--', ...
        'Color', [0.85, 0.20, 0.20], 'LineWidth', 0.9, ...
        'HandleVisibility', 'off');
end
plot(nan, nan, '-', 'Color', [0.20, 0.45, 0.85], ...
    'LineWidth', 1.0, 'DisplayName', ...
    '$\int_0^s \beta(\tau)d\tau$');
plot(nan, nan, '--', 'Color', [0.85, 0.20, 0.20], ...
    'LineWidth', 0.9, 'DisplayName', '$h_0 + Bs$');
grid on;
xlabel('s');
ylabel('cumulative variance');
title('Integral HOCBF: cumulative variance vs relaxed budget');
legend('Location', 'best', 'Interpreter', 'latex');

if has_constraint_traces
    fig_qp = figure('Color', 'w', 'WindowStyle', 'normal', ...
        'Units', 'normalized', 'Position', [0.17, 0.12, 0.72, 0.62]);
    movegui(fig_qp, 'center');
    tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    hold on;
    for sample_now = unique_samples
        rows_now = sample_idx == sample_now;
        plot(trace_t(rows_now), integral_bound(rows_now), ...
            'Color', [0.10, 0.55, 0.25], 'LineWidth', 0.9, ...
            'HandleVisibility', 'off');
        plot(trace_t(rows_now), terminal_bound(rows_now), ...
            'Color', [0.85, 0.20, 0.20], 'LineWidth', 0.9, ...
            'HandleVisibility', 'off');
        if has_anchor_clf_traces
            plot(trace_t(rows_now), anchor_clf_bound(rows_now), ...
                'Color', [0.45, 0.20, 0.75], 'LineWidth', 0.9, ...
                'HandleVisibility', 'off');
        end
    end
    plot(nan, nan, '-', 'Color', [0.10, 0.55, 0.25], ...
        'LineWidth', 0.9, 'DisplayName', 'integral bound');
    plot(nan, nan, '-', 'Color', [0.85, 0.20, 0.20], ...
        'LineWidth', 0.9, 'DisplayName', 'terminal bound');
    if has_anchor_clf_traces
        plot(nan, nan, '-', 'Color', [0.45, 0.20, 0.75], ...
            'LineWidth', 0.9, 'DisplayName', 'anchor PTCLF bound');
    end
    grid on;
    xlabel('s');
    ylabel('raw QP bound');
    title('Raw QP upper bounds before slack: smaller bound is tighter');
    legend('Location', 'best');

    nexttile;
    hold on;
    for sample_now = unique_samples
        rows_now = sample_idx == sample_now;
        plot(trace_t(rows_now), hocbf_residual(rows_now), ...
            'Color', [0.10, 0.55, 0.25], 'LineWidth', 0.9, ...
            'HandleVisibility', 'off');
        plot(trace_t(rows_now), terminal_residual(rows_now), ...
            'Color', [0.85, 0.20, 0.20], 'LineWidth', 0.9, ...
            'HandleVisibility', 'off');
        if has_anchor_clf_traces
            plot(trace_t(rows_now), anchor_clf_residual(rows_now), ...
                'Color', [0.45, 0.20, 0.75], 'LineWidth', 0.9, ...
                'HandleVisibility', 'off');
        end
    end
    yline(0, '--', 'Color', [0.20, 0.20, 0.20], ...
        'LineWidth', 1.0, 'DisplayName', '0');
    plot(nan, nan, '-', 'Color', [0.10, 0.55, 0.25], ...
        'LineWidth', 0.9, 'DisplayName', 'integral residual');
    plot(nan, nan, '-', 'Color', [0.85, 0.20, 0.20], ...
        'LineWidth', 0.9, 'DisplayName', 'terminal residual');
    if has_anchor_clf_traces
        plot(nan, nan, '-', 'Color', [0.45, 0.20, 0.75], ...
            'LineWidth', 0.9, 'DisplayName', 'anchor PTCLF residual');
    end
    grid on;
    xlabel('s');
    ylabel('A u - b - slack');
    title('QP constraint residuals: should be <= 0');
    legend('Location', 'best');
end

if cfg.output.enabled
    this_file = mfilename('fullpath');
    output_dir = fullfile(fileparts(this_file), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    output_path = fullfile(output_dir, 'trajectory_gp_hocbf_psi_trace_matlab.emf');
    export_graphics_compat(fig, output_path);
    if exist('fig_integral', 'var') && isgraphics(fig_integral, 'figure')
        integral_output_path = fullfile(output_dir, ...
            'trajectory_gp_hocbf_integral_budget_trace_matlab.emf');
        export_graphics_compat(fig_integral, integral_output_path);
    end
    if exist('fig_qp', 'var') && isgraphics(fig_qp, 'figure')
        qp_output_path = fullfile(output_dir, ...
            'trajectory_gp_hocbf_ptcbf_qp_constraints_matlab.emf');
        export_graphics_compat(fig_qp, qp_output_path);
    end
end
end
