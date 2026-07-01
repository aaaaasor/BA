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
has_psi1_dot = isfield(trace, 'trace_psi1_dot') && ...
    ~isempty(trace.trace_psi1_dot);
if has_psi1_dot
    psi1_dot = trace.trace_psi1_dot(:);
else
    psi1_dot = [];
end
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
has_u_trace = isfield(trace, 'trace_correction_norm') && ...
    ~isempty(trace.trace_correction_norm);
if has_u_trace
    correction_norm = trace.trace_correction_norm(:);
    if isfield(trace, 'trace_max_abs_u') && ...
            ~isempty(trace.trace_max_abs_u)
        max_abs_u = trace.trace_max_abs_u(:);
    else
        max_abs_u = nan(size(correction_norm));
    end
else
    correction_norm = [];
    max_abs_u = [];
end
barrier_b = trace.trace_barrier_b(:);
relaxed_barrier_b = trace.trace_relaxed_barrier_b(:);
h_bar = trace.trace_h_bar(:);
[~, sort_idx] = sortrows([sample_idx, step_idx, stage_idx]);
trace_t = trace_t(sort_idx);
psi_values = psi_values(sort_idx, :);
if has_psi1_dot
    psi1_dot = psi1_dot(sort_idx);
end
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
if has_u_trace
    correction_norm = correction_norm(sort_idx);
    max_abs_u = max_abs_u(sort_idx);
end
barrier_b = barrier_b(sort_idx);
relaxed_barrier_b = relaxed_barrier_b(sort_idx);
h_bar = h_bar(sort_idx);
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
sgtitle('PT-HOCBF \psi diagnostics', 'Interpreter', 'tex');

if has_psi1_dot
    fig_psi1_dot = figure('Color', 'w', 'WindowStyle', 'normal', ...
        'Units', 'normalized', 'Position', [0.16, 0.16, 0.70, 0.52]);
    movegui(fig_psi1_dot, 'center');
    tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    hold on;
    for sample_now = unique_samples
        rows_now = sample_idx == sample_now & stage_idx == 1;
        t_now = trace_t(rows_now);
        psi1_now = psi_values(rows_now, 2);
        psi1_dot_now = psi1_dot(rows_now);
        if numel(t_now) < 2
            continue;
        end
        t_fd = t_now(1:end-1);
        psi1_dot_fd = diff(psi1_now) ./ diff(t_now);
        plot(t_now, psi1_dot_now, '-', 'Color', [0.20, 0.45, 0.85], ...
            'LineWidth', 1.0, 'HandleVisibility', 'off');
        plot(t_fd, psi1_dot_fd, '--', ...
            'Color', [0.90, 0.45, 0.10], 'LineWidth', 1.0, ...
            'HandleVisibility', 'off');
    end
    grid on;
    xlabel('s');
    ylabel('$\dot{\psi}_1$', 'Interpreter', 'latex');
    title('$\dot{\psi}_1$: analytic versus finite difference', ...
        'Interpreter', 'latex');
    plot(nan, nan, '-', 'Color', [0.20, 0.45, 0.85], ...
        'LineWidth', 1.0, 'DisplayName', 'analytic');
    plot(nan, nan, '--', 'Color', [0.90, 0.45, 0.10], ...
        'LineWidth', 1.0, 'DisplayName', 'finite difference');
    legend('Location', 'best', 'Interpreter', 'latex');

    nexttile;
    hold on;
    for sample_now = unique_samples
        rows_now = sample_idx == sample_now & stage_idx == 1;
        t_now = trace_t(rows_now);
        psi1_now = psi_values(rows_now, 2);
        psi1_dot_now = psi1_dot(rows_now);
        if numel(t_now) < 2
            continue;
        end
        t_fd = t_now(1:end-1);
        psi1_dot_fd = diff(psi1_now) ./ diff(t_now);
        psi1_dot_error = psi1_dot_fd - psi1_dot_now(1:end-1);
        plot(t_fd, psi1_dot_error, ...
            'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.0, ...
            'HandleVisibility', 'off');
    end
    yline(0, '--', 'Color', [0.85, 0.20, 0.20], ...
        'LineWidth', 1.0, 'DisplayName', '0');
    grid on;
    xlabel('s');
    ylabel('FD - analytic');
    title('$\dot{\psi}_1$ finite-difference error', 'Interpreter', 'latex');
    legend('Location', 'best', 'Interpreter', 'latex');
    sgtitle('PT-HOCBF \psi_1 derivative diagnostic', 'Interpreter', 'tex');
end

if has_u_trace
    fig_u = figure('Color', 'w', 'WindowStyle', 'normal', ...
        'Units', 'normalized', 'Position', [0.18, 0.18, 0.70, 0.52]);
    movegui(fig_u, 'center');
    hold on;
    for sample_now = unique_samples
        rows_now = sample_idx == sample_now;
        plot(trace_t(rows_now), correction_norm(rows_now), ...
            'Color', [0.20, 0.45, 0.85], 'LineWidth', 1.0, ...
            'HandleVisibility', 'off');
    end
    grid on;
    xlabel('s');
    ylabel('||u||_2');
    title('PT-HOCBF correction u over time');
end



fig_barrier = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.14, 0.14, 0.70, 0.58]);
movegui(fig_barrier, 'center');
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
barrier_names = {'$b(s)$', '$-\bar{h}(s)$', '$b(s)+\bar{h}(s)$'};
barrier_values = {barrier_b, -h_bar, relaxed_barrier_b};
barrier_colors = {[0.10, 0.55, 0.25], [0.90, 0.45, 0.10], [0.20, 0.45, 0.85]};
for barrier_idx = 1:numel(barrier_names)
    nexttile;
    hold on;
    for sample_now = unique_samples
        rows_now = sample_idx == sample_now;
        plot(trace_t(rows_now), barrier_values{barrier_idx}(rows_now), ...
            'Color', barrier_colors{barrier_idx}, 'LineWidth', 1.0, ...
            'HandleVisibility', 'off');
    end
    yline(0, '--', 'Color', [0.85, 0.20, 0.20], 'LineWidth', 1.1, 'DisplayName', '0');
    grid on;
    xlabel('s');
    ylabel(barrier_names{barrier_idx}, 'Interpreter', 'latex');
    title([barrier_names{barrier_idx}, ' over RK4 stages'], 'Interpreter', 'latex');
    legend('Location', 'best', 'Interpreter', 'latex');
end
sgtitle('PT-HOCBF barrier and relaxation diagnostics', 'Interpreter', 'tex');


if cfg.output.enabled
    this_file = mfilename('fullpath');
    output_dir = fullfile(fileparts(this_file), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    output_path = fullfile(output_dir, 'trajectory_gp_hocbf_psi_trace_matlab.emf');
    local_export_graphics(fig, output_path);
    if has_psi1_dot
        psi1_dot_output_path = fullfile(output_dir, ...
            'trajectory_gp_hocbf_psi1_dot_compare_matlab.emf');
        local_export_graphics(fig_psi1_dot, psi1_dot_output_path);
    end
    if has_u_trace
        u_output_path = fullfile(output_dir, ...
            'trajectory_gp_hocbf_u_trace_matlab.emf');
        if exist('fig_u', 'var') && isgraphics(fig_u, 'figure')
            local_export_graphics(fig_u, u_output_path);
        end
    end
    if exist('fig_barrier', 'var') && isgraphics(fig_barrier, 'figure')
        barrier_output_path = fullfile(output_dir, 'trajectory_gp_hocbf_b_hbar_trace_matlab.emf');
        local_export_graphics(fig_barrier, barrier_output_path);
    end
end
end

function local_export_graphics(target, output_path)
export_graphics_compat(target, output_path);
end
