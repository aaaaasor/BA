function plot_increment_control_traces(cfg, rollout_diagnostics, constraint_cfg)
% Plot the third-level control for the sample with the largest ||u||_2.
% trace_u is the final QP correction u; the applied state velocity is
% v = mu + u.  The decomposition makes a large stabilizing PTCLF action
% visible even when the state trajectory itself remains bounded.

if ~isfield(rollout_diagnostics, 'hocbf') || ...
		~isfield(rollout_diagnostics.hocbf, 'trace_u')
	return;
end
trace = rollout_diagnostics.hocbf;
if isempty(trace.trace_u) || isempty(trace.trace_t)
	return;
end

block_dim = struct_field_default(constraint_cfg, ...
	'increment_control_block_dim', 0);
n_blocks = struct_field_default(constraint_cfg, ...
	'increment_control_block_count', 0);
if n_blocks <= 0
	n_blocks = struct_field_default(cfg, 'segment_points_per_segment', 0);
end
if block_dim <= 0 && n_blocks > 0 && ...
		mod(size(trace.trace_u, 2), n_blocks) == 0
	block_dim = size(trace.trace_u, 2) / n_blocks;
end
if block_dim <= 0 || n_blocks <= 0 || ...
		block_dim * n_blocks ~= size(trace.trace_u, 2)
	warning(['Cannot plot increment controls: trace_u has %d columns, ', ...
		'but the configured block shape is %d x %d.'], ...
		size(trace.trace_u, 2), block_dim, n_blocks);
	return;
end

sample_idx = trace.trace_sample_idx(:);
step_idx = trace.trace_step_idx(:);
trace_t = trace.trace_t(:);
if isfield(trace, 'trace_stage_idx') && ~isempty(trace.trace_stage_idx)
	stage_idx = trace.trace_stage_idx(:);
else
	stage_idx = ones(size(sample_idx));
end
[~, order] = sortrows([sample_idx, step_idx, stage_idx]);
sample_idx = sample_idx(order);
trace_t = trace_t(order);
u_trace = trace.trace_u(order, :);

u_norm = row_norm(u_trace);
unique_samples = unique(sample_idx)';
sample_peak = zeros(size(unique_samples));
for idx = 1:numel(unique_samples)
	rows_now = sample_idx == unique_samples(idx);
	sample_peak(idx) = max(u_norm(rows_now));
end
[peak_u, worst_idx] = max(sample_peak);
worst_sample = unique_samples(worst_idx);
rows = sample_idx == worst_sample;
s = trace_t(rows);
u = u_trace(rows, :);

mu = ordered_field(trace, 'trace_mu', order, rows, size(u, 2));
v = ordered_field(trace, 'trace_v', order, rows, size(u, 2));
u_ptclf = ordered_field(trace, 'trace_u_ptclf_reference', ...
	order, rows, size(u, 2));
du_ptcbf = ordered_field(trace, 'trace_u_ptcbf_correction', ...
	order, rows, size(u, 2));
du_hocbf = ordered_field(trace, 'trace_u_hocbf_correction', ...
	order, rows, size(u, 2));

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
	'Units', 'normalized', 'Position', [0.10, 0.08, 0.78, 0.78]);
movegui(fig, 'center');
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
plot_if_available(s, mu, '--', 1.3, '||\mu||_2 (GP nominal)');
plot_if_available(s, u_ptclf, '-', 1.3, '||u_{PTCLF}||_2');
plot_if_available(s, du_ptcbf, '-', 1.3, '||\Delta u_{PTCBF}||_2');
plot_if_available(s, du_hocbf, '-', 1.3, '||\Delta u_{HOCBF}||_2');
plot(s, row_norm(u), 'k-', 'LineWidth', 2.0, ...
	'DisplayName', '||u||_2 (final QP correction)');
plot_if_available(s, v, ':', 1.8, '||v=\mu+u||_2 (applied)');
grid on;
ylabel('control / velocity norm');
title(sprintf('Worst sample %d: max ||u||_2 = %.4g', ...
	worst_sample, peak_u));
legend('Location', 'best');

nexttile;
hold on;
colors = lines(n_blocks);
for block_idx = 1:n_blocks
	cols = (block_idx - 1) * block_dim + (1:block_dim);
	plot(s, row_norm(u(:, cols)), 'Color', colors(block_idx, :), ...
		'LineWidth', 1.4, 'DisplayName', sprintf('||u_%d||_2', block_idx));
end
grid on;
xlabel('s (including RK4 stages)');
ylabel('increment-block norm');
title('Final QP correction by increment block');
legend('Location', 'best', 'NumColumns', min(n_blocks, 5));
sgtitle('Third-level control diagnostic');

fprintf(['Third-level u diagnostic: worst sample=%d, max||u||_2=%.6g, ', ...
	'max||mu||_2=%.6g, max||v||_2=%.6g.\n'], ...
	worst_sample, peak_u, max_or_nan(row_norm(mu)), max_or_nan(row_norm(v)));

if isfield(cfg, 'output') && cfg.output.enabled
	this_file = mfilename('fullpath');
	output_dir = fullfile(fileparts(this_file), 'outputs');
	if ~exist(output_dir, 'dir'); mkdir(output_dir); end
	export_graphics_compat(fig, fullfile(output_dir, ...
		'third_level_u_diagnostic_matlab.emf'));
	export_graphics_compat(fig, fullfile(output_dir, ...
		'third_level_u_diagnostic_matlab.png'));
end

% Advisor-style overview: one row per increment block and one curve per
% third-level rollout sample. Each row keeps its own y scale so localized
% control spikes are not hidden by the globally worst block.
fig_all = figure('Color', 'w', 'WindowStyle', 'normal', ...
	'Units', 'normalized', 'Position', [0.04, 0.05, 0.92, 0.86]);
movegui(fig_all, 'center');
tiledlayout(n_blocks, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
colors = lines(n_blocks);
for block_idx = 1:n_blocks
	nexttile;
	hold on;
	cols = (block_idx - 1) * block_dim + (1:block_dim);
	block_norm = row_norm(u_trace(:, cols));
	for sample_now = unique_samples
		rows_now = sample_idx == sample_now;
		plot(trace_t(rows_now), block_norm(rows_now), ...
			'Color', colors(block_idx, :), 'LineWidth', 0.65, ...
			'HandleVisibility', 'off');
	end
	grid on;
	ylabel(sprintf('||u_%d||_2', block_idx));
	title(sprintf('Increment control block u_%d', block_idx));
	xlim([min(trace_t), max(trace_t)]);
	if block_idx == n_blocks
		xlabel('s (including RK4 stages)');
	end
end
sgtitle('Third-level final QP correction by increment control block (all samples)');

if isfield(cfg, 'output') && cfg.output.enabled
	export_graphics_compat(fig_all, fullfile(output_dir, ...
		'third_level_u_by_block_all_samples_matlab.emf'));
	export_graphics_compat(fig_all, fullfile(output_dir, ...
		'third_level_u_by_block_all_samples_matlab.png'));
end
end

function values = ordered_field(trace, field_name, order, rows, n_cols)
values = [];
if ~isfield(trace, field_name) || isempty(trace.(field_name))
	return;
end
candidate = trace.(field_name);
if size(candidate, 1) ~= numel(order) || size(candidate, 2) ~= n_cols
	return;
end
candidate = candidate(order, :);
values = candidate(rows, :);
end

function plot_if_available(s, values, line_style, line_width, label)
if isempty(values)
	return;
end
plot(s, row_norm(values), line_style, 'LineWidth', line_width, ...
	'DisplayName', label);
end

function value = max_or_nan(values)
if isempty(values)
	value = NaN;
else
	value = max(values);
end
end

function values = row_norm(matrix)
if isempty(matrix)
	values = [];
else
	values = sqrt(sum(matrix .^ 2, 2));
end
end
