function plot_increment_control_traces(cfg, rollout_diagnostics, constraint_cfg)
% 按增量控制块绘制最终实际施加的 QP 修正控制 u1,...,u5。
% trace_u 保存的是所有局部 PTCLF/PTCBF 以及后续 HOCBF 修正完成后的
% 最终控制 u，不包含 GP nominal velocity mu。

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
unique_samples = unique(sample_idx)';

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
	'Units', 'normalized', 'Position', [0.10, 0.08, 0.78, 0.78]);
movegui(fig, 'center');
tiledlayout(n_blocks, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

colors = lines(n_blocks);
for block_idx = 1:n_blocks
	nexttile;
	hold on;
	cols = (block_idx - 1) * block_dim + (1:block_dim);
	u_norm = sqrt(sum(u_trace(:, cols) .^ 2, 2));
	for sample_now = unique_samples
		rows_now = sample_idx == sample_now;
		plot(trace_t(rows_now), u_norm(rows_now), ...
			'Color', colors(block_idx, :), 'LineWidth', 0.9, ...
			'HandleVisibility', 'off');
	end
	grid on;
	ylabel(sprintf('||u_%d||_2', block_idx));
	title(sprintf('Increment control block u_%d', block_idx));
	if block_idx == n_blocks
		xlabel('s');
	end
end
sgtitle('Third-level final QP correction by increment control block');

if isfield(cfg, 'output') && cfg.output.enabled
	this_file = mfilename('fullpath');
	output_dir = fullfile(fileparts(this_file), 'outputs');
	if ~exist(output_dir, 'dir'); mkdir(output_dir); end
	output_path = fullfile(output_dir, ...
		'third_level_increment_control_traces_matlab.emf');
	export_graphics_compat(fig, output_path);
end
end
