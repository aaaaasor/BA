function output_path = animate_third_level_diagnostics(cfg, rollout_times, ...
	segment_plot_path, source_points, rollout_diagnostics, constraint_cfg, ...
	selected_sample_ids)
% 单段三级控制链诊断：PTCLF reference -> obstacle PTCBF -> HOCBF post-filter。
% 默认自动选择三级控制链中任一控制/修正峰值最大的 segment，也可用
% cfg.animation.third_level_diagnostic_segment 指定所选 segment 的序号。

output_path = "";
if ~isfield(rollout_diagnostics, 'hocbf') || ...
		empty_trace(rollout_diagnostics.hocbf)
	return;
end
trace = rollout_diagnostics.hocbf;
required_fields = {'trace_u', 'trace_u_ptclf_reference', ...
	'trace_u_after_ptcbf', 'trace_u_ptcbf_correction', ...
	'trace_u_hocbf_correction', 'trace_grad_norm', ...
	'trace_stage_hocbf_residual_after_ptcbf'};
for field_idx = 1:numel(required_fields)
	if ~isfield(trace, required_fields{field_idx}) || ...
			empty_trace_field(trace.(required_fields{field_idx}))
		error(['Third-level cascade diagnostics are missing ', ...
		required_fields{field_idx}, '. Rerun the third-level rollout.']);
	end
end

selected_sample_ids = selected_sample_ids(:)';
n_selected = numel(selected_sample_ids);
trace_sample_idx = trace.trace_sample_idx(:);
trace_t = trace.trace_t(:);
segment_position = struct_field_default(cfg.animation, ...
	'third_level_diagnostic_segment', 0);
if segment_position < 1 || segment_position > n_selected
	segment_position = select_worst_segment(trace, trace_sample_idx, ...
		selected_sample_ids);
end
sample_id = selected_sample_ids(segment_position);
rows = find(trace_sample_idx == sample_id);
if isempty(rows)
	error('No third-level diagnostic trace found for sample %d.', sample_id);
end

block_dim = struct_field_default(constraint_cfg, ...
	'increment_control_block_dim', 0);
n_blocks = struct_field_default(constraint_cfg, ...
	'increment_control_block_count', 0);
if block_dim <= 0 || n_blocks <= 0 || ...
		block_dim * n_blocks ~= size(trace.trace_u, 2)
	error('Third-level cascade diagnostics require a valid increment block shape.');
end

t_trace = trace_t(rows);
u_ref = trace.trace_u_ptclf_reference(rows, :);
u_ptcbf = trace.trace_u_after_ptcbf(rows, :);
u_final = trace.trace_u(rows, :);
du_ptcbf = trace.trace_u_ptcbf_correction(rows, :);
du_hocbf = trace.trace_u_hocbf_correction(rows, :);
ref_blocks = control_block_norms(u_ref, block_dim, n_blocks);
ptcbf_blocks = control_block_norms(du_ptcbf, block_dim, n_blocks);
hocbf_blocks = control_block_norms(du_hocbf, block_dim, n_blocks);
stage_max = [max(ref_blocks, [], 2), ...
	max(control_block_norms(u_ptcbf, block_dim, n_blocks), [], 2), ...
	max(control_block_norms(u_final, block_dim, n_blocks), [], 2)];
stage_plot = stage_max;
if isfield(trace, 'trace_obstacle_filter_active') && ...
		numel(trace.trace_obstacle_filter_active) >= max(rows)
	obstacle_filter_active = logical( ...
		trace.trace_obstacle_filter_active(rows));
else
	obstacle_enabled = struct_field_default(constraint_cfg, ...
		'obstacle_enabled', false);
	obstacle_start = struct_field_default(constraint_cfg, ...
		'obstacle_activation_time', inf);
	obstacle_filter_active = obstacle_enabled & t_trace >= obstacle_start;
end
stage_plot(~obstacle_filter_active, 2) = nan;
if isfield(trace, 'trace_hocbf_filter_active') && ...
		numel(trace.trace_hocbf_filter_active) >= max(rows)
	hocbf_active = logical(trace.trace_hocbf_filter_active(rows));
	stage_plot(~hocbf_active, 3) = nan;
end
correction_max = [max(ptcbf_blocks, [], 2), max(hocbf_blocks, [], 2)];
correction_max(~obstacle_filter_active, 1) = nan;
if exist('hocbf_active', 'var')
	correction_max(~hocbf_active, 2) = nan;
end
hocbf_row_norm = trace.trace_grad_norm(rows);
hocbf_prefilter_residual = ...
	trace.trace_stage_hocbf_residual_after_ptcbf(rows);
% On a log axis, show only the positive part: residual <= 0 means the
% post-filter receives an already feasible HOCBF row and needs no action.
hocbf_trigger = [hocbf_row_norm, max(hocbf_prefilter_residual, eps)];
if exist('hocbf_active', 'var')
	hocbf_trigger(~hocbf_active, :) = nan;
end

n_points = cfg.animation.segment_plot_points_per_segment;
feature_dim = size(segment_plot_path, 3) / n_points;
source_start = (segment_position - 1) * n_points + 1;
source_end = source_start + n_points - 1;
source_curve = source_points(source_start:source_end, 1:2);

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
	'Units', 'normalized', 'Position', [0.03, 0.05, 0.94, 0.86]);
movegui(fig, 'center');
layout = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax_traj = nexttile(layout, 1, [3, 1]); hold(ax_traj, 'on'); grid(ax_traj, 'on');
plot(ax_traj, source_curve(:, 1), source_curve(:, 2), '--', ...
	'Color', [0.45 0.45 0.45], 'LineWidth', 1.1, 'DisplayName', 'source segment');
h_curve = plot(ax_traj, nan, nan, '.-', 'Color', [0.05 0.35 0.90], ...
	'LineWidth', 1.7, 'MarkerSize', 13, 'DisplayName', 'rollout segment');
if isfield(cfg, 'obstacle'); draw_obstacles_on_axes(ax_traj, cfg.obstacle); end
all_xy = [source_curve; collect_selected_xy(segment_plot_path, ...
	segment_position, n_points, feature_dim)];
pad = 0.08 .* max(max(all_xy, [], 1) - min(all_xy, [], 1), eps);
xlim(ax_traj, [min(all_xy(:, 1)) - pad(1), max(all_xy(:, 1)) + pad(1)]);
ylim(ax_traj, [min(all_xy(:, 2)) - pad(2), max(all_xy(:, 2)) + pad(2)]);
axis(ax_traj, 'equal'); xlabel(ax_traj, 'x'); ylabel(ax_traj, 'y');
legend(ax_traj, 'Location', 'best', 'AutoUpdate', 'off');
status_text = text(ax_traj, 0.02, 0.98, '', 'Units', 'normalized', ...
	'VerticalAlignment', 'top', 'FontName', 'Consolas', 'FontSize', 9, ...
	'BackgroundColor', 'w', 'Margin', 4);

ax_stage = nexttile(layout, 2); hold(ax_stage, 'on'); grid(ax_stage, 'on');
stage_labels = {'PTCLF reference', 'after PTCBF (active only)', ...
	'after HOCBF (active only)'};
stage_colors = [0.45 0.20 0.75; 0.90 0.45 0.05; 0.10 0.55 0.25];
h_stage = make_lines(ax_stage, 3, stage_labels, stage_colors);
set(ax_stage, 'YScale', 'log'); xlabel(ax_stage, 's'); ylabel(ax_stage, 'max_i ||u_i||_2');
title(ax_stage, 'Actual control after each stage'); legend(ax_stage, 'Location', 'best');
xlim(ax_stage, [min(rollout_times), max(rollout_times)]);
set_fixed_log_limits(ax_stage, stage_plot);

ax_hocbf = nexttile(layout, 4); hold(ax_hocbf, 'on'); grid(ax_hocbf, 'on');
h_hocbf = make_lines(ax_hocbf, 2, ...
	{'||A_H||_2', 'max(0, A_H u_{PTCBF} - b_H)'}, ...
	[0.15 0.35 0.85; 0.75 0.15 0.15]);
set(ax_hocbf, 'YScale', 'log'); xlabel(ax_hocbf, 's');
ylabel(ax_hocbf, 'HOCBF trigger value');
title(ax_hocbf, 'HOCBF row effectiveness and pre-filter violation');
legend(ax_hocbf, 'Location', 'best');
xlim(ax_hocbf, [min(rollout_times), max(rollout_times)]);
set_fixed_log_limits(ax_hocbf, hocbf_trigger);

ax_corr = nexttile(layout, 6); hold(ax_corr, 'on'); grid(ax_corr, 'on');
h_corr = make_lines(ax_corr, 2, {'Delta u PTCBF', 'Delta u HOCBF'}, ...
	stage_colors(2:3, :));
set(ax_corr, 'YScale', 'log'); xlabel(ax_corr, 's'); ylabel(ax_corr, 'max_i ||Delta u_i||_2');
title(ax_corr, 'Added correction by filter'); legend(ax_corr, 'Location', 'best');
xlim(ax_corr, [min(rollout_times), max(rollout_times)]);
set_fixed_log_limits(ax_corr, correction_max);

axes_to_mark = [ax_stage, ax_hocbf, ax_corr];
mark_switch_times(axes_to_mark, constraint_cfg);

frame_indices = 1:cfg.animation.frame_stride:numel(rollout_times);
if frame_indices(end) ~= numel(rollout_times)
	frame_indices(end + 1) = numel(rollout_times);
end
output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
if cfg.output.enabled && ~exist(output_dir, 'dir'); mkdir(output_dir); end
output_path = fullfile(output_dir, 'third_level_diagnostics_dashboard.gif');

for frame_nr = 1:numel(frame_indices)
	time_idx = frame_indices(frame_nr);
	s_now = rollout_times(time_idx);
	curve_row = reshape(segment_plot_path(time_idx, segment_position, :), 1, []);
	curve = reshape(curve_row, feature_dim, [])';
	set(h_curve, 'XData', curve(:, 1), 'YData', curve(:, 2));
	history = find(t_trace <= s_now + 10 * eps(max(1, abs(s_now))));
	if isempty(history); history = 1; end
	set_history(h_stage, t_trace, stage_plot, history, true);
	set_history(h_hocbf, t_trace, hocbf_trigger, history, true);
	set_history(h_corr, t_trace, correction_max, history, true);
	[~, current_pos] = min(abs(t_trace - s_now));
	current_levels = [stage_max(current_pos, 1), ...
		correction_max(current_pos, 1), correction_max(current_pos, 2)];
	[dominant_value, dominant_idx] = max(current_levels);
	dominant_labels = {'PTCLF reference', 'PTCBF correction', 'HOCBF correction'};
	set(status_text, 'String', sprintf([ ...
		'segment position = %d / %d\nsample id = %d\ns = %.3f\n', ...
		'dominant = %s\npeak = %.3g'], segment_position, n_selected, ...
		sample_id, s_now, dominant_labels{dominant_idx}, dominant_value));
	title(ax_traj, sprintf('Selected third-level segment, s = %.3f', s_now));
	title(layout, sprintf( ...
		'Third-level cascade diagnosis: segment %d (sample %d)', ...
		segment_position, sample_id));
	drawnow;
	frame = getframe(fig);
	[image_rgb, ~] = frame2im(frame);
	[image_indexed, color_map] = rgb2ind(image_rgb, 256);
	if cfg.output.enabled
		if frame_nr == 1
			imwrite(image_indexed, color_map, output_path, 'gif', ...
				'LoopCount', inf, 'DelayTime', cfg.animation.delay_time);
		else
			imwrite(image_indexed, color_map, output_path, 'gif', ...
				'WriteMode', 'append', 'DelayTime', cfg.animation.delay_time);
		end
	end
end
end

function tf = empty_trace(trace)
tf = ~isfield(trace, 'trace_t') || isempty(trace.trace_t);
end

function tf = empty_trace_field(value)
tf = isempty(value) || size(value, 1) == 0;
end

function position = select_worst_segment(trace, sample_indices, sample_ids)
peaks = -inf(size(sample_ids));
fields = {'trace_u_ptclf_reference', 'trace_u_ptcbf_correction', ...
	'trace_u_hocbf_correction', 'trace_u'};
for idx = 1:numel(sample_ids)
	rows = sample_indices == sample_ids(idx);
	if any(rows)
		for field_idx = 1:numel(fields)
			u = trace.(fields{field_idx})(rows, :);
			peaks(idx) = max(peaks(idx), ...
				max(sqrt(sum(u .^ 2, 2)), [], 'omitnan'));
		end
	end
end
[~, position] = max(peaks);
end

function result = control_block_norms(u, block_dim, n_blocks)
result = nan(size(u, 1), n_blocks);
for block_idx = 1:n_blocks
	cols = (block_idx - 1) * block_dim + (1:block_dim);
	result(:, block_idx) = sqrt(sum(u(:, cols) .^ 2, 2));
end
end

function handles = make_lines(ax, count, labels, colors)
handles = gobjects(count, 1);
for idx = 1:count
	handles(idx) = plot(ax, nan, nan, 'Color', colors(idx, :), ...
		'LineWidth', 1.15, 'DisplayName', labels{idx});
end
end

function set_history(handles, t, data, history, use_log)
for idx = 1:numel(handles)
	y = data(history, idx);
	if use_log
		y(~isfinite(y)) = nan;
		y(y <= 0) = eps;
	end
	set(handles(idx), 'XData', t(history), 'YData', y);
end
end

function set_fixed_log_limits(ax, data)
values = data(isfinite(data) & data > 0);
if isempty(values)
	ylim(ax, [1e-16, 1]);
	return;
end
lower_limit = 10 ^ floor(log10(min(values)));
upper_limit = 10 ^ ceil(log10(max(values)));
if lower_limit == upper_limit
	lower_limit = lower_limit / 10;
	upper_limit = upper_limit * 10;
end
ylim(ax, [lower_limit, upper_limit]);
end

function mark_switch_times(axes_list, constraint_cfg)
times = [struct_field_default(constraint_cfg, 'obstacle_activation_time', nan), ...
	struct_field_default(constraint_cfg, 'hocbf_filter_end_time', nan)];
styles = {'--', ':'};
for ax = axes_list
	for idx = 1:numel(times)
		if isfinite(times(idx))
			xline(ax, times(idx), styles{idx}, 'Color', [0.45 0.45 0.45], ...
				'HandleVisibility', 'off');
		end
	end
end
end

function xy = collect_selected_xy(segment_path, segment_position, n_points, feature_dim)
xy = zeros(size(segment_path, 1) * n_points, 2);
for frame_idx = 1:size(segment_path, 1)
	row = reshape(segment_path(frame_idx, segment_position, :), 1, []);
	curve = reshape(row, feature_dim, [])';
	rows = (frame_idx - 1) * n_points + (1:n_points);
	xy(rows, :) = curve(:, 1:2);
end
end

function draw_obstacles_on_axes(ax, obstacle)
if isempty(obstacle) || ~isfield(obstacle, 'enabled') || ~obstacle.enabled
	return;
end
theta = linspace(0, 2 * pi, 200);
for obstacle_idx = 1:size(obstacle.centers, 2)
	center = obstacle.centers(:, obstacle_idx);
	fill(ax, center(1) + obstacle.semi_axes(1, obstacle_idx) * cos(theta), ...
		center(2) + obstacle.semi_axes(2, obstacle_idx) * sin(theta), ...
		[0.82 0.82 0.82], 'EdgeColor', [0.35 0.35 0.35], ...
		'FaceAlpha', 0.6, 'LineWidth', 1.0, 'HandleVisibility', 'off');
end
end
