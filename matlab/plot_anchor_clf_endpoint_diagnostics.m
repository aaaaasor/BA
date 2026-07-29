function plot_anchor_clf_endpoint_diagnostics(cfg, level_label, rollout_diagnostics)
if isempty(rollout_diagnostics) || ~isfield(rollout_diagnostics, 'hocbf')
	return;
end
trace = rollout_diagnostics.hocbf;
required = {'trace_anchor_clf_first_v', 'trace_anchor_clf_last_v', ...
	'trace_anchor_clf_first_raw_residual', ...
	'trace_anchor_clf_last_raw_residual', ...
	'trace_anchor_clf_first_effective_bound', ...
	'trace_anchor_clf_last_effective_bound', ...
	'trace_anchor_clf_first_owner_grad_norm', ...
	'trace_anchor_clf_last_owner_grad_norm'};
if ~all(isfield(trace, required)) || isempty(trace.trace_anchor_clf_first_v)
	return;
end

sample_idx = trace.trace_sample_idx(:);
step_idx = trace.trace_step_idx(:);
stage_idx = trace.trace_stage_idx(:);
trace_t = trace.trace_t(:);
[~, order] = sortrows([sample_idx, step_idx, stage_idx]);
sample_idx = sample_idx(order);
trace_t = trace_t(order);
unique_samples = unique(sample_idx)';

first = endpoint_trace(trace, 'first', order);
last = endpoint_trace(trace, 'last', order);
if ~any(isfinite(first.v)) && ~any(isfinite(last.v))
	return;
end

first_color = [0.10, 0.45, 0.85];
last_color = [0.85, 0.25, 0.15];
fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
	'Units', 'normalized', 'Position', [0.10, 0.08, 0.76, 0.78]);
movegui(fig, 'center');
tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
plot_samples(unique_samples, sample_idx, trace_t, first.v, first_color, '-');
plot_samples(unique_samples, sample_idx, trace_t, first.gbar, first_color, '--');
plot_samples(unique_samples, sample_idx, trace_t, last.v, last_color, '-');
plot_samples(unique_samples, sample_idx, trace_t, last.gbar, last_color, '--');
legend_handles(first_color, last_color, 'g', 'gbar');
ylabel('g, gbar');
title('Endpoint PTCLF state and envelope');
finish_axis();

nexttile;
hold on;
plot_samples(unique_samples, sample_idx, trace_t, first.bound, first_color, ':');
plot_samples(unique_samples, sample_idx, trace_t, last.bound, last_color, ':');
plot_samples(unique_samples, sample_idx, trace_t, first.effective_bound, ...
	first_color, '-');
plot_samples(unique_samples, sample_idx, trace_t, last.effective_bound, ...
	last_color, '-');
plot(nan, nan, ':', 'Color', first_color, 'DisplayName', 'P1 full bound');
plot(nan, nan, '-', 'Color', first_color, 'DisplayName', 'P1 effective bound');
plot(nan, nan, ':', 'Color', last_color, 'DisplayName', 'P5 full bound');
plot(nan, nan, '-', 'Color', last_color, 'DisplayName', 'P5 effective bound');
yline(0, ':', 'Color', [0.25, 0.25, 0.25], 'HandleVisibility', 'off');
ylabel('QP bound');
title('Split-row bounds after fixing earlier increment controls');
finish_axis();

nexttile;
hold on;
plot_samples(unique_samples, sample_idx, trace_t, first.raw, first_color, ':');
plot_samples(unique_samples, sample_idx, trace_t, first.residual, first_color, '-');
plot_samples(unique_samples, sample_idx, trace_t, last.raw, last_color, ':');
plot_samples(unique_samples, sample_idx, trace_t, last.residual, last_color, '-');
plot_samples(unique_samples, sample_idx, trace_t, first.slack, first_color, '--');
plot_samples(unique_samples, sample_idx, trace_t, last.slack, last_color, '--');
yline(0, '-', 'Color', [0.15, 0.15, 0.15], 'DisplayName', '0');
plot(nan, nan, ':', 'Color', first_color, 'DisplayName', 'P1 raw');
plot(nan, nan, '-', 'Color', first_color, 'DisplayName', 'P1 relaxed');
plot(nan, nan, ':', 'Color', last_color, 'DisplayName', 'P5 raw');
plot(nan, nan, '-', 'Color', last_color, 'DisplayName', 'P5 relaxed');
plot(nan, nan, '--', 'Color', [0.35, 0.35, 0.35], ...
	'DisplayName', 'slack (endpoint color)');
ylabel('residual, slack');
title('Split-row residuals (relaxed residual should be <= 0)');
finish_axis();

nexttile;
hold on;
plot_samples(unique_samples, sample_idx, trace_t, first.grad, first_color, '-');
plot_samples(unique_samples, sample_idx, trace_t, last.grad, last_color, '-');
plot_samples(unique_samples, sample_idx, trace_t, first.full_grad, first_color, ':');
plot_samples(unique_samples, sample_idx, trace_t, last.full_grad, last_color, ':');
plot(nan, nan, '-', 'Color', first_color, 'DisplayName', 'P1 owner ||grad g||');
plot(nan, nan, '-', 'Color', last_color, 'DisplayName', 'P5 owner ||grad g||');
plot(nan, nan, ':', 'Color', first_color, 'DisplayName', 'P1 full ||grad g||');
plot(nan, nan, ':', 'Color', last_color, 'DisplayName', 'P5 full ||grad g||');
xlabel('s');
ylabel('norm');
title('Row leverage');
finish_axis();

sgtitle([level_label, ': split endpoint PTCLF diagnostics'], ...
	'Interpreter', 'none');
if cfg.output.enabled
	output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
	if ~exist(output_dir, 'dir'); mkdir(output_dir); end
	output_path = fullfile(output_dir, ...
		['trajectory_anchor_clf_endpoint_diagnostics_', ...
		strrep(level_label, ' ', '_'), '_matlab.emf']);
	export_graphics_compat(fig, output_path);
end
end

function data = endpoint_trace(trace, endpoint_name, order)
prefix = ['trace_anchor_clf_', endpoint_name, '_'];
data.v = ordered_field(trace, [prefix, 'v'], order);
data.gbar = ordered_field(trace, [prefix, 'ptzf_bound'], order);
data.bound = ordered_field(trace, [prefix, 'bound'], order);
data.effective_bound = ordered_field(trace, ...
	[prefix, 'effective_bound'], order);
data.raw = ordered_field(trace, [prefix, 'raw_residual'], order);
data.residual = ordered_field(trace, [prefix, 'residual'], order);
data.slack = ordered_field(trace, [prefix, 'slack'], order);
data.full_grad = ordered_field(trace, [prefix, 'grad_norm'], order);
data.grad = ordered_field(trace, [prefix, 'owner_grad_norm'], order);
end

function values = ordered_field(trace, field_name, order)
values = trace.(field_name);
values = values(order);
end

function plot_samples(samples, sample_idx, trace_t, values, color, style)
for sample_now = samples
	rows = sample_idx == sample_now;
	plot(trace_t(rows), values(rows), style, 'Color', color, ...
		'LineWidth', 0.9, 'HandleVisibility', 'off');
end
end

function legend_handles(first_color, last_color, value_name, bound_name)
plot(nan, nan, '-', 'Color', first_color, ...
	'DisplayName', ['P1 ', value_name]);
plot(nan, nan, '--', 'Color', first_color, ...
	'DisplayName', ['P1 ', bound_name]);
plot(nan, nan, '-', 'Color', last_color, ...
	'DisplayName', ['P5 ', value_name]);
plot(nan, nan, '--', 'Color', last_color, ...
	'DisplayName', ['P5 ', bound_name]);
end

function finish_axis()
grid on;
legend('Location', 'best', 'Interpreter', 'none');
end
