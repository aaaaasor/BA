function enabled = slack_enabled_for_constraints(constraint_cfg, constraint_types, t)
global_enabled = struct_field_default(constraint_cfg, 'slack_enabled', true);
if nargin < 3
	t = -inf;
end
enabled = false(numel(constraint_types), 1);
for idx = 1:numel(constraint_types)
	switch constraint_types(idx)
		case "integral"
			enabled(idx) = struct_field_default(constraint_cfg, ...
				'hocbf_slack_enabled', global_enabled);
		case "terminal"
			enabled(idx) = struct_field_default(constraint_cfg, ...
				'terminal_variance_slack_enabled', global_enabled);
			enabled(idx) = enabled(idx) || local_late_stage_enabled( ...
				constraint_cfg, 'terminal_variance_slack_late_start_time', t);
		case "anchor_clf"
			enabled(idx) = struct_field_default(constraint_cfg, ...
				'anchor_clf_slack_enabled', global_enabled);
			% 生成后期把 slack 从 CLF 挪到 PTCBF：CLF 到了后期反而变硬，
			% 逼着它真正精确收敛到目标点，不再靠 slack 偷懒。
			enabled(idx) = enabled(idx) && ~local_late_stage_enabled( ...
				constraint_cfg, 'anchor_clf_slack_hard_after_time', t);
		otherwise
			error('Unknown QP constraint type: %s', constraint_types(idx));
	end
end
end

function is_late = local_late_stage_enabled(constraint_cfg, field_name, t)
% 只在 t >= 配置的起始时刻之后，才把该约束的 slack 打开（"生成后期"）。
% 未配置该字段时，默认 inf（永不因为"后期"而额外开启）。
late_start_time = struct_field_default(constraint_cfg, field_name, inf);
is_late = t >= late_start_time;
end
