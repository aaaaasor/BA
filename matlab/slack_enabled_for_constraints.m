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
			enabled(idx) = enabled(idx) || local_late_stage_enabled( ...
				constraint_cfg, 'hocbf_slack_late_start_time', t);
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
		case "anchor_clf_first"
			enabled(idx) = struct_field_default(constraint_cfg, ...
				'anchor_clf_slack_enabled', global_enabled);
			enabled(idx) = enabled(idx) && ~local_late_stage_enabled( ...
				constraint_cfg, 'anchor_clf_slack_hard_after_time', t);
		case "anchor_clf_last"
			enabled(idx) = struct_field_default(constraint_cfg, ...
				'anchor_clf_slack_enabled', global_enabled);
			enabled(idx) = enabled(idx) && ~local_late_stage_enabled( ...
				constraint_cfg, 'anchor_clf_slack_hard_after_time', t);
		case "obstacle"
			% 前期软、后期硬：t < switch 时带 slack（配合此时 variance
			% 硬，让 flow matching 先把分布捏出来）；t >= switch 后变硬
			% （此时 variance 转软，让避障优先）。与 hocbf/terminal 的
			% "前硬后软"方向相反，和 anchor_clf 同向。未配置切换时刻时
			% 默认 inf，退化为始终跟随 obstacle_slack_enabled（近似硬）。
			enabled(idx) = struct_field_default(constraint_cfg, ...
				'obstacle_slack_enabled', global_enabled);
			enabled(idx) = enabled(idx) && ~local_late_stage_enabled( ...
				constraint_cfg, 'obstacle_slack_hard_after_time', t);
		case "boundary"
			enabled(idx) = struct_field_default(constraint_cfg, ...
				'track_boundary_slack_enabled', false);
			enabled(idx) = enabled(idx) && ~local_late_stage_enabled( ...
				constraint_cfg, ...
				'track_boundary_slack_hard_after_time', t);
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
