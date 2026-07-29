function weights = slack_weights_for_constraints(constraint_cfg, constraint_types)
weights = zeros(numel(constraint_types), 1);
for idx = 1:numel(constraint_types)
	switch constraint_types(idx)
		case "integral"
			% HOCBF 是累计方差安全约束，通常应设置较大的权重。
			weights(idx) = struct_field_default(constraint_cfg, ...
				'hocbf_slack_weight', 1e8);
		case "terminal"
			% Terminal PTCBF 是末端方差安全约束，通常也应设置较大的权重。
			weights(idx) = struct_field_default(constraint_cfg, ...
				'terminal_variance_slack_weight', 1e8);
		case "anchor_clf"
			% PTCLF 是 anchor tracking/收敛约束，通常可以比 CBF 更软。
			weights(idx) = struct_field_default(constraint_cfg, ...
				'anchor_clf_slack_weight', 1e4);
		case "anchor_clf_first"
			weights(idx) = struct_field_default(constraint_cfg, ...
				'anchor_clf_first_slack_weight', ...
				struct_field_default(constraint_cfg, ...
				'anchor_clf_slack_weight', 1e4));
		case "anchor_clf_last"
			weights(idx) = struct_field_default(constraint_cfg, ...
				'anchor_clf_last_slack_weight', ...
				struct_field_default(constraint_cfg, ...
				'anchor_clf_slack_weight', 1e4));
		case "obstacle"
			% 障碍 CBF slack 只作可行性兜底，权重设大 -> 近似硬约束。
			weights(idx) = struct_field_default(constraint_cfg, ...
				'obstacle_slack_weight', 1e4);
		otherwise
			error('Unknown QP constraint type: %s', constraint_types(idx));
	end
end
weights = max(weights, eps);
end
