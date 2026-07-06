function enabled = slack_enabled_for_constraints(constraint_cfg, constraint_types)
global_enabled = struct_field_default(constraint_cfg, 'slack_enabled', true);
enabled = false(numel(constraint_types), 1);
for idx = 1:numel(constraint_types)
	switch constraint_types(idx)
		case "integral"
			enabled(idx) = struct_field_default(constraint_cfg, ...
				'hocbf_slack_enabled', global_enabled);
		case "terminal"
			enabled(idx) = struct_field_default(constraint_cfg, ...
				'terminal_variance_slack_enabled', global_enabled);
		case "anchor_clf"
			enabled(idx) = struct_field_default(constraint_cfg, ...
				'anchor_clf_slack_enabled', global_enabled);
		otherwise
			error('Unknown QP constraint type: %s', constraint_types(idx));
	end
end
end
