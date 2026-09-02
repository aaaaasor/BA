function subset = subset_rollout_rows(base_constraint, row_indices)
%SUBSET_ROLLOUT_ROWS Select explicitly indexed sample-specific constraint rows.

row_indices = row_indices(:);
subset = base_constraint;
per_sample_fields = { ...
	'anchor_clf_targets', ...
	'track_boundary_reference_s_min_targets', ...
	'track_boundary_reference_s_max_targets'};
for idx = 1:numel(per_sample_fields)
	name = per_sample_fields{idx};
	if isfield(subset, name)
		subset.(name) = subset.(name)(row_indices, :);
	end
end
subset.live_trajectory_plot_enabled = false;
end
