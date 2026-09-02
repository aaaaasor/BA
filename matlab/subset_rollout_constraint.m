function subset = subset_rollout_constraint(base_constraint, parent_indices, n_segments)
%SUBSET_ROLLOUT_CONSTRAINT Select sample-specific rows for chosen parents.
% The returned constraint is suitable for a local rerun whose parent rows
% are packed densely in the same order as parent_indices.

parent_indices = parent_indices(:);
subset = base_constraint;
row_indices = zeros(numel(parent_indices) * n_segments, 1);
for idx = 1:numel(parent_indices)
	local_rows = (idx - 1) * n_segments + (1:n_segments);
	row_indices(local_rows) = ...
		(parent_indices(idx) - 1) * n_segments + (1:n_segments);
end

per_sample_fields = { ...
	'anchor_clf_targets', ...
	'track_boundary_reference_s_min_targets', ...
	'track_boundary_reference_s_max_targets'};
for field_idx = 1:numel(per_sample_fields)
	field_name = per_sample_fields{field_idx};
	if isfield(subset, field_name)
		values = subset.(field_name);
		subset.(field_name) = values(row_indices, :);
	end
end

if isfield(subset, 'live_trajectory_plot_sample_indices')
	requested = subset.live_trajectory_plot_sample_indices;
	[~, local_indices] = ismember(requested, row_indices);
	subset.live_trajectory_plot_sample_indices = ...
		reshape(local_indices, size(requested));
	subset.live_trajectory_original_sample_indices = row_indices;
	if ~any(local_indices)
		subset.live_trajectory_plot_enabled = false;
	end
end
end
