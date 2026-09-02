function [constraint, targets, matrix, offset, owner_std, parent_segment_data] = ...
	refresh_third_level_anchor_constraint(constraint, final_segment_data, ...
	parent_segment_rows, n_parents, n_parent_segments, points_per_segment, ...
	data_transform)
%REFRESH_THIRD_LEVEL_ANCHOR_CONSTRAINT Rebuild all L3 endpoint targets.

parent_segment_data = final_segment_data(parent_segment_rows, :);
[targets, matrix, offset, owner_std] = ...
	build_increment_endpoint_clf_targets(parent_segment_data, n_parents, 1, ...
	n_parent_segments, points_per_segment, data_transform);
error_scale = 1.0 ./ owner_std(:);
if struct_field_default(constraint, 'ptclf_enabled', false)
	constraint.anchor_clf_targets = targets .* error_scale';
	constraint.anchor_clf_matrix = matrix .* error_scale;
	constraint.anchor_clf_offset = offset .* error_scale;
else
	constraint.anchor_clf_targets = targets;
	constraint.anchor_clf_matrix = matrix;
	constraint.anchor_clf_offset = offset;
end
end
