% Maximum discrete tangent-continuity error at internal anchors.
function max_error = max_anchor_tangent_error(points, anchor_idx)
n_points = size(points, 1);
anchor_idx = unique(anchor_idx(:)');
internal_anchor_idx = anchor_idx(anchor_idx > 1 & anchor_idx < n_points);
if isempty(internal_anchor_idx)
    max_error = 0.0;
    return;
end

max_error = 0.0;
for anchor = internal_anchor_idx
    left_tangent = points(anchor, :, :) - points(anchor - 1, :, :);
    right_tangent = points(anchor + 1, :, :) - points(anchor, :, :);
    tangent_error = right_tangent - left_tangent;
    max_error = max(max_error, max(abs(tangent_error), [], 'all'));
end
end
