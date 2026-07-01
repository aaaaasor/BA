function segment_state = anchor_segment_to_increment_state(start_point, ...
    end_point, n_points_per_segment, feature_dim)
segment_state = zeros(n_points_per_segment, feature_dim);
segment_state(1, :) = start_point;
if n_points_per_segment > 1
    increment_xy = (end_point(1:2) - start_point(1:2)) ./ ...
        (n_points_per_segment - 1);
    for point_idx = 2:n_points_per_segment
        segment_state(point_idx, :) = end_point;
        segment_state(point_idx, 1:2) = increment_xy;
    end
end
end
