% Stitch anchor-to-anchor segment samples into full feature trajectories.
function trajectory_points = stitch_segment_points(segment_data, ...
    n_trajectories, n_segments, n_points_per_segment)
n_full_points = n_segments * (n_points_per_segment - 1) + 1;
feature_dim = size(segment_data, 2) / n_points_per_segment;
trajectory_points = zeros(n_full_points, n_trajectories, feature_dim);
for traj_idx = 1:n_trajectories
    for segment_idx = 1:n_segments
        sample_idx = (traj_idx - 1) * n_segments + segment_idx;
        segment_curve = reshape(segment_data(sample_idx, :), feature_dim, [])';
        if segment_idx == 1
            trajectory_points(1:n_points_per_segment, traj_idx, :) = segment_curve;
        else
            start_idx = (segment_idx - 1) * (n_points_per_segment - 1) + 2;
            trajectory_points(start_idx:(start_idx + n_points_per_segment - 2), ...
                traj_idx, :) = segment_curve(2:end, :);
        end
    end
end
end
