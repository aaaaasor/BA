% Stitch anchor-to-anchor segment samples into full 2D trajectories.
function trajectory_points = stitch_segment_points(segment_data, ...
    n_trajectories, n_segments, n_points_per_segment)
%% Dimensions
n_full_points = n_segments * (n_points_per_segment - 1) + 1;
trajectory_points = zeros(n_full_points, n_trajectories, 2);

%% Stitch Segments
for traj_idx = 1:n_trajectories
    write_idx = 1;
    for segment_idx = 1:n_segments
        sample_idx = (traj_idx - 1) * n_segments + segment_idx;
        segment_curve = reshape(segment_data(sample_idx, :), 2, [])';
        if segment_idx == 1
            trajectory_points(write_idx:(write_idx + n_points_per_segment - 1), ...
                traj_idx, :) = segment_curve;
            write_idx = write_idx + n_points_per_segment;
        else
            segment_tail = segment_curve(2:end, :);
            n_tail_points = size(segment_tail, 1);
            trajectory_points(write_idx:(write_idx + n_tail_points - 1), ...
                traj_idx, :) = segment_tail;
            write_idx = write_idx + n_tail_points;
        end
    end
end
end
