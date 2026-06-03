% Stitch overlapping 5-point window predictions into full dense trajectories.
function trajectory_points = stitch_sliding_window_points(window_data, ...
    n_trajectories, n_windows, n_points_per_window)
%% Dimensions
n_full_points = n_windows + n_points_per_window - 1;
trajectory_points = zeros(n_full_points, n_trajectories, 2);
point_counts = zeros(n_full_points, n_trajectories);

%% Average Overlapping Window Predictions
for window_idx = 1:n_windows
    for traj_idx = 1:n_trajectories
        sample_idx = (window_idx - 1) * n_trajectories + traj_idx;
        window_curve = reshape(window_data(sample_idx, :), 2, [])';
        point_idx_set = window_idx:(window_idx + n_points_per_window - 1);
        trajectory_points(point_idx_set, traj_idx, :) = ...
            trajectory_points(point_idx_set, traj_idx, :) + ...
            reshape(window_curve, [], 1, 2);
        point_counts(point_idx_set, traj_idx) = ...
            point_counts(point_idx_set, traj_idx) + 1;
    end
end

for traj_idx = 1:n_trajectories
    for point_idx = 1:n_full_points
        if point_counts(point_idx, traj_idx) > 0
            trajectory_points(point_idx, traj_idx, :) = ...
                trajectory_points(point_idx, traj_idx, :) ./ ...
                point_counts(point_idx, traj_idx);
        end
    end
end
end
