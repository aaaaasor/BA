% Refine coarse 2D trajectories by linear sampling between anchors.
function refined_points = refine_trajectory_points(coarse_points, ...
    n_points_per_segment)
%% Dimensions
if nargin < 2 || isempty(n_points_per_segment)
    n_points_per_segment = 5;
end
n_coarse_points = size(coarse_points, 1);
n_trajectories = size(coarse_points, 2);
n_segments = n_coarse_points - 1;
n_refined_points = n_segments * (n_points_per_segment - 1) + 1;
refined_points = zeros(n_refined_points, n_trajectories, 2);

%% Refine Trajectories
for traj_idx = 1:n_trajectories
    coarse_curve = squeeze(coarse_points(:, traj_idx, :));
    refined_curve = sample_curve_linearly(coarse_curve, ...
        n_points_per_segment);
    refined_points(:, traj_idx, :) = refined_curve;
end
end

function refined_curve = sample_curve_linearly(coarse_curve, ...
    n_points_per_segment)
n_coarse_points = size(coarse_curve, 1);
n_segments = n_coarse_points - 1;
n_refined_points = n_segments * (n_points_per_segment - 1) + 1;
refined_curve = zeros(n_refined_points, 2);
write_idx = 1;
for segment_idx = 1:n_segments
    start_point = coarse_curve(segment_idx, :);
    end_point = coarse_curve(segment_idx + 1, :);
    t = linspace(0, 1, n_points_per_segment)';
    segment_curve = (1.0 - t) .* start_point + t .* end_point;
    if segment_idx == 1
        refined_curve(write_idx:(write_idx + n_points_per_segment - 1), :) = ...
            segment_curve;
        write_idx = write_idx + n_points_per_segment;
    else
        segment_tail = segment_curve(2:end, :);
        n_tail_points = size(segment_tail, 1);
        refined_curve(write_idx:(write_idx + n_tail_points - 1), :) = ...
            segment_tail;
        write_idx = write_idx + n_tail_points;
    end
end
anchor_idx = 1:(n_points_per_segment - 1):n_refined_points;
refined_curve(anchor_idx, :) = coarse_curve;
end
