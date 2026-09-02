function audit_50_tmp
root_dir = fileparts(mfilename('fullpath'));
S = load(fullfile(root_dir, 'outputs', ...
    'Racing_ThirdLevel_Rollout_SerialTest.mat'));
C = S.saved_third_segment_variance_constraint;
T = S.third_segment_data_transform;

states = squeeze(S.third_traj_path_10d(end, :, :));
states_local = states .* T.std' + T.mean';
states_global = local_increment_rows_to_global(states_local, ...
    T.feature_dim, 5);

n_trajectories = size(states_global, 1) / 16;
points = zeros(65, n_trajectories, 2);
for trajectory_idx = 1:n_trajectories
    for segment_idx = 1:16
        sample_idx = (trajectory_idx - 1) * 16 + segment_idx;
        curve = reshape(states_global(sample_idx, :), T.feature_dim, [])';
        if segment_idx == 1
            points(1:5, trajectory_idx, :) = curve(:, 1:2);
        else
            first_row = (segment_idx - 1) * 4 + 2;
            points(first_row:first_row + 3, trajectory_idx, :) = ...
                curve(2:5, 1:2);
        end
    end
end

left = C.track_boundary_geometry.curves(1).control_points;
right = C.track_boundary_geometry.curves(2).control_points;
track_x = [left(:, 1); flipud(right(:, 1))];
track_y = [left(:, 2); flipud(right(:, 2))];

outside_points = 0;
outside_point_samples = false(1, n_trajectories);
outside_segments = 0;
outside_segment_samples = false(1, n_trajectories);
obstacle_points = 0;
obstacle_point_samples = false(1, n_trajectories);
obstacle_segments = 0;
obstacle_segment_samples = false(1, n_trajectories);
obstacle_points_each = zeros(1, size(C.obstacle_physical_geometry.centers, 2));
obstacle_segments_each = zeros(size(obstacle_points_each));
max_abs_coordinate = zeros(1, n_trajectories);
max_segment_length = zeros(1, n_trajectories);
curve_length = zeros(1, n_trajectories);

for trajectory_idx = 1:n_trajectories
    curve = squeeze(points(:, trajectory_idx, :));
    segment_lengths = vecnorm(diff(curve, 1, 1), 2, 2);
    max_abs_coordinate(trajectory_idx) = max(abs(curve), [], 'all');
    max_segment_length(trajectory_idx) = max(segment_lengths);
    curve_length(trajectory_idx) = sum(segment_lengths);
    [inside, on] = inpolygon(curve(:, 1), curve(:, 2), track_x, track_y);
    bad_points = ~(inside | on);
    outside_points = outside_points + nnz(bad_points);
    outside_point_samples(trajectory_idx) = any(bad_points);

    [point_h, point_h_each] = physical_obstacle_h(curve', ...
        C.obstacle_physical_geometry);
    obstacle_points = obstacle_points + nnz(point_h < 0);
    obstacle_point_samples(trajectory_idx) = any(point_h < 0);
    obstacle_points_each = obstacle_points_each + sum(point_h_each < 0, 2)';

    for point_idx = 1:64
        lambda = linspace(0, 1, 101);
        dense = curve(point_idx, :)' .* (1 - lambda) + ...
            curve(point_idx + 1, :)' .* lambda;
        [dense_inside, dense_on] = inpolygon(dense(1, :), dense(2, :), ...
            track_x, track_y);
        if any(~(dense_inside | dense_on))
            outside_segments = outside_segments + 1;
            outside_segment_samples(trajectory_idx) = true;
        end
        [dense_h, dense_h_each] = physical_obstacle_h(dense, ...
            C.obstacle_physical_geometry);
        if any(dense_h < 0)
            obstacle_segments = obstacle_segments + 1;
            obstacle_segment_samples(trajectory_idx) = true;
        end
        obstacle_segments_each = obstacle_segments_each + ...
            any(dense_h_each < 0, 2)';
    end
end

fprintf('AUDIT trajectories=%d points_per_trajectory=65\n', n_trajectories);
fprintf('AUDIT obstacle_inside_points=%d affected_trajectories=%d\n', ...
    obstacle_points, nnz(obstacle_point_samples));
fprintf('AUDIT obstacle_crossing_segments=%d affected_trajectories=%d\n', ...
    obstacle_segments, nnz(obstacle_segment_samples));
fprintf('AUDIT obstacle_inside_points_each=%s\n', ...
    mat2str(obstacle_points_each));
fprintf('AUDIT obstacle_crossing_segments_each=%s\n', ...
    mat2str(obstacle_segments_each));
fprintf('AUDIT track_outside_points=%d affected_trajectories=%d\n', ...
    outside_points, nnz(outside_point_samples));
fprintf('AUDIT track_outside_segments=%d affected_trajectories=%d\n', ...
    outside_segments, nnz(outside_segment_samples));
[~, coordinate_order] = sort(max_abs_coordinate, 'descend');
[~, jump_order] = sort(max_segment_length, 'descend');
[~, length_order] = sort(curve_length, 'descend');
fprintf('AUDIT largest_abs_coordinate [traj value]=%s\n', ...
    mat2str([coordinate_order(1:10)', ...
    max_abs_coordinate(coordinate_order(1:10))'], 6));
fprintf('AUDIT largest_segment_jump [traj value]=%s\n', ...
    mat2str([jump_order(1:10)', ...
    max_segment_length(jump_order(1:10))'], 6));
fprintf('AUDIT largest_curve_length [traj value]=%s\n', ...
    mat2str([length_order(1:10)', curve_length(length_order(1:10))'], 6));

for trajectory_idx = unique([coordinate_order(1:3), jump_order(1:3)])
    curve = squeeze(points(:, trajectory_idx, :));
    [~, linear_idx] = max(abs(curve), [], 'all', 'linear');
    [point_idx, coordinate_idx] = ind2sub(size(curve), linear_idx);
    fprintf(['AUDIT divergent trajectory=%d point=%d coordinate=%d ', ...
        'xy=[%.9g %.9g]\n'], trajectory_idx, point_idx, coordinate_idx, ...
        curve(point_idx, 1), curve(point_idx, 2));
end

thresholds = [1.5, 2, 5, 10];
first_crossing = nan(size(thresholds));
first_samples = nan(size(thresholds));
first_points = nan(size(thresholds));
global_max = -inf;
global_max_info = nan(1, 4);
for time_idx = 1:size(S.third_traj_path_10d, 1)
    state_now = squeeze(S.third_traj_path_10d(time_idx, :, :));
    local_now = state_now .* T.std' + T.mean';
    global_now = local_increment_rows_to_global(local_now, T.feature_dim, 5);
    physical = reshape(global_now, size(global_now, 1), T.feature_dim, 5);
    xy = physical(:, 1:2, :);
    [time_max, time_linear_idx] = max(abs(xy), [], 'all', 'linear');
    if time_max > global_max
        [sample_idx, coordinate_idx, point_idx] = ...
            ind2sub(size(xy), time_linear_idx);
        global_max = time_max;
        global_max_info = [time_idx, sample_idx, point_idx, coordinate_idx];
    end
    for threshold_idx = 1:numel(thresholds)
        if isnan(first_crossing(threshold_idx)) && ...
                time_max > thresholds(threshold_idx)
            [sample_idx, ~, point_idx] = ...
                ind2sub(size(xy), time_linear_idx);
            first_crossing(threshold_idx) = S.third_rollout_times(time_idx);
            first_samples(threshold_idx) = sample_idx;
            first_points(threshold_idx) = point_idx;
        end
    end
end
for threshold_idx = 1:numel(thresholds)
    sample_idx = first_samples(threshold_idx);
    if isnan(sample_idx)
        fprintf('AUDIT threshold>%g never crossed\n', thresholds(threshold_idx));
    else
        fprintf(['AUDIT first threshold>%g t=%.6f sample=%d ', ...
            'trajectory=%d segment=%d point=%d\n'], thresholds(threshold_idx), ...
            first_crossing(threshold_idx), sample_idx, ...
            ceil(sample_idx / 16), mod(sample_idx - 1, 16) + 1, ...
            first_points(threshold_idx));
    end
end
fprintf(['AUDIT global rollout max=%g t=%.6f sample=%d trajectory=%d ', ...
    'segment=%d point=%d coordinate=%d\n'], global_max, ...
    S.third_rollout_times(global_max_info(1)), global_max_info(2), ...
    ceil(global_max_info(2) / 16), mod(global_max_info(2)-1, 16)+1, ...
    global_max_info(3), global_max_info(4));

probe_samples = [394, 486, 576];
probe_times = [0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.72, 0.79, 0.83, 0.9, 0.995];
for sample_idx = probe_samples
    fprintf('AUDIT sample_trace sample=%d trajectory=%d segment=%d\n', ...
        sample_idx, ceil(sample_idx / 16), mod(sample_idx - 1, 16) + 1);
    for requested_time = probe_times
        [~, time_idx] = min(abs(S.third_rollout_times - requested_time));
        state_now = squeeze(S.third_traj_path_10d(time_idx, sample_idx, :))';
        local_now = state_now .* T.std' + T.mean';
        global_now = local_increment_rows_to_global(local_now, T.feature_dim, 5);
        curve_now = reshape(global_now, T.feature_dim, [])';
        fprintf('  t=%.4f maxabs=%.6g points_xy=%s\n', ...
            S.third_rollout_times(time_idx), max(abs(curve_now(:, 1:2)), [], 'all'), ...
            mat2str(curve_now(:, 1:2), 4));
    end
end
end

function [h_min, h_each] = physical_obstacle_h(points, geometry)
n_points = size(points, 2);
h_min = inf(1, n_points);
h_each = inf(size(geometry.centers, 2), n_points);
for obstacle_idx = 1:size(geometry.centers, 2)
    angle = geometry.angles(obstacle_idx);
    rotation = [cos(angle), -sin(angle); sin(angle), cos(angle)];
    local = rotation' * ...
        (points - geometry.centers(:, obstacle_idx));
    scaled = local ./ geometry.semi_axes(:, obstacle_idx);
    exponent = geometry.exponents(obstacle_idx);
    power_sum = sum(abs(scaled) .^ exponent, 1);
    if exponent > 2
        h_now = power_sum .^ (1 / exponent) - 1;
    else
        h_now = power_sum - 1;
    end
    h_each(obstacle_idx, :) = h_now;
    h_min = min(h_min, h_now);
end
end
