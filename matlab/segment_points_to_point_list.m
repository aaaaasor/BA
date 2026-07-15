% 把逐段 rollout 结果直接展开成点列表，不做任何拼接/去重：
% 相邻两段各自的边界点估计都保留（前段末点和后段首点是两个独立的点，
% 接不上时如实显示），输出 n_segments * n_points_per_segment 个点。
function trajectory_points = segment_points_to_point_list(segment_data, ...
    n_trajectories, n_segments, n_points_per_segment)
feature_dim = size(segment_data, 2) / n_points_per_segment;
n_full_points = n_segments * n_points_per_segment;
trajectory_points = zeros(n_full_points, n_trajectories, feature_dim);
for traj_idx = 1:n_trajectories
    for segment_idx = 1:n_segments
        sample_idx = (traj_idx - 1) * n_segments + segment_idx;
        segment_curve = reshape(segment_data(sample_idx, :), feature_dim, [])';
        row0 = (segment_idx - 1) * n_points_per_segment;
        trajectory_points(row0 + (1:n_points_per_segment), traj_idx, :) = ...
            reshape(segment_curve, n_points_per_segment, 1, feature_dim);
    end
end
end
