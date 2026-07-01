function global_rows = local_increment_rows_to_global(local_rows, ...
    feature_dim, n_points_per_segment)
n_samples = size(local_rows, 1);
global_rows = zeros(size(local_rows));
for sample_idx = 1:n_samples
    local_curve = reshape(local_rows(sample_idx, :), feature_dim, [])';
    global_curve = local_curve;
	%把相邻增量还原成绝对坐标
    for point_idx = 2:n_points_per_segment
        global_curve(point_idx, 1:2) = ...
            global_curve(point_idx - 1, 1:2) + ...
            local_curve(point_idx, 1:2);
    end
    global_rows(sample_idx, :) = reshape(global_curve', 1, []);
end
end
