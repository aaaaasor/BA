function point_maps = build_obstacle_point_maps(data_transform, feature_dim, ...
    n_points_per_segment, apply_points, mode)
% Build linear maps from ODE state to physical obstacle-check points:
%     p_m = M_m * x + o_m.
% Only integer trajectory point indices are supported. This keeps the
% obstacle CBF acting on the selected generated points, not interpolated
% midpoint constraints.
position_dim = 2;
n_rows = feature_dim * n_points_per_segment;
std_vec = data_transform.std(:);
mean_vec = data_transform.mean(:);
if numel(std_vec) ~= n_rows
    error(['build_obstacle_point_maps: data_transform.std length %d does ', ...
        'not match feature_dim*n_points_per_segment=%d.'], ...
        numel(std_vec), n_rows);
end

point_maps = struct('point_index', {}, 'M', {}, 'o', {});
for k = 1:numel(apply_points)
    m = apply_points(k);
    if abs(m - round(m)) > eps
        error('build_obstacle_point_maps: point index %.3g must be an integer.', m);
    end
    if m < 1 || m > n_points_per_segment
        error('build_obstacle_point_maps: point index %.3g out of range [1,%d].', ...
            m, n_points_per_segment);
    end
    M = zeros(position_dim, n_rows);
    o = zeros(position_dim, 1);
    switch mode
        case 'absolute'
            cols = (m - 1) * feature_dim + (1:position_dim);
            M(:, cols) = diag(std_vec(cols));
            o = mean_vec(cols);
        case 'increment'
            for block_idx = 1:m
                cols = (block_idx - 1) * feature_dim + (1:position_dim);
                M(:, cols) = diag(std_vec(cols));
                o = o + mean_vec(cols);
            end
        otherwise
            error('build_obstacle_point_maps: unknown mode "%s".', mode);
    end
    point_maps(end + 1) = struct('point_index', m, 'M', M, 'o', o); %#ok<AGROW>
end
end
