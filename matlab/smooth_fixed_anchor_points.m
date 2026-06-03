% Smooth generated 2D trajectories while keeping anchor points fixed.
function smoothed_points = smooth_fixed_anchor_points(points, anchor_idx, ...
    curvature_weight)
if nargin < 3 || isempty(curvature_weight)
    curvature_weight = 10.0;
end

n_points = size(points, 1);
n_trajectories = size(points, 2);
smoothed_points = points;
anchor_idx = unique(anchor_idx(:)');
free_idx = setdiff(1:n_points, anchor_idx);
if n_points < 3 || isempty(free_idx)
    return;
end

tangent_anchor_idx = anchor_idx(anchor_idx > 1 & anchor_idx < n_points);
curvature_center_idx = setdiff(2:(n_points - 1), tangent_anchor_idx);
second_diff = zeros(numel(curvature_center_idx), n_points);
for row_idx = 1:numel(curvature_center_idx)
    point_idx = curvature_center_idx(row_idx);
    second_diff(row_idx, (point_idx - 1):(point_idx + 1)) = ...
        [1.0, -2.0, 1.0];
end

tangent_system = zeros(numel(tangent_anchor_idx), numel(free_idx));
tangent_rhs_base = zeros(numel(tangent_anchor_idx), numel(anchor_idx));
for row_idx = 1:numel(tangent_anchor_idx)
    anchor = tangent_anchor_idx(row_idx);
    left_idx = anchor - 1;
    right_idx = anchor + 1;
    for idx = [left_idx, right_idx]
        free_pos = find(free_idx == idx, 1);
        if ~isempty(free_pos)
            tangent_system(row_idx, free_pos) = ...
                tangent_system(row_idx, free_pos) + 1.0;
        else
            anchor_pos = find(anchor_idx == idx, 1);
            tangent_rhs_base(row_idx, anchor_pos) = ...
                tangent_rhs_base(row_idx, anchor_pos) - 1.0;
        end
    end
    anchor_pos = find(anchor_idx == anchor, 1);
    tangent_rhs_base(row_idx, anchor_pos) = ...
        tangent_rhs_base(row_idx, anchor_pos) + 2.0;
end

diff_free = second_diff(:, free_idx);
diff_anchor = second_diff(:, anchor_idx);
normal_matrix = eye(numel(free_idx)) + ...
    curvature_weight * (diff_free' * diff_free);
kkt_matrix = [normal_matrix, tangent_system'; ...
    tangent_system, zeros(size(tangent_system, 1))];

for traj_idx = 1:n_trajectories
    curve = squeeze(points(:, traj_idx, :));
    if size(curve, 2) ~= 2
        curve = reshape(curve, n_points, 2);
    end
    anchor_values = curve(anchor_idx, :);
    for coord_idx = 1:2
        x_anchor = anchor_values(:, coord_idx);
        x_free_original = curve(free_idx, coord_idx);
        rhs = x_free_original - ...
            curvature_weight * diff_free' * diff_anchor * x_anchor;
        tangent_rhs = tangent_rhs_base * x_anchor;
        solution = kkt_matrix \ [rhs; tangent_rhs];
        curve(free_idx, coord_idx) = solution(1:numel(free_idx));
    end
    curve(anchor_idx, :) = anchor_values;
    smoothed_points(:, traj_idx, :) = curve;
end
end
