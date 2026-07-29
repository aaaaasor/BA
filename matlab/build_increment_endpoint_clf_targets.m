function [targets, clf_matrix, clf_offset, endpoint_owner_std] = ...
    build_increment_endpoint_clf_targets(parent_segment_data, ...
    n_anchor_samples, n_child_samples, n_parent_segments, ...
    n_points_per_segment, data_transform)
% 为第三层 anchor PTCLF 构造目标。parent_segment_data 是第二层 rollout
% 的逐段结果(物理坐标, 每行一段 5 个点)，不做任何跨段拼接：第二层每段
% 自己的 5 个点细分出 4 个第三层 segment，第 j 个的目标端点 = 父段曲线
% 的第 j / j+1 点。相邻父段边界处两侧各自的端点估计互不混合（首尾接不
% 上时如实体现在目标里）。切向特征归一化为单位向量后作为目标。
%
% 第三层状态是标准化的局部增量表示 z，物理增量 v = std.*z + mean，
% v = [S0, S1-S0, ..., S4-S3]（只有位置 x/y 取增量，切向保持原值）。
% 末点绝对位置 = S0 + 所有位置增量之和，是 z 的线性组合而非下标子集，
% 因此返回线性映射: 物理首尾端点 = clf_matrix * z + clf_offset，
% CLF 误差 e = clf_matrix * z + clf_offset - target(物理坐标)。
feature_dim = size(parent_segment_data, 2) / n_points_per_segment;
n_rows = feature_dim * n_points_per_segment;
position_dim = 2;
tangent_rows = (position_dim + 1):feature_dim;

std_vec = data_transform.std(:);
mean_vec = data_transform.mean(:);

% A: 物理增量表示 v -> 物理首尾端点 [首点(4); 末点(4)]。
A = zeros(2 * feature_dim, n_rows);
% 首点 = v 的第一个 block(绝对量)。
A(1:feature_dim, 1:feature_dim) = eye(feature_dim);
% 末点位置 = S0(pos) + 所有增量 block 的位置分量之和。
for point_idx = 1:n_points_per_segment
    col0 = (point_idx - 1) * feature_dim;
    A(feature_dim + (1:position_dim), col0 + (1:position_dim)) = ...
        eye(position_dim);
end
% 末点切向 = 最后一个 block 的切向(保持绝对值，不做增量)。
last_col0 = (n_points_per_segment - 1) * feature_dim;
A(feature_dim + tangent_rows, last_col0 + tangent_rows) = ...
    eye(feature_dim - position_dim);

% v = diag(std)*z + mean, 端点 = A*v = (A*diag(std))*z + A*mean。
clf_matrix = A .* std_vec';
clf_offset = A * mean_vec;

% The split sequential QP assigns P1 to the first increment block and P5
% to the last increment block.  Standardize each endpoint residual with
% the std of its owner block.  For P5 this does not assign targets to the
% earlier increments; it only changes the metric of the cumulative
% endpoint residual.  In particular, the owner-block part of the endpoint
% Jacobian is unit-scaled in all four position/tangent channels.
first_cols = 1:feature_dim;
last_cols = (n_points_per_segment - 1) * feature_dim + (1:feature_dim);
endpoint_owner_std = [std_vec(first_cols); std_vec(last_cols)];

% targets 行顺序与第三层 rollout 的 sample/segment 顺序一致：
% 第三层 eval 内按父段 1..n_parent_segments、父段内子段 1..4 展开。
n_children_per_parent = n_points_per_segment - 1;
n_third_segments = n_parent_segments * n_children_per_parent;
n_eval = n_anchor_samples * n_child_samples;
targets = zeros(n_eval * n_third_segments, 2 * feature_dim);
for anchor_sample_idx = 1:n_anchor_samples
    for child_sample_idx = 1:n_child_samples
        eval_idx = (anchor_sample_idx - 1) * n_child_samples + ...
            child_sample_idx;
        for parent_idx = 1:n_parent_segments
            parent_row = (anchor_sample_idx - 1) * n_parent_segments + ...
                parent_idx;
            parent_curve = reshape( ...
                parent_segment_data(parent_row, :), feature_dim, [])';
            % 切向归一化为单位向量（与训练目标的切向尺度一致）。
            tangent_norm = max(sqrt(sum( ...
                parent_curve(:, tangent_rows) .^ 2, 2)), eps);
            parent_curve(:, tangent_rows) = ...
                parent_curve(:, tangent_rows) ./ tangent_norm;
            for sub_idx = 1:n_children_per_parent
                segment_idx = (parent_idx - 1) * n_children_per_parent + ...
                    sub_idx;
                row_idx = (eval_idx - 1) * n_third_segments + segment_idx;
                endpoint_points = [parent_curve(sub_idx, :); ...
                    parent_curve(sub_idx + 1, :)];
                targets(row_idx, :) = reshape(endpoint_points', 1, []);
            end
        end
    end
end
end
