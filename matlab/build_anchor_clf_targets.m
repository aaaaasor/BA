function [targets, indices] = build_anchor_clf_targets( ...
    anchor_points, n_anchor_samples, n_child_samples, n_segments, ...
    n_points_per_segment, data_transform)
% 为第二层 anchor PTCLF 构造目标点。
% anchor_points 来自第一层 rollout 的 5 个点；第二层每条曲线被拆成
% n_segments 段，所以每一段只约束该段起点和终点的全部特征
% （位置 x/y + 切向 dx/ds, dy/ds）。
feature_dim = size(anchor_points, 3);
% 第二层状态向量按每段的 5 个点顺序展开。
% PTCLF 约束第 1 个点和第 5 个点的全部特征（位置 x/y 以及切向
% dx/ds, dy/ds），这样相邻段拼接处不仅位置对齐，切线方向也连续。
position_dim = feature_dim;
first_point_idx = 1:position_dim;
last_point_idx = (n_points_per_segment - 1) * feature_dim + ...
    (1:position_dim);
indices = [first_point_idx, last_point_idx];

% 每个第一层 anchor sample 会生成 n_child_samples 条第二层 rollout。
% targets 的行顺序需要和第二层 rollout 的 sample/segment 顺序一致：
% sample 1 的 segment 1..4，然后 sample 2 的 segment 1..4，依此类推。
n_eval = n_anchor_samples * n_child_samples;
targets = zeros(n_eval * n_segments, numel(indices));
for anchor_sample_idx = 1:n_anchor_samples
    for child_sample_idx = 1:n_child_samples
        eval_idx = (anchor_sample_idx - 1) * n_child_samples + ...
            child_sample_idx;
        for segment_idx = 1:n_segments
            row_idx = (eval_idx - 1) * n_segments + segment_idx;

            % 当前第二层 segment 的目标端点是第一层相邻两个 anchor 点：
            % segment 1: anchor 1 -> anchor 2,
            % segment 2: anchor 2 -> anchor 3, 以此类推。
            endpoint_points = [
                squeeze(anchor_points(segment_idx, anchor_sample_idx, ...
                    1:position_dim))'; ...
                squeeze(anchor_points(segment_idx + 1, anchor_sample_idx, ...
                    1:position_dim))'];
            endpoint_vector = reshape(endpoint_points', [], 1);

            % rollout 状态在 GP/ODE 里使用标准化坐标，因此 CLF target
            % 也必须使用相同的 mean/std 转换，不能直接使用原始 x/y。
            mean_vector = data_transform.mean(indices);
            std_vector = data_transform.std(indices);
            targets(row_idx, :) = ((endpoint_vector - mean_vector(:)) ./ ...
                std_vector(:))';
        end
    end
end
end
