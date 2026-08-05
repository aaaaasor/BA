%TRACK_LATERAL_OFFSET 计算二维点相对赛道中心线的带符号横向偏移。
%
%   [lateral, nearest_idx] = TRACK_LATERAL_OFFSET(segment, xy)
%
% 对每个查询点取最近的中心线点，再投影到该处的左法向：
%   lateral > 0 在中心线左侧，lateral < 0 在右侧。
% nearest_idx 是对应的中心线索引，调用方据此取该点真正的左右半宽——每个点
% 的对应关系都不同，不能按点序号均匀映射。
%
% 走廊判定（生成时的拒绝采样）和占用率可视化都用这一个函数，避免两处各写
% 一套对应关系而出现"图上没贴边、实际被拒绝"的不一致。
function [lateral, nearest_idx] = track_lateral_offset(segment, xy)
center = segment.center;
normal_left = [-segment.tangent(:, 2), segment.tangent(:, 1)];

n_query = size(xy, 1);
lateral = zeros(n_query, 1);
nearest_idx = zeros(n_query, 1);

% 分块计算，避免 n_query * n_center 的距离矩阵吃满内存。
block_size = 512;
for block_start = 1:block_size:n_query
    block_stop = min(block_start + block_size - 1, n_query);
    block = xy(block_start:block_stop, :);
    d2 = sum((permute(block, [1, 3, 2]) - permute(center, [3, 1, 2])) .^ 2, 3);
    [~, idx] = min(d2, [], 2);
    nearest_idx(block_start:block_stop) = idx;
    lateral(block_start:block_stop) = ...
        sum((block - center(idx, :)) .* normal_left(idx, :), 2);
end
end
