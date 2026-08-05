%NEAREST_POINT_INDEX 返回二维点集中与 query 欧氏距离最小的行索引。
function idx = nearest_point_index(points, query)
[~, idx] = min(sum((points - query) .^ 2, 2));
end
