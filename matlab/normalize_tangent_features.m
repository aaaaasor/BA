% 把每个点的 tangent 向量归一化成单位向量
function points = normalize_tangent_features(points)
tangent_norm = sqrt(points(:, :, 3) .^ 2 + points(:, :, 4) .^ 2);
tangent_norm = max(tangent_norm, eps);
points(:, :, 3) = points(:, :, 3) ./ tangent_norm;
points(:, :, 4) = points(:, :, 4) ./ tangent_norm;
end
