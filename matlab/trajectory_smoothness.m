function [cs, as_val] = trajectory_smoothness(xy)
% 曲率平滑度 CS (论文式 35) 与加速度平滑度 AS (论文式 37)。
%   xy: n x 2 轨迹点。
%   cs   = mean_{k=2..n-1} (1 - cos theta_k),  theta_k 为相邻段夹角
%   as_val = mean_{k} ||s_{k+1} - 2 s_k + s_{k-1}||
% 值越小越平滑。CS 反映方向突变, AS 反映速度突变。
n = size(xy, 1);
cs = 0;
as_val = 0;
if n < 3
    return;
end
w = diff(xy, 1, 1);                 % (n-1) x 2 段向量
cos_terms = zeros(n - 2, 1);
for k = 1:(n - 2)
    w1 = w(k, :);
    w2 = w(k + 1, :);
    denom = norm(w1) * norm(w2);
    if denom > eps
        cos_terms(k) = 1 - dot(w1, w2) / denom;
    end
end
cs = mean(cos_terms);
acc = xy(3:n, :) - 2 * xy(2:(n - 1), :) + xy(1:(n - 2), :);
as_val = mean(sqrt(sum(acc .^ 2, 2)));
end
