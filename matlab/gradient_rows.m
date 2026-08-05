%GRADIENT_ROWS 分别沿行索引估计二维曲线的 x、y 一阶导数。
function g = gradient_rows(xy)
g = [gradient(xy(:, 1)), gradient(xy(:, 2))];
end
