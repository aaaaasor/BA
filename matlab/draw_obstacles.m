function draw_obstacles(obstacle, varargin)
% 在当前 axes 上画出所有椭圆障碍（灰色填充）。
% obstacle: 含 enabled / centers (2 x n_obs) / semi_axes (2 x n_obs)。
if isempty(obstacle) || ~isfield(obstacle, 'enabled') || ~obstacle.enabled
    return;
end
theta = linspace(0, 2 * pi, 200);
washeld = ishold;
hold on;
for j = 1:size(obstacle.centers, 2)
    c = obstacle.centers(:, j);
    a = obstacle.semi_axes(1, j);
    b = obstacle.semi_axes(2, j);
    fill(c(1) + a * cos(theta), c(2) + b * sin(theta), [0.82 0.82 0.82], ...
        'EdgeColor', [0.35 0.35 0.35], 'FaceAlpha', 0.6, 'LineWidth', 1.0, ...
        varargin{:});
end
if ~washeld
    hold off;
end
end
