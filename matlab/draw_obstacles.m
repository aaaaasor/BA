function draw_obstacles(obstacle, varargin)
% 在当前 axes 上画出所有椭圆障碍（灰色填充）。
% obstacle: 含 enabled / centers (2 x n_obs) / semi_axes (2 x n_obs)。
if isempty(obstacle) || ~isfield(obstacle, 'enabled') || ~obstacle.enabled
    return;
end
washeld = ishold;
hold on;
for j = 1:size(obstacle.centers, 2)
    xy = obstacle_outline_points(obstacle, j);
    fill(xy(:, 1), xy(:, 2), [0.82 0.82 0.82], ...
        'EdgeColor', [0.35 0.35 0.35], 'FaceAlpha', 0.6, 'LineWidth', 1.0, ...
        varargin{:});
end
if ~washeld
    hold off;
end
end
