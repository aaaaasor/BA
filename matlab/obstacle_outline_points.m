function xy = obstacle_outline_points(obstacle, obstacle_idx, n_points)
%OBSTACLE_OUTLINE_POINTS Return drawing points for a rotated superellipse.
if nargin < 3
    n_points = 240;
end

theta = linspace(0, 2 * pi, n_points);
exponent = 2.0;
if isfield(obstacle, 'exponents') && ~isempty(obstacle.exponents)
    exponent = obstacle.exponents(min(obstacle_idx, ...
        numel(obstacle.exponents)));
end
angle = 0.0;
if isfield(obstacle, 'angles') && ~isempty(obstacle.angles)
    angle = obstacle.angles(min(obstacle_idx, numel(obstacle.angles)));
end

semi_axes = obstacle.semi_axes(:, obstacle_idx);
local_xy = [ ...
    semi_axes(1) .* sign(cos(theta)) .* abs(cos(theta)) .^ (2 / exponent); ...
    semi_axes(2) .* sign(sin(theta)) .* abs(sin(theta)) .^ (2 / exponent)];
rotation = [cos(angle), -sin(angle); sin(angle), cos(angle)];
xy = (rotation * local_xy + obstacle.centers(:, obstacle_idx))';
end
