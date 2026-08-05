function [s_star, h_star] = closest_obstacle_point_on_segment( ...
    pa, pb, obstacle, obstacle_idx)
%CLOSEST_OBSTACLE_POINT_ON_SEGMENT Minimize a convex obstacle level on a line.
% The zero-level sets used here are convex superellipses. Their power-sum
% derivative along a line is monotone, so a short bisection finds the global
% minimizer without invoking the comparatively expensive fminbnd routine.

c = obstacle.centers(:, obstacle_idx);
a = obstacle.semi_axes(:, obstacle_idx);
angle = geometry_value(obstacle, 'angles', obstacle_idx, 0.0);
exponent = geometry_value(obstacle, 'exponents', obstacle_idx, 2.0);
rotation = [cos(angle), -sin(angle); sin(angle), cos(angle)];
q0 = rotation' * (pa(:) - c);
dq = rotation' * (pb(:) - pa(:));

derivative = @(s) sum(exponent .* sign(q0 + s .* dq) .* ...
    abs(q0 + s .* dq) .^ (exponent - 1) .* dq ./ (a .^ exponent));
g0 = derivative(0.0);
g1 = derivative(1.0);
if g0 >= 0
    s_star = 0.0;
elseif g1 <= 0
    s_star = 1.0;
else
    lo = 0.0;
    hi = 1.0;
    for iteration = 1:24
        mid = 0.5 * (lo + hi);
        if derivative(mid) < 0
            lo = mid;
        else
            hi = mid;
        end
    end
    s_star = 0.5 * (lo + hi);
end

h_star = obstacle_level_and_gradient( ...
    pa(:) + s_star .* (pb(:) - pa(:)), obstacle, obstacle_idx);
end

function value = geometry_value(geometry, field_name, obstacle_idx, default_value)
if ~isfield(geometry, field_name) || isempty(geometry.(field_name))
    value = default_value;
elseif isscalar(geometry.(field_name))
    value = geometry.(field_name);
else
    value = geometry.(field_name)(obstacle_idx);
end
end
