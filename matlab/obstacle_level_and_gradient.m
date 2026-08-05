function [h, grad_p, escape_direction] = obstacle_level_and_gradient( ...
    p, obstacle, obstacle_idx)
%OBSTACLE_LEVEL_AND_GRADIENT Evaluate a rotated superellipse level set.
% h >= 0 is outside (safe), h = 0 is the boundary, and h < 0 is inside.

c = obstacle.centers(:, obstacle_idx);
semi_axes = obstacle.semi_axes(:, obstacle_idx);
angle = obstacle_value(obstacle, 'angles', obstacle_idx, 0.0);
exponent = obstacle_value(obstacle, 'exponents', obstacle_idx, 2.0);

if any(semi_axes <= 0) || ~isfinite(exponent) || exponent < 2
    error('obstacle_level_and_gradient:InvalidGeometry', ...
        'Obstacle semi-axes must be positive and exponent must be >= 2.');
end

rotation = [cos(angle), -sin(angle); sin(angle), cos(angle)];
q = rotation' * (p(:) - c);
scaled = q ./ semi_axes;
power_sum = sum(abs(scaled) .^ exponent);
if exponent > 2
    % The p-norm form has the same zero-level boundary as power_sum-1,
    % but grows linearly in the far field and avoids ill-conditioned QPs.
    radial_level = power_sum .^ (1 / exponent);
    h = radial_level - 1.0;
    if power_sum <= eps
        grad_q = zeros(2, 1);
    else
        grad_q = power_sum .^ (1 / exponent - 1) .* ...
            sign(q) .* abs(q) .^ (exponent - 1) ./ ...
            (semi_axes .^ exponent);
    end
else
    h = power_sum - 1.0;
    grad_q = exponent .* sign(q) .* abs(q) .^ (exponent - 1) ./ ...
        (semi_axes .^ exponent);
end
grad_p = rotation * grad_q;

[~, short_axis] = min(semi_axes);
escape_local = zeros(2, 1);
escape_local(short_axis) = 1.0;
if q(short_axis) < 0
    escape_local(short_axis) = -1.0;
end
escape_direction = rotation * escape_local;
end

function value = obstacle_value(obstacle, field_name, obstacle_idx, default_value)
if ~isfield(obstacle, field_name) || isempty(obstacle.(field_name))
    value = default_value;
    return;
end
values = obstacle.(field_name);
if isscalar(values)
    value = values;
else
    value = values(obstacle_idx);
end
end
