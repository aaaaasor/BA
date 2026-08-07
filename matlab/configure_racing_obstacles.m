function obstacle = configure_racing_obstacles(track_segment, obstacle)
%CONFIGURE_RACING_OBSTACLES Place a square and an ellipse along the track.
% Both obstacle sizes scale with the local track width.

if nargin < 2 || isempty(obstacle)
    obstacle = struct();
end

n_center = size(track_segment.center, 1);
centers = zeros(2, 0);
semi_axes = zeros(2, 0);
angles = zeros(1, 0);
types = strings(1, 0);
exponents = zeros(1, 0);

%% Square
square_cfg = struct_field_default(obstacle, 'square', struct());
if struct_field_default(square_cfg, 'enabled', true)
    square_fraction = struct_field_default(square_cfg, ...
        'track_fraction', 0.75);
    square_index = max(2, min(n_center - 1, ...
        round(1 + square_fraction * (n_center - 1))));

    centerline_point = track_segment.center(square_index, :)';
    tangent = track_segment.tangent(square_index, :)';
    tangent = tangent ./ max(norm(tangent), eps);
    left_normal = [-tangent(2); tangent(1)];
    left_half_width = max(track_segment.half_width_left(square_index), eps);
    local_width = max(min(left_half_width, ...
        track_segment.half_width_right(square_index)), eps);

    square_half_size = struct_field_default(square_cfg, ...
        'half_size_ratio', 1.05) * local_width;
    square_exponent = struct_field_default(square_cfg, 'exponent', 12);
    relative_angle = struct_field_default(square_cfg, ...
        'relative_angle', pi / 4);

    % Account for the rotated superellipse's projection onto the track
    % normal when placing the requested fraction inside the corridor.
    dual_exponent = square_exponent / (square_exponent - 1);
    normal_half_extent = square_half_size * ...
        (abs(sin(relative_angle)) ^ dual_exponent + ...
         abs(cos(relative_angle)) ^ dual_exponent) ^ (1 / dual_exponent);
    inside_ratio = min(max(struct_field_default(square_cfg, ...
        'track_inside_ratio', 0.50), 0), 1);
    intrusion_depth = 2 * inside_ratio * normal_half_extent;
    center_offset = left_half_width + normal_half_extent - intrusion_depth;
    square_center = centerline_point + center_offset * left_normal;
    square_angle = atan2(tangent(2), tangent(1)) + relative_angle;

    centers(:, end + 1) = square_center;
    semi_axes(:, end + 1) = [square_half_size; square_half_size];
    angles(end + 1) = square_angle;
    types(end + 1) = "square";
    exponents(end + 1) = square_exponent;
end

%% Ellipse
ellipse_cfg = struct_field_default(obstacle, 'ellipse', struct());
if struct_field_default(ellipse_cfg, 'enabled', true)
    ellipse_fraction = struct_field_default(ellipse_cfg, ...
        'track_fraction', 0.52);
    ellipse_index = max(2, min(n_center - 1, ...
        round(1 + ellipse_fraction * (n_center - 1))));
    centerline_point = track_segment.center(ellipse_index, :)';
    tangent = track_segment.tangent(ellipse_index, :)';
    tangent = tangent ./ max(norm(tangent), eps);
    ellipse_local_width = max(min( ...
        track_segment.half_width_left(ellipse_index), ...
        track_segment.half_width_right(ellipse_index)), eps);
    semi_major = struct_field_default(ellipse_cfg, ...
        'semi_major_ratio', 3.6) * ellipse_local_width;
    semi_minor = struct_field_default(ellipse_cfg, ...
        'semi_minor_ratio', 2.0) * ellipse_local_width;
    relative_angle = struct_field_default(ellipse_cfg, ...
        'relative_angle', 0.0);
    ellipse_angle = atan2(tangent(2), tangent(1)) + relative_angle;
    ellipse_center = centerline_point;
    ellipse_center(1) = struct_field_default(ellipse_cfg, ...
        'center_x', ellipse_center(1));
    ellipse_center(2) = struct_field_default(ellipse_cfg, ...
        'center_y', ellipse_center(2));

    centers(:, end + 1) = ellipse_center;
    semi_axes(:, end + 1) = [semi_major; semi_minor];
    angles(end + 1) = ellipse_angle;
    types(end + 1) = "ellipse";
    exponents(end + 1) = 2;
end

% Preserve the caller's master switch. If no switch was supplied, enabling
% obstacles remains the standalone helper's default behaviour.
obstacle.enabled = struct_field_default(obstacle, 'enabled', true);
obstacle.centers = centers;
obstacle.semi_axes = semi_axes;
obstacle.angles = angles;
obstacle.types = types;
obstacle.exponents = exponents;
end
