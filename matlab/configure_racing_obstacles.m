function obstacle = configure_racing_obstacles(track_segment, obstacle)
%CONFIGURE_RACING_OBSTACLES Place one square obstacle beside the track.
% The square is mostly outside the left track boundary. Only a shallow part
% intrudes into the corridor, and its size scales with the local track width.

if nargin < 2 || isempty(obstacle)
    obstacle = struct();
end

% Put the obstacle at a nested anchor location: s = 0.75 is represented in
% the first, second, and third levels, so control points are concentrated
% around the obstacle. The former ellipse and circle remain removed.
fraction = 0.75;
n_center = size(track_segment.center, 1);
index = max(2, min(n_center - 1, ...
    round(1 + fraction * (n_center - 1))));

centerline_point = track_segment.center(index, :)';
tangent = track_segment.tangent(index, :)';
tangent = tangent ./ max(norm(tangent), eps);
left_normal = [-tangent(2); tangent(1)];
left_half_width = max(track_segment.half_width_left(index), eps);
local_width = max(min(left_half_width, ...
    track_segment.half_width_right(index)), eps);

% Rotate the square by 45 degrees relative to the local track tangent and
% make it larger than the previous 0.85-width version.
square_half_size = 1.05 * local_width;
square_exponent = 12;
relative_angle = pi / 4;

% Rotation increases the square's projection onto the track normal. Use
% the superellipse support function. Setting the intrusion to one projected
% half-extent puts the local boundary through the square centre, so roughly
% 50% of this centre-symmetric obstacle lies inside the track.
dual_exponent = square_exponent / (square_exponent - 1);
normal_half_extent = square_half_size * ...
    (abs(sin(relative_angle)) ^ dual_exponent + ...
     abs(cos(relative_angle)) ^ dual_exponent) ^ (1 / dual_exponent);
intrusion_depth = normal_half_extent;
center_offset = left_half_width + normal_half_extent - intrusion_depth;
center = centerline_point + center_offset * left_normal;

% Preserve the caller's master switch. If no switch was supplied, enabling
% obstacles remains the standalone helper's default behaviour.
obstacle.enabled = struct_field_default(obstacle, 'enabled', true);
obstacle.centers = center;
obstacle.semi_axes = [square_half_size; square_half_size];
obstacle.angles = atan2(tangent(2), tangent(1)) + relative_angle;
obstacle.types = "square";
obstacle.exponents = square_exponent;
end
