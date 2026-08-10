function geometry = build_track_boundary_geometry(track_segment, n_control_points)
%BUILD_TRACK_BOUNDARY_GEOMETRY Build left/right parametric boundary splines.
% Each boundary uses n_control_points knots, hence n_control_points-1 spline
% intervals. The center spline is retained only to select the inward side.

if nargin < 2 || isempty(n_control_points)
    n_control_points = 400;
end
if ~isscalar(n_control_points) || n_control_points < 4 || ...
        n_control_points ~= round(n_control_points)
    error('n_control_points must be an integer >= 4.');
end
required = {'left', 'right', 'center'};
if ~all(isfield(track_segment, required))
    error('track_segment must contain left, right, and center curves.');
end
n_track = size(track_segment.center, 1);
if n_track < n_control_points
    error('Track has %d points but %d boundary controls were requested.', ...
        n_track, n_control_points);
end

control_indices = round(linspace(1, n_track, n_control_points));
s_control = linspace(0.0, 1.0, n_control_points);
center_control = track_segment.center(control_indices, :);
geometry.n_control_points = n_control_points;
geometry.n_intervals = n_control_points - 1;
geometry.control_indices = control_indices;
geometry.s_control = s_control;
geometry.center_pp_x = pchip(s_control, center_control(:, 1)');
geometry.center_pp_y = pchip(s_control, center_control(:, 2)');

names = {'left', 'right'};
curves = repmat(struct(), 2, 1);
for curve_idx = 1:2
    points = track_segment.(names{curve_idx})(control_indices, :);
    curves(curve_idx).name = names{curve_idx};
    curves(curve_idx).control_points = points;
    curves(curve_idx).pp_x = pchip(s_control, points(:, 1)');
    curves(curve_idx).pp_y = pchip(s_control, points(:, 2)');
    curves(curve_idx).dpp_x = pp_derivative(curves(curve_idx).pp_x);
    curves(curve_idx).dpp_y = pp_derivative(curves(curve_idx).pp_y);
end
geometry.curves = curves;
end
