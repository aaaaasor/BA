function [h, grad] = evaluate_track_implicit_field(fields, boundary_idx, p)
%EVALUATE_TRACK_IMPLICIT_FIELD Evaluate one fixed global track field.
% h is the bilinear interpolant of the stored field and grad is that
% interpolant's own analytic derivative, so grad is exactly the gradient of
% the h the CBF row uses.
%
% The previous version interpolated a separately stored finite-difference
% gradient grid.  That grid is not the derivative of the bilinear h, so the
% QP predicted hdot with one function and then observed h change according to
% another: measured 3.95 deg median (22.9 deg max) direction disagreement and
% a 0.974 median delivery ratio.  Where the nominal flow nearly cancels the
% CBF correction that few-percent inconsistency consumed more than half of
% the net progress, so value and gradient must come from one function.

x_max = fields.x_min + fields.dx * (fields.n_x - 1);
y_max = fields.y_min + fields.dy * (fields.n_y - 1);
p = p(:);
p_grid = [min(max(p(1), fields.x_min), x_max); ...
    min(max(p(2), fields.y_min), y_max)];
x_coordinate = (p_grid(1) - fields.x_min) / fields.dx + 1;
y_coordinate = (p_grid(2) - fields.y_min) / fields.dy + 1;
x_idx = min(max(floor(x_coordinate), 1), fields.n_x - 1);
y_idx = min(max(floor(y_coordinate), 1), fields.n_y - 1);
x_fraction = x_coordinate - x_idx;
y_fraction = y_coordinate - y_idx;

grid_values = fields.h{boundary_idx};
h_00 = grid_values(y_idx, x_idx);
h_10 = grid_values(y_idx, x_idx + 1);
h_01 = grid_values(y_idx + 1, x_idx);
h_11 = grid_values(y_idx + 1, x_idx + 1);

h = (1 - x_fraction) * (1 - y_fraction) * h_00 + ...
    x_fraction * (1 - y_fraction) * h_10 + ...
    (1 - x_fraction) * y_fraction * h_01 + ...
    x_fraction * y_fraction * h_11;
grad = [((1 - y_fraction) * (h_10 - h_00) + ...
    y_fraction * (h_11 - h_01)) / fields.dx, ...
    ((1 - x_fraction) * (h_01 - h_00) + ...
    x_fraction * (h_11 - h_10)) / fields.dy];
% Off-grid queries keep the boundary cell's linear extension.
h = h + grad * (p - p_grid);
end
