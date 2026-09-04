function [h_out, g_norm] = softmin_normalize_grid(h_in, x_grid, y_grid, grad_tol)
%SOFTMIN_NORMALIZE_GRID Convert level-set values to first-order distances.
% Mirrors joint_safety_softmin_info: each component is divided by |grad h|
% so the soft minimum compares metres rather than mixing signed distances
% (rails, |grad h| = 1) with the dimensionless superellipse value
% (obstacles, |grad h| ~ 41).
dx = x_grid(2) - x_grid(1);
dy = y_grid(2) - y_grid(1);
h_out = h_in;
g_norm = ones(size(h_in));
for c = 1:size(h_in, 3)
    [gx, gy] = gradient(h_in(:, :, c), dx, dy);
    gn = max(hypot(gx, gy), grad_tol);
    g_norm(:, :, c) = gn;
    h_out(:, :, c) = h_in(:, :, c) ./ gn;
end
end
