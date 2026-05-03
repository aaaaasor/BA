function [slice_times, x_slices, y_slices, x_plot, y_plot] = build_training_data(x0, x1, n_time_slices, t_min, t_max)
if nargin < 4
    t_min = 0.0;
end
if nargin < 5
    t_max = 1.0;
end

if n_time_slices <= 0
    error('n_time_slices must be positive.');
end

if size(x0, 2) ~= 2 || any(size(x0) ~= size(x1))
    error('x0 and x1 must both have shape (n_samples, 2).');
end

dt = (t_max - t_min) / n_time_slices;
slice_times = t_min + (0:n_time_slices-1)' * dt;
n_samples = size(x0, 1);

x_slices = zeros(n_time_slices, n_samples, 2);
y_slices = zeros(n_time_slices, n_samples, 2);
velocity = x1 - x0;

for i = 1:n_time_slices
    t = slice_times(i);
    x_slices(i, :, :) = (1 - t) * x0 + t * x1;
    y_slices(i, :, :) = velocity;
end

x_plot = [repelem(slice_times, n_samples), reshape(permute(x_slices, [2, 1, 3]), [], 2)];
y_plot = reshape(permute(y_slices, [2, 1, 3]), [], 2);
end
