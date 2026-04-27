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

dt = (t_max - t_min) / n_time_slices;
slice_times = t_min + (0:n_time_slices-1)' * dt;
x0_row = x0(:)';
x1_row = x1(:)';
velocity_row = (x1(:) - x0(:))';

x_slices = (1 - slice_times) .* x0_row + slice_times .* x1_row;
y_slices = repmat(velocity_row, numel(slice_times), 1);

x_plot = [repelem(slice_times, numel(x0)), reshape(x_slices.', [], 1)];
y_plot = reshape(y_slices.', [], 1);
end
