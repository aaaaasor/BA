% Build time-slice training pairs from pre-generated 2D trajectories.
% The MAT file stores one trajectory per column in the interleaved format
% [x1; y1; x2; y2; ...]. Each time slice provides 2D states x_t and
% finite-difference velocities v_t for LocalGP training.
function [x_slices, y_slices, trajectory_points, x_plot, y_plot, trajectory_data, t_slices] = build_training_data(mat_path, t_min, t_max, max_trajectories)
if nargin < 2
    t_min = 0.0;
end
if nargin < 3
    t_max = 1.0;
end

loaded_data = load(mat_path, 'data_train');
if ~isfield(loaded_data, 'data_train')
    error('The MAT file does not contain the variable data_train.');
end

trajectory_data = double(loaded_data.data_train);
if nargin >= 4 && ~isempty(max_trajectories)
    max_trajectories = min(max_trajectories, size(trajectory_data, 2));
    trajectory_data = trajectory_data(:, 1:max_trajectories);
end

n_rows = size(trajectory_data, 1);
if mod(n_rows, 2) ~= 0
    error('Trajectory data must have an even number of rows.');
end

n_samples = size(trajectory_data, 2);
n_time_slices = n_rows / 2;
t_slices = linspace(t_min, t_max, n_time_slices)';

%% Allocate one 2D point tensor per trajectory for plotting
trajectory_points = zeros(n_time_slices, n_samples, 2);
y_slices = zeros(n_time_slices, n_samples, 2);

if n_time_slices > 1
    dt = t_slices(2) - t_slices(1);
else
    dt = 1.0;
end

%% Unpack each column trajectory and approximate its velocity by finite differences
for sample_idx = 1:n_samples
    trajectory_matrix = reshape(trajectory_data(:, sample_idx), 2, []);
    positions = trajectory_matrix';
    velocities = zeros(n_time_slices, 2);

    if n_time_slices > 1
        velocities(1, :) = (positions(2, :) - positions(1, :)) / dt;
        velocities(end, :) = (positions(end, :) - positions(end - 1, :)) / dt;
    end
    for time_idx = 2:(n_time_slices - 1)
        velocities(time_idx, :) = (positions(time_idx + 1, :) - positions(time_idx - 1, :)) / (2.0 * dt);
    end

    trajectory_points(:, sample_idx, :) = positions;
    y_slices(:, sample_idx, :) = velocities;
end

x_slices = trajectory_points;
x_plot = [repelem(t_slices, n_samples), reshape(permute(trajectory_points, [2, 1, 3]), [], 2)];
y_plot = reshape(permute(y_slices, [2, 1, 3]), [], 2);
end
