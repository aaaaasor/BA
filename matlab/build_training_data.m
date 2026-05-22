% Build 10D trajectory-space flow-matching training data.
% Each MAT-file column is one interleaved trajectory:
% [x1; y1; x2; y2; ...].
function [s_slices, x_slices, y_slices, target_points, source_points, ...
    target_data, source_data, trajectory_t_slices, data_transform] = build_training_data( ...
    mat_path, t_min, t_max, max_trajectories, n_s_slices)
%% Default Arguments
if nargin < 2
    t_min = 0.0;
end
if nargin < 3
    t_max = 1.0;
end
if nargin < 5 || isempty(n_s_slices)
    n_s_slices = 25;
end

%% Load Target Trajectories
loaded_data = load(mat_path, 'data_train');
if ~isfield(loaded_data, 'data_train')
    error('The MAT file does not contain the variable data_train.');
end

target_data = double(loaded_data.data_train);
if nargin >= 4 && ~isempty(max_trajectories)
    max_trajectories = min(max_trajectories, size(target_data, 2));
    target_data = target_data(:, 1:max_trajectories);
end

%% Dimension Checks
n_rows = size(target_data, 1);
if mod(n_rows, 2) ~= 0
    error('Trajectory data must have an even number of rows.');
end

n_samples = size(target_data, 2);
n_points_per_traj = n_rows / 2;
trajectory_t_slices = linspace(t_min, t_max, n_points_per_traj)';
s_slices = linspace(t_min, t_max, n_s_slices)';

target_points = zeros(n_points_per_traj, n_samples, 2);
source_points = zeros(n_points_per_traj, n_samples, 2);

%% Standardize Target Distribution
target_mean = mean(target_data, 'all');
target_std = std(target_data, 0, 'all');
target_std = max(target_std, eps);

target_data_model = (target_data - target_mean) ./ target_std;
source_data = randn(size(target_data));
data_transform.mean = target_mean;
data_transform.std = target_std;

%% Target And Source Trajectories
for sample_idx = 1:n_samples
    target_matrix = reshape(target_data(:, sample_idx), 2, [])';
    source_matrix = reshape(source_data(:, sample_idx), 2, [])';

    target_points(:, sample_idx, :) = target_matrix;
    source_points(:, sample_idx, :) = source_matrix;
end

target_vectors = target_data_model';
source_vectors = source_data';
velocity_vectors = target_vectors - source_vectors;

%% Flow-Matching Pairs
x_slices = zeros(numel(s_slices), n_samples, n_rows);
y_slices = zeros(numel(s_slices), n_samples, n_rows);
for slice_idx = 1:numel(s_slices)
    s = s_slices(slice_idx);
    x_slices(slice_idx, :, :) = (1.0 - s) * source_vectors + s * target_vectors;
    y_slices(slice_idx, :, :) = velocity_vectors;
end
end
