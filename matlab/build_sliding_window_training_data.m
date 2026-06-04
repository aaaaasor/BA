% Build segment-level flow-matching data from sliding windows on dense paths.
function [s_slices, x_slices, y_slices, target_data, source_data, ...
    data_transform] = build_sliding_window_training_data( ...
    dense_points, t_min, t_max, n_s_slices, n_points_per_window)
%% Dimensions
if nargin < 5 || isempty(n_points_per_window)
    n_points_per_window = 5;
end
n_dense_points = size(dense_points, 1);
n_trajectories = size(dense_points, 2);
n_windows = n_dense_points - n_points_per_window + 1;
n_samples = n_trajectories * n_windows;
n_rows = 2 * n_points_per_window;

s_slices = linspace(t_min, t_max, n_s_slices)';
target_data = zeros(n_rows, n_samples);

%% Sliding Window Targets
sample_idx = 1;
for window_idx = 1:n_windows
    for traj_idx = 1:n_trajectories
        window_points = dense_points(window_idx:(window_idx + ...
            n_points_per_window - 1), traj_idx, :);
        window_curve = squeeze(window_points);
        target_data(:, sample_idx) = reshape(window_curve', [], 1);
        sample_idx = sample_idx + 1;
    end
end

%% Standardize Window Target Distribution
target_mean = mean(target_data, 'all');
target_std = std(target_data, 0, 'all');
target_std = max(target_std, eps);

target_data_model = (target_data - target_mean) ./ target_std;
source_data = randn(size(target_data));
data_transform.mean = target_mean;
data_transform.std = target_std;

target_vectors = target_data_model';
source_vectors = source_data';
velocity_vectors = target_vectors - source_vectors;

%% Flow-Matching Pairs
x_slices = zeros(numel(s_slices), n_samples, n_rows);
y_slices = zeros(numel(s_slices), n_samples, n_rows);
for slice_idx = 1:numel(s_slices)
    s = s_slices(slice_idx);
    x_slices(slice_idx, :, :) = (1.0 - s) * source_vectors + ...
        s * target_vectors;
    y_slices(slice_idx, :, :) = velocity_vectors;
end
end
