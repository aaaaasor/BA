% Build sliding-window flow-matching data in adjacent-increment coordinates.
function [s_slices, x_slices, y_slices, target_data, source_data, ...
    data_transform] = build_local_increment_sliding_window_training_data( ...
    dense_points, t_min, t_max, n_s_slices, n_points_per_window, ...
    window_stride)
%% Defaults
if nargin < 5 || isempty(n_points_per_window)
    n_points_per_window = 5;
end
if nargin < 6 || isempty(window_stride)
    window_stride = 1;
end

%% Dimensions
n_dense_points = size(dense_points, 1);
n_trajectories = size(dense_points, 2);
feature_dim = size(dense_points, 3);
if feature_dim < 2
    error('Local increment windows require at least x/y features.');
end
window_start_idx = 1:window_stride:(n_dense_points - n_points_per_window + 1);
n_windows = numel(window_start_idx);
n_samples = n_trajectories * n_windows;
n_rows = feature_dim * n_points_per_window;

s_slices = linspace(t_min, t_max, n_s_slices)';
target_data = zeros(n_rows, n_samples);
source_data = randn(n_rows, n_samples);

%% Increment Window Targets
sample_idx = 1;
for window_start = window_start_idx
    for traj_idx = 1:n_trajectories
        window_points = dense_points(window_start:(window_start + ...
            n_points_per_window - 1), traj_idx, :);
        window_curve = squeeze(window_points);
        if n_points_per_window == 1
            window_curve = reshape(window_curve, 1, []);
        end
        increment_curve = global_window_to_adjacent_increment(window_curve);
        target_data(:, sample_idx) = reshape(increment_curve', [], 1);
        sample_idx = sample_idx + 1;
    end
end

%% Standardize Local Target Distribution
target_mean = mean(target_data, 2);
target_std = std(target_data, 1, 2);
target_std = max(target_std, eps);

target_data_model = (target_data - target_mean) ./ target_std;
data_transform.mean = target_mean;
data_transform.std = target_std;
data_transform.feature_dim = feature_dim;
data_transform.is_local_increment = true;
data_transform.window_start_idx = window_start_idx;

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

function increment_curve = global_window_to_adjacent_increment(window_curve)
increment_curve = window_curve;
if size(window_curve, 1) > 1
    increment_curve(2:end, 1:2) = diff(window_curve(:, 1:2), 1, 1);
end
end
