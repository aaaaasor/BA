% Build segment-level flow-matching data from coarse 2D trajectory points.
function [s_slices, x_slices, y_slices, target_points, source_points, ...
    target_data, source_data, data_transform] = build_segment_training_data( ...
    coarse_points, t_min, t_max, n_s_slices, n_points_per_segment)
%% Dimensions
if nargin < 5 || isempty(n_points_per_segment)
    n_points_per_segment = 5;
end
n_coarse_points = size(coarse_points, 1);
n_trajectories = size(coarse_points, 2);
n_segments = n_coarse_points - 1;
n_samples = n_trajectories * n_segments;
n_rows = 2 * n_points_per_segment;

s_slices = linspace(t_min, t_max, n_s_slices)';
target_data = zeros(n_rows, n_samples);
target_points = zeros(n_points_per_segment, n_samples, 2);

%% Segment Targets
sample_idx = 1;
for traj_idx = 1:n_trajectories
    for segment_idx = 1:n_segments
        start_point = squeeze(coarse_points(segment_idx, traj_idx, :))';
        end_point = squeeze(coarse_points(segment_idx + 1, traj_idx, :))';
        segment_curve = sample_segment_by_arclength(start_point, ...
            end_point, n_points_per_segment);
        target_points(:, sample_idx, :) = segment_curve;
        target_data(:, sample_idx) = reshape(segment_curve', [], 1);
        sample_idx = sample_idx + 1;
    end
end

%% Standardize Segment Target Distribution
target_mean = mean(target_data, 'all');
target_std = std(target_data, 0, 'all');
target_std = max(target_std, eps);

target_data_model = (target_data - target_mean) ./ target_std;
source_data = randn(size(target_data));
data_transform.mean = target_mean;
data_transform.std = target_std;

source_points = zeros(n_points_per_segment, n_samples, 2);
for sample_idx = 1:n_samples
    source_points(:, sample_idx, :) = ...
        reshape(source_data(:, sample_idx), 2, [])';
end

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

function segment_curve = sample_segment_by_arclength(start_point, ...
    end_point, n_points_per_segment)
t_dense_set = linspace(0, 1, 5000);
x_dense_set = (1.0 - t_dense_set) * start_point(1) + ...
    t_dense_set * end_point(1);
y_dense_set = (1.0 - t_dense_set) * start_point(2) + ...
    t_dense_set * end_point(2);
ds_dense_set = sqrt((x_dense_set(2:end) - x_dense_set(1:(end - 1))).^2 + ...
    (y_dense_set(2:end) - y_dense_set(1:(end - 1))).^2);
s_dense_set = [0, cumsum(ds_dense_set)];
segment_length = s_dense_set(end);

s_set = linspace(0, segment_length, n_points_per_segment);
if segment_length <= eps
    t_train_i = linspace(0, 1, n_points_per_segment);
else
    t_train_i = interp1(s_dense_set, t_dense_set, s_set);
end

x_train_i = (1.0 - t_train_i) * start_point(1) + ...
    t_train_i * end_point(1);
y_train_i = (1.0 - t_train_i) * start_point(2) + ...
    t_train_i * end_point(2);
segment_curve = [x_train_i(:), y_train_i(:)];
end
