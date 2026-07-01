% Build trajectory-space flow-matching training data from generated targets.
% Each sample is provided as target_points_input(point, sample, feature).
function [s_slices, x_slices, y_slices, target_points, source_points, ...
    target_data, source_data, trajectory_t_slices, data_transform] = build_training_data( ...
    t_min, t_max, n_s_slices, target_points_input)
%% Load Target Trajectories
feature_dim = size(target_points_input, 3); %特征维度
target_data = reshape(permute(target_points_input, [3, 1, 2]), ...
    [], size(target_points_input, 2));
n_rows = size(target_data, 1); %每条轨迹总维度
n_samples = size(target_data, 2); %轨迹数量
n_points_per_traj = n_rows / feature_dim;
trajectory_t_slices = linspace(t_min, t_max, n_points_per_traj)';%轨迹内部点的位置参数
s_slices = linspace(t_min, t_max, n_s_slices)'; %生成 Flow Matching 的时间切片

target_points = zeros(n_points_per_traj, n_samples, feature_dim);
source_points = zeros(n_points_per_traj, n_samples, feature_dim);

%% Standardize Target Distribution
target_mean = mean(target_data, 2);
target_std = std(target_data, 0, 2);
target_std = max(target_std, eps);

target_data_model = (target_data - target_mean) ./ target_std;
source_data = randn(size(target_data));
data_transform.mean = target_mean;
data_transform.std = target_std;
data_transform.feature_dim = feature_dim;

%% Target And Source Trajectories
for sample_idx = 1:n_samples
    target_matrix = reshape(target_data(:, sample_idx), feature_dim, [])';
    source_matrix = reshape(source_data(:, sample_idx), feature_dim, [])';

    target_points(:, sample_idx, :) = target_matrix; %保存原始空间target
    source_points(:, sample_idx, :) = source_matrix; %source轨迹的点形式
end

target_vectors = target_data_model'; %保存标准化空间target
source_vectors = source_data'; %source轨迹的向量形式
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
