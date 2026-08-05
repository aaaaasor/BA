function dataset = build_track_dataset(opts)
% 未传参数时使用默认值。
if nargin < 1 || isempty(opts)
    opts = struct();
end

% 基本参数。
n_points = struct_field_default(opts, 'n_points', 65);
n_trajectories = struct_field_default(opts, 'n_trajectories', 200);
s_range_m = struct_field_default(opts, 's_range_m', [250, 1150]);
save_path = struct_field_default(opts, 'save_path', '');

this_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(this_dir, 'trajectory_data');
track_csv = fullfile(data_dir, 'Nuerburgring.csv');
raceline_csv = fullfile(data_dir, 'Nuerburgring_raceline.csv');

% 截取并归一化赛段。
segment = extract_track_segment(track_csv, raceline_csv, ...
    struct('s_range_m', s_range_m));

% 生成轨迹。
generator_opts = struct_field_default(opts, 'generator', struct());
[points, gates] = generate_track_training_points_2d(n_points, ...
    n_trajectories, segment, generator_opts);

dataset.points = points;
dataset.segment = segment;
dataset.gates = gates;
dataset.meta.n_points = n_points;
dataset.meta.n_trajectories = n_trajectories;
dataset.meta.feature_names = {'x', 'y', 'dx_ds', 'dy_ds'};
dataset.meta.flattened_dim = n_points * size(points, 3);
dataset.meta.created = datetime('now');
dataset.meta.boundary_handling = 'dense_curve_rejection_sampling';

% OCP 接口默认关闭。
ocp_opts = struct_field_default(opts, 'ocp', struct());
dataset.ocp = run_optional_track_ocp(points, segment, ocp_opts);

% 按需保存。
if ~isempty(save_path)
    % 自动创建目标目录；save_path 只有文件名时 fileparts 可能为空。
    if ~isfolder(fileparts(save_path))
        mkdir(fileparts(save_path));
    end

    save(save_path, 'dataset', '-v7.3');
    fprintf('saved %s (%d points x %d trajectories, dim %d)\n', save_path, ...
        n_points, n_trajectories, dataset.meta.flattened_dim);
end
end
