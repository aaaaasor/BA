%BUILD_TRACK_DATASET 构造指定赛道段上的二维 Flow Matching 训练数据集。
%
%   dataset = BUILD_TRACK_DATASET()
%   dataset = BUILD_TRACK_DATASET(opts)
%
% 本函数是赛车数据集生成流程的总入口，依次完成：
%   1. 从 trajectory_data 目录读取赛道中心线、左右宽度和赛车线；
%   2. 按米制弧长截取指定赛段，并将其等比例映射到 [0,1]^2；
%   3. 沿赛车线自动布置随机控制点 gate；
%   4. 通过分段三次 Hermite 插值和走廊拒绝采样生成平滑轨迹；
%   5. 将轨迹、赛段几何、gate 参数和元数据打包为 dataset 结构体。
%
% 输入 opts 为可选结构体，支持字段：
%   n_points       - 每条轨迹的等弧长采样点数，默认 65。
%   n_trajectories - 需要生成的轨迹数量，默认 30，与 cfg.n_train 对齐。
%                    下游是 LoG-GP（不是神经网络 FM），训练对数量为
%                    n_train * n_time_slices，30 条对应 450 对，正好是
%                    cfg.gp.first_level_n_pretrain。论文那种 5000 条是给
%                    神经网络用的规模，GP 跑不动，不要当默认值。
%   s_range_m      - 原始中心线上的截取弧长 [起点, 终点]，单位 m，
%                    默认 [250,1150]。
%   save_path      - 非空时将 dataset 保存到该 MAT 文件；默认只返回不保存。
%   generator      - 传给 generate_track_training_points_2d 的参数结构体。
%   ocp            - 可选 OCP 接口配置。默认 enabled=false，完全不求解 OCP；
%                    同事需要时可提供 solver 回调，接口说明见
%                    run_optional_track_ocp.m。
%
% 输出 dataset 的主要字段：
%   points  - n_points × n_trajectories × 4 数组，第三维依次为
%             [x, y, dx/ds, dy/ds]。x、y 是归一化坐标，后两项是单位切向量。
%   segment - 截取后的中心线、左右边界、赛车线和坐标变换。
%   gates   - 控制点圆盘的中心、半径、参考航向及航向扰动范围。
%   ocp     - 可选 OCP 回调的结果；默认 status='not_requested' 且数据为空。
%   meta    - 数据维数、特征名称和创建时间等说明信息。
%
% 注意：本数据集描述的是二维几何路径，不含速度、转向角、加速度或车辆动力学。

function dataset = build_track_dataset(opts)
% 未传 opts 时使用全部默认参数，便于直接调用 build_track_dataset()。
if nargin < 1 || isempty(opts)
    opts = struct();
end

% 读取顶层数据规模、赛段范围和保存位置。
n_points = struct_field_default(opts, 'n_points', 65);
n_trajectories = struct_field_default(opts, 'n_trajectories', 30);
s_range_m = struct_field_default(opts, 's_range_m', [250, 1150]);
save_path = struct_field_default(opts, 'save_path', '');

this_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(this_dir, 'trajectory_data');
track_csv = fullfile(data_dir, 'Nuerburgring.csv');
raceline_csv = fullfile(data_dir, 'Nuerburgring_raceline.csv');

% 将原始米制赛道截取并重采样，得到后续生成器使用的归一化赛段。
segment = extract_track_segment(track_csv, raceline_csv, ...
    struct('s_range_m', s_range_m));

% generator 子结构体只负责轨迹分布参数，使顶层接口与生成细节解耦。
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

% 默认路径只执行上面的几何生成与边界拒绝采样，不解 OCP。这里保留一个批量
% 回调接口：同事可在不改变 dataset.points 的前提下附加 states/controls 等结果。
ocp_opts = struct_field_default(opts, 'ocp', struct());
dataset.ocp = run_optional_track_ocp(points, segment, ocp_opts);

% 只有显式给出 save_path 时才写文件，避免预览或调参时意外覆盖数据。
if ~isempty(save_path)
    % 自动创建目标目录；save_path 只有文件名时 fileparts 可能为空。
    if ~isfolder(fileparts(save_path))
        mkdir(fileparts(save_path));
    end
    % v7.3 支持尺寸较大的训练集，但文件采用 HDF5 容器。
    save(save_path, 'dataset', '-v7.3');
    fprintf('saved %s (%d points x %d trajectories, dim %d)\n', save_path, ...
        n_points, n_trajectories, dataset.meta.flattened_dim);
end
end
