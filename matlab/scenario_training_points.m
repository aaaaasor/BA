%SCENARIO_TRAINING_POINTS 按 cfg.scenario 取该层的训练轨迹点。
%
%   points = SCENARIO_TRAINING_POINTS(cfg, n_points, n_trajectories)
%   [points, segment] = SCENARIO_TRAINING_POINTS(...)
%
% cfg.scenario:
%   'obstacle' - 单位方块里的合成轨迹（原三层 demo），每次按需现场生成；
%                segment 返回空。
%   'racing'   - 赛道段数据集，从 cfg.track_dataset_path 读一次并缓存，
%                按 stride 抽点得到该层的网格；segment 返回赛段几何。
%
% 三层必须来自同一批轨迹：第一层 5 点、第二层 17 点、第三层 65 点在锚点上
% 必须逐位相同，否则 main_demo 的 anchor_reconstruction_error 检查会报错。
%   - obstacle: 生成器内部固定 rng，等弧长采样天然嵌套；
%   - racing:   直接从同一个 65 点数组按 stride 抽，子集自然一致。
function [points, segment] = scenario_training_points(cfg, n_points, ...
    n_trajectories)
scenario = struct_field_default(cfg, 'scenario', 'obstacle');
segment = [];

switch scenario
    case 'obstacle'
        points = generate_original_training_points_2d(n_points, n_trajectories);

    case 'racing'
        dataset = load_track_dataset(cfg);
        segment = dataset.segment;
        n_total_points = size(dataset.points, 1);
        n_available = size(dataset.points, 2);

        if n_trajectories > n_available
            error('scenario_training_points:notEnoughTrajectories', ...
                ['Track dataset has %d trajectories but %d were requested. ', ...
                'Regenerate with build_track_dataset(struct(''n_trajectories'', %d)).'], ...
                n_available, n_trajectories, n_trajectories);
        end

        stride = (n_total_points - 1) / (n_points - 1);
        if mod(stride, 1) ~= 0
            error('scenario_training_points:strideMismatch', ...
                ['Cannot take %d points from a %d-point dataset: ', ...
                '(%d-1) is not divisible by (%d-1). Regenerate the dataset ', ...
                'with a compatible n_points.'], ...
                n_points, n_total_points, n_total_points, n_points);
        end

        points = dataset.points(1:stride:n_total_points, 1:n_trajectories, :);

    otherwise
        error('scenario_training_points:unknownScenario', ...
            'cfg.scenario must be ''obstacle'' or ''racing'', got ''%s''.', ...
            scenario);
end
end

%% ------------------------------------------------------------------------
function dataset = load_track_dataset(cfg)
%LOAD_TRACK_DATASET 读取赛道数据集并按路径缓存，避免每层重复读盘。
persistent cached_path cached_dataset
dataset_path = struct_field_default(cfg, 'track_dataset_path', ...
    fullfile('trajectory_data', 'track_dataset_arena.mat'));
if ~isfile(dataset_path)
    this_dir = fileparts(mfilename('fullpath'));
    dataset_path = fullfile(this_dir, dataset_path);
end
if ~isfile(dataset_path)
    error('scenario_training_points:missingDataset', ...
        ['Track dataset not found: %s. Generate it with ', ...
        'build_track_dataset(struct(''save_path'', ...)).'], dataset_path);
end
if isempty(cached_path) || ~strcmp(cached_path, dataset_path)
    loaded = load(dataset_path);
    cached_dataset = loaded.dataset;
    cached_path = dataset_path;
end
dataset = cached_dataset;
end
