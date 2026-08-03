%RUN_OPTIONAL_TRACK_OCP 为赛车数据集保留一个默认关闭的 OCP 回调接口。
%
%   ocp = RUN_OPTIONAL_TRACK_OCP(points, segment)
%   ocp = RUN_OPTIONAL_TRACK_OCP(points, segment, opts)
%
% 当前二维数据集不要求动力学一致性，因此默认不解 OCP。该函数只提供一个
% 很薄的接口层，方便其他使用同一数据集的人以后接入自己的车辆模型和求解器，
% 同时不改动几何路径 dataset.points。
%
% 输入：
%   points  - n_points × n_trajectories × feature_dim 的几何路径数据；
%             当前前两维为 [x,y]，第 3、4 维为单位切向量。
%   segment - extract_track_segment 返回的赛段几何及坐标变换。
%   opts    - 可选结构体：
%       enabled     - 是否调用 OCP，默认 false。
%       solver      - enabled=true 时必需的函数句柄。
%       solver_opts - 原样转交给 solver 的自定义配置结构体。
%
% solver 的统一调用形式为：
%
%   result = solver(points, segment, solver_opts)
%
% solver 必须返回结构体。推荐至少使用以下字段，具体数组尺寸由车辆模型决定：
%
%   result.states   - 状态轨迹，例如 [x,y,theta,v]；
%   result.controls - 控制轨迹，例如 [steering,acceleration]；
%   result.info     - 求解状态、代价、失败样本索引等附加信息。
%
% 默认返回：
%   ocp.enabled  = false
%   ocp.status   = 'not_requested'
%   ocp.states   = []
%   ocp.controls = []
%   ocp.info     = struct()
%
% 启用示例：
%
%   opts.ocp.enabled = true;
%   opts.ocp.solver = @my_track_ocp_solver;
%   opts.ocp.solver_opts = struct('dt', 0.1, 'wheelbase', 2.7);
%   dataset = build_track_dataset(opts);
%
% 说明：赛道边界在几何数据生成阶段已经由密集曲线拒绝采样保证。OCP 回调是否
% 再次施加边界、动力学或控制限制，由具体 solver 自己决定。

function ocp = run_optional_track_ocp(points, segment, opts)
if nargin < 3 || isempty(opts)
    opts = struct();
end

enabled = struct_field_default(opts, 'enabled', false);

% 建立固定的空结果格式，使调用方不需要先判断 dataset 是否存在 ocp 字段。
ocp.enabled = logical(enabled);
ocp.status = 'not_requested';
ocp.states = [];
ocp.controls = [];
ocp.info = struct();

if ~enabled
    return;
end

solver = struct_field_default(opts, 'solver', []);
if ~isa(solver, 'function_handle')
    error('run_optional_track_ocp:missingSolver', ...
        ['opts.enabled=true 时必须通过 opts.solver 提供函数句柄，调用形式为 ' ...
         'result = solver(points, segment, solver_opts)。']);
end

solver_opts = struct_field_default(opts, 'solver_opts', struct());
result = solver(points, segment, solver_opts);
if ~isstruct(result)
    error('run_optional_track_ocp:invalidResult', ...
        'OCP solver 必须返回结构体，实际返回类型为 %s。', class(result));
end

% 保留 solver 自定义字段，同时补齐统一状态字段，便于下游读取。
ocp = result;
ocp.enabled = true;
if ~isfield(ocp, 'status') || isempty(ocp.status)
    ocp.status = 'completed';
end
if ~isfield(ocp, 'states')
    ocp.states = [];
end
if ~isfield(ocp, 'controls')
    ocp.controls = [];
end
if ~isfield(ocp, 'info')
    ocp.info = struct();
end
end
