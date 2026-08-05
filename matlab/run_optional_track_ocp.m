% 可选 OCP 接口，默认关闭。
% solver 调用形式：result = solver(points, segment, solver_opts)。
% result 可包含 states、controls 和 info 字段。

function ocp = run_optional_track_ocp(points, segment, opts)
if nargin < 3 || isempty(opts)
    opts = struct();
end

enabled = struct_field_default(opts, 'enabled', false);

% 默认不运行 OCP。
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

% 补齐统一输出字段。
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
