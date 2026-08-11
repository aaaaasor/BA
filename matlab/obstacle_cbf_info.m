function obstacle_info = obstacle_cbf_info(stats, constraint_cfg, t)
% SafeFlow 障碍规定时间 CBF（FMBF）。对每个作用点 m、每个椭圆障碍 j，
% 生成一条 QP 线性约束（项目约定 A*u <= bound）：
%   h_m   = (p_m - c)' Q (p_m - c) - 1 >= 0        Q = diag(1/a^2, 1/b^2)
%   grad  = d h_m / d x = 2 (Q (p_m - c))' M_m      (1 x N)
%   要求  d h_m/dt + phi(t,h_m)*h_m >= 0,  d h_m/dt = grad*(mu + u)
%   =>    A = -grad,  bound = grad*mu + phi(t,h_m)*h_m
% phi 规定时间:  h>=0 -> phi0;  h<0 -> phi1 = omega/(1-t_eff)^2 (blow-up)。
% t_eff = t + ptzf_time_shift（把 t_max<1 映射到 t_eff=1），与其它 PTZF 约束一致。
obstacle_info.enabled = struct_field_default(constraint_cfg, ...
    'obstacle_enabled', false);
n_u = numel(stats.mu);
obstacle_info.A = zeros(0, n_u);
obstacle_info.bounds = zeros(0, 1);
obstacle_info.h_values = zeros(0, 1);
obstacle_info.h_min = inf;
obstacle_info.residual_without_u = zeros(0, 1);
obstacle_info.max_residual_without_u = -inf;
obstacle_info.n_rows = 0;
% Candidate diagnostics include rows rejected by grad_tol.
obstacle_info.candidate_h_values = zeros(0, 1);
obstacle_info.candidate_point_indices = zeros(0, 1);
obstacle_info.candidate_obstacle_indices = zeros(0, 1);
obstacle_info.candidate_grad_true = zeros(0, n_u);
obstacle_info.candidate_grad_ctrl = zeros(0, n_u);
obstacle_info.candidate_grad_true_norm = zeros(0, 1);
obstacle_info.candidate_grad_ctrl_norm = zeros(0, 1);
obstacle_info.candidate_drift = zeros(0, 1);
obstacle_info.candidate_phi = zeros(0, 1);
obstacle_info.candidate_row_active = false(0, 1);
% 落在椭圆中心驻点上、改用逃逸方向代替真实梯度的行。
obstacle_info.candidate_row_escape = false(0, 1);
obstacle_info.active_candidate_indices = zeros(0, 1);
obstacle_info.row_point_indices = zeros(0, 1);
obstacle_info.row_is_escape = false(0, 1);
obstacle_info.n_escape_rows = 0;
% 可按层延迟启用避障约束。启用时刻之前不向 QP 添加 obstacle 行，
% 让名义 GP 和其它约束先完成初期整形。
activation_time = struct_field_default(constraint_cfg, ...
    'obstacle_activation_time', 0.0);
if ~obstacle_info.enabled
    obstacle_info.enabled = false;
    return;
end
if ~isfield(constraint_cfg, 'obstacle_point_maps')
    error(['obstacle_cbf_info: obstacle_enabled 为真但缺 obstacle_point_maps，', ...
        '需在 main_demo 里用 build_obstacle_point_maps 构造并挂到约束上。']);
end
point_maps = constraint_cfg.obstacle_point_maps;
if isfield(constraint_cfg, 'obstacle_geometry')
    obstacle_geometry = constraint_cfg.obstacle_geometry;
else
    obstacle_geometry = struct( ...
        'centers', constraint_cfg.obstacle_centers, ...
        'semi_axes', constraint_cfg.obstacle_semi_axes);
end
centers = obstacle_geometry.centers;
n_obs = size(centers, 2);
activation_times = struct_field_default(constraint_cfg, ...
    'obstacle_activation_times', activation_time);
if isscalar(activation_times)
    activation_times = repmat(activation_times, 1, n_obs);
else
    activation_times = reshape(activation_times, 1, []);
end
if numel(activation_times) ~= n_obs || ...
        any(~isfinite(activation_times)) || any(activation_times < 0)
    error(['obstacle_activation_times must be a finite nonnegative scalar ', ...
        'or contain one value per obstacle.']);
end
if t < min(activation_times)
    obstacle_info.enabled = false;
    return;
end
phi0 = struct_field_default(constraint_cfg, 'obstacle_phi0', 2.0);
omega = struct_field_default(constraint_cfg, 'obstacle_phi1_omega', 4.0);
phi1_early = struct_field_default(constraint_cfg, ...
    'obstacle_phi1_early_gain', omega);
phi1_switch_time = struct_field_default(constraint_cfg, ...
    'obstacle_phi1_switch_time', 0.0);
if ~isscalar(phi1_early) || ~isfinite(phi1_early) || phi1_early < 0
    error('obstacle_phi1_early_gain must be a finite nonnegative scalar.');
end
if ~isscalar(phi1_switch_time) || ~isfinite(phi1_switch_time) || ...
        phi1_switch_time < 0
    error('obstacle_phi1_switch_time must be a finite nonnegative scalar.');
end
grad_tol = struct_field_default(constraint_cfg, 'grad_tol', 1e-6);
% 落在椭圆中心驻点上时的逃逸速度(物理单位/时间)。留空则按椭圆短半轴自适应。
escape_speed = struct_field_default(constraint_cfg, ...
    'obstacle_escape_speed', []);
if ~isempty(escape_speed) && ...
        (~isscalar(escape_speed) || ~isfinite(escape_speed) || ...
        escape_speed <= 0)
    error('obstacle_escape_speed must be empty or a finite positive scalar.');
end
time_shift = struct_field_default(constraint_cfg, 'ptzf_time_shift', 0.0);
t_eff = t + time_shift;
tau = max(1.0 - t_eff, eps);
phi1 = omega / (tau ^ 2);        % blow-up (h<0 时使用)

x_now = stats.x(:);
mu = stats.mu(:);
rows_A = zeros(0, n_u);
rows_b = zeros(0, 1);
rows_h = zeros(0, 1);
row_point_indices = zeros(0, 1);
row_is_escape = false(0, 1);
for pm_idx = 1:numel(point_maps)
    M = point_maps(pm_idx).M;
    o = point_maps(pm_idx).o;
    p = M * x_now + o;           % 物理位置 (2x1)
    for obs_idx = 1:n_obs
        if t < activation_times(obs_idx)
            continue;
        end
        [h, grad_h_p, escape_direction] = ...
            obstacle_level_and_gradient(p, obstacle_geometry, obs_idx);
        % The QP and the physical rollout use the same true cumulative map.
        grad_h_true = grad_h_p' * M;    % 1 x N
        grad_h_ctrl = grad_h_true;
        if h >= 0
            phi = phi0;
        elseif t < phi1_switch_time
            phi = phi1_early;
        else
            phi = phi1;
        end
        candidate_idx = numel(obstacle_info.candidate_h_values) + 1;
        obstacle_info.candidate_h_values(candidate_idx, 1) = h;
        obstacle_info.candidate_point_indices(candidate_idx, 1) = pm_idx;
        obstacle_info.candidate_obstacle_indices(candidate_idx, 1) = obs_idx;
        obstacle_info.candidate_grad_true(candidate_idx, :) = grad_h_true;
        obstacle_info.candidate_grad_ctrl(candidate_idx, :) = grad_h_ctrl;
        obstacle_info.candidate_grad_true_norm(candidate_idx, 1) = ...
            norm(grad_h_true);
        obstacle_info.candidate_grad_ctrl_norm(candidate_idx, 1) = ...
            norm(grad_h_ctrl);
        obstacle_info.candidate_drift(candidate_idx, 1) = grad_h_true * mu;
        obstacle_info.candidate_phi(candidate_idx, 1) = phi;
        obstacle_info.candidate_row_active(candidate_idx, 1) = false;
        obstacle_info.candidate_row_escape(candidate_idx, 1) = false;
        if norm(grad_h_ctrl) < grad_tol
            % 梯度退化：作用点落在椭圆中心的驻点上。h 的一阶信息为 0，
            % 正常 CBF 行会退化成 0'*u <= 负数(一阶不可行)。原先直接
            % continue，等于恰恰在最危险的地方(h≈-1，已深入障碍内部)
            % 把这条避障约束整个扔掉。
            %
            % 导师方案：解不出来就随便给它个方向，先脱离驻点。驻点是
            % 孤立的(grad = 2*Q*(p-c) 是线性的，只有 p=c 一个零点)，
            % 只要离开中心梯度立刻线性恢复，正常 PTCBF 就接管了。
            if h >= 0
                % 安全侧的梯度退化(例如 M 让该点对 x 不敏感)不需要推开，
                % 保持原来的跳过行为。
                continue;
            end
            % "随便"这里取确定性的短轴方向，而不是 rand：
            %   1) RK4 的 k1..k4 必须看到同一个方向。随机方向会让四个
            %      stage 互相抵消，合成位移接近 0，反而逃不出去；
            %   2) 短轴(Q 最大特征值方向，即半轴较小的那个坐标轴)是离开
            %      椭圆的最短路径，逃逸代价最小；
            %   3) 可复现，便于复盘。
            % 符号由 d 在该轴上的投影决定(恰在中心时取正)，使方向在整个
            % 中心邻域内恒定，不随数值噪声翻转。
            e_phys = escape_direction;
            % 逃逸方向必须拉回控制空间：u 只能通过 M 影响这个点。
            e_ctrl = e_phys' * M;               % 1 x N
            if norm(e_ctrl) < grad_tol
                % M 把逃逸方向也湮灭了，u 根本推不动这个点。此时造行只会
                % 得到又一条 0'*u <= 负数，只能放弃。
                continue;
            end
            % 要求沿逃逸方向的物理速度至少 v_escape：
            %   e'*M*(mu + u) >= v_escape
            %   => -(e'*M)*u <= (e'*M)*mu - v_escape
            % 仍然是一条普通 QP 行，和其它约束一起协商，不直接覆盖 u。
            v_escape = escape_speed;
            if isempty(v_escape)
                % 缺省取短半轴：一个时间单位内走出椭圆的量级。
                v_escape = min(obstacle_geometry.semi_axes(:, obs_idx));
            end
            obstacle_info.candidate_row_active(candidate_idx, 1) = true;
            obstacle_info.candidate_row_escape(candidate_idx, 1) = true;
            obstacle_info.active_candidate_indices(end + 1, 1) = candidate_idx;
            rows_A = [rows_A; -e_ctrl];                     %#ok<AGROW>
            rows_b = [rows_b; e_ctrl * mu - v_escape];      %#ok<AGROW>
            rows_h = [rows_h; h];                           %#ok<AGROW>
            row_point_indices(end + 1, 1) = ...
                point_maps(pm_idx).point_index;             %#ok<AGROW>
            row_is_escape(end + 1, 1) = true;               %#ok<AGROW>
            continue;
        end
        obstacle_info.candidate_row_active(candidate_idx, 1) = true;
        obstacle_info.active_candidate_indices(end + 1, 1) = candidate_idx;
        rows_A = [rows_A; -grad_h_ctrl];                    %#ok<AGROW>
        rows_b = [rows_b; grad_h_true * mu + phi * h];      %#ok<AGROW>
        rows_h = [rows_h; h];                               %#ok<AGROW>
        row_is_escape(end + 1, 1) = false;                  %#ok<AGROW>
        row_point_indices(end + 1, 1) = point_maps(pm_idx).point_index; %#ok<AGROW>
    end
end
obstacle_info.A = rows_A;
obstacle_info.bounds = rows_b;
obstacle_info.h_values = rows_h;
obstacle_info.row_point_indices = row_point_indices;
obstacle_info.row_is_escape = row_is_escape;
obstacle_info.n_escape_rows = sum(row_is_escape);
obstacle_info.residual_without_u = -rows_b;   % u=0 时的违反量(>0 表示需要修正)
obstacle_info.n_rows = size(rows_A, 1);
if ~isempty(obstacle_info.candidate_h_values)
    obstacle_info.h_min = min(obstacle_info.candidate_h_values);
end
if ~isempty(rows_h)
    obstacle_info.max_residual_without_u = max(-rows_b);
end
end
