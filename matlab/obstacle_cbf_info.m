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
obstacle_info.active_candidate_indices = zeros(0, 1);
obstacle_info.row_point_indices = zeros(0, 1);
% 可按层延迟启用避障约束。启用时刻之前不向 QP 添加 obstacle 行，
% 让名义 GP 和其它约束先完成初期整形。
activation_time = struct_field_default(constraint_cfg, ...
    'obstacle_activation_time', 0.0);
if ~obstacle_info.enabled || t < activation_time
    obstacle_info.enabled = false;
    return;
end
if ~isfield(constraint_cfg, 'obstacle_point_maps')
    error(['obstacle_cbf_info: obstacle_enabled 为真但缺 obstacle_point_maps，', ...
        '需在 main_demo 里用 build_obstacle_point_maps 构造并挂到约束上。']);
end
point_maps = constraint_cfg.obstacle_point_maps;
centers = constraint_cfg.obstacle_centers;      % 2 x n_obs
semi_axes = constraint_cfg.obstacle_semi_axes;  % 2 x n_obs
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
time_shift = struct_field_default(constraint_cfg, 'ptzf_time_shift', 0.0);
t_eff = t + time_shift;
tau = max(1.0 - t_eff, eps);
phi1 = omega / (tau ^ 2);        % blow-up (h<0 时使用)

x_now = stats.x(:);
mu = stats.mu(:);
n_obs = size(centers, 2);
rows_A = zeros(0, n_u);
rows_b = zeros(0, 1);
rows_h = zeros(0, 1);
row_point_indices = zeros(0, 1);
for pm_idx = 1:numel(point_maps)
    M = point_maps(pm_idx).M;
    o = point_maps(pm_idx).o;
    p = M * x_now + o;           % 物理位置 (2x1)
    for obs_idx = 1:n_obs
        c = centers(:, obs_idx);
        a = semi_axes(1, obs_idx);
        b = semi_axes(2, obs_idx);
        Q = diag([1 / a^2, 1 / b^2]);
        d = p - c;
        h = d' * Q * d - 1.0;
        % The QP and the physical rollout use the same true cumulative map.
        grad_h_true = 2.0 * (Q * d)' * M;    % 1 x N
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
        if norm(grad_h_ctrl) < grad_tol
            continue;
        end
        obstacle_info.candidate_row_active(candidate_idx, 1) = true;
        obstacle_info.active_candidate_indices(end + 1, 1) = candidate_idx;
        rows_A = [rows_A; -grad_h_ctrl];                    %#ok<AGROW>
        rows_b = [rows_b; grad_h_true * mu + phi * h];      %#ok<AGROW>
        rows_h = [rows_h; h];                               %#ok<AGROW>
        row_point_indices(end + 1, 1) = point_maps(pm_idx).point_index; %#ok<AGROW>
    end
end
obstacle_info.A = rows_A;
obstacle_info.bounds = rows_b;
obstacle_info.h_values = rows_h;
obstacle_info.row_point_indices = row_point_indices;
obstacle_info.residual_without_u = -rows_b;   % u=0 时的违反量(>0 表示需要修正)
obstacle_info.n_rows = size(rows_A, 1);
if ~isempty(obstacle_info.candidate_h_values)
    obstacle_info.h_min = min(obstacle_info.candidate_h_values);
end
if ~isempty(rows_h)
    obstacle_info.max_residual_without_u = max(-rows_b);
end
end
