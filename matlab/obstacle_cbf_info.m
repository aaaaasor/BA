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
if ~obstacle_info.enabled
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
        grad_h = 2.0 * (Q * d)' * M;         % 1 x N
        % 中心退化: grad h = 2 Q (p-c) 在椭圆中心为 0，一阶控制失效。
        % 论文 Proposition 1 直接假设 dh/ds != 0 处处成立，跳过此点不做
        % 特殊处理（与论文一致，不额外补逃逸机制）。
        if norm(grad_h) < grad_tol
            continue;
        end
        if h >= 0
            phi = phi0;
        else
            phi = phi1;
        end
        rows_A = [rows_A; -grad_h];                    %#ok<AGROW>
        rows_b = [rows_b; grad_h * mu + phi * h];      %#ok<AGROW>
        rows_h = [rows_h; h];                          %#ok<AGROW>
    end
end
obstacle_info.A = rows_A;
obstacle_info.bounds = rows_b;
obstacle_info.h_values = rows_h;
obstacle_info.residual_without_u = -rows_b;   % u=0 时的违反量(>0 表示需要修正)
obstacle_info.n_rows = size(rows_A, 1);
if ~isempty(rows_h)
    obstacle_info.h_min = min(rows_h);
    obstacle_info.max_residual_without_u = max(-rows_b);
end
end
