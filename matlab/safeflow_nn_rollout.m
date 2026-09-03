function out = safeflow_nn_rollout(net, mode, n_gen, opts)
%SAFEFLOW_NN_ROLLOUT SafeFlow 的 CFMBF 引导与终端安全滤波（论文式 26-32）。
%
%   out = safeflow_nn_rollout(net, 'fm',       100)
%   out = safeflow_nn_rollout(net, 'safeflow', 100)
%
% 与论文的对应：
%   式 26-27  CFMBF:  grad_h'*(v+u) + phi(t,h)*h >= 0
%   式 42     phi0 = 1;  phi1 = 1+4t^3 (t<0.9);  1/(1-t) (t>=0.9)
%   t_bar     CFMBF 仅在 t >= 0.5 后激活
%   式 29-30  逐点解耦 QP:  min |u_k|^2 + sum_j delta_j^2,  u_k in R^2
%             —— 每个轨迹点一个二维 QP，包含该点的全部 N 个约束
%   Alg.1 6-9 反标准化 -> 物理空间求 u -> 标准化后加回 NN 速度
%   式 32     终端滤波: min |T - T^{1-}|  s.t. h_j >= 0，逐点二维投影
%   状态      每点 [x,y,dx/ds,dy/ds]，总维度 65*4=260；安全修正只作用于 x,y。
%
% 与论文的已知偏差（需在文中注明）：
%   1) 积分器为固定步 RK4（100 步，至 t=0.996），而非原文的自适应 dopri5。
%   2) 式 32 的终端投影用序贯线性化求解，得到的是局部极小（原问题非凸）。
%      论文未指定求解器，故这不是与某个"论文算法"的偏离，而是全局最优性
%      无证据。缓解：不可行时 fmincon 多起点兜底；terminal_audit_n>0 时
%      抽样与 fmincon 多起点参考解比较投影位移，量化局部/全局差距。
%   3) 边界约束含 margin（引导与投影用 h-m，Safety 指标用 h>=0）；原文
%      平面导航实验的 h 是障碍物精确水平集，不含 margin。

if nargin < 3 || isempty(n_gen), n_gen = 100; end
if nargin < 4, opts = struct(); end
gf = @(f,d) struct_field_default(opts, f, d);

t_bar   = gf('activation_time', 0.5);
eps_sw  = gf('phi1_switch_time', 0.9);
phi0    = gf('phi0', 1.0);
n_rk    = gf('n_steps', 100);
t_max   = gf('t_max', 0.996);
term_on = gf('terminal_filter', strcmp(mode,'safeflow'));
term_it = gf('terminal_max_iter', 200);
term_tol= gf('terminal_tol', 1e-10);
seed    = gf('seed', 11);
% Algorithm 1 第 8/10 行的字面读法：u_t 每个 RK4 步只解一次（用该步的
% t_k 与 T_t），然后作为固定向量加进流场，四级只重算 v^theta。
% true 则每级重解 QP（数值更自洽，但 QP 次数与耗时约 4 倍，属偏离论文）。
u_stage = gf('u_per_stage', false);
% 终端投影审计：对最多这么多个被移动的点，另用 fmincon 多起点重解式 32，
% 记录我们的序贯线性化解与该参考解的距离差，用来量化"局部 vs 全局"的差距。
% 0 = 关闭（默认）。失败点的 fmincon 兜底与本开关无关，始终启用。
audit_n = gf('terminal_audit_n', 0);

cfg = get_config();
rng(cfg.random_seed);
[~, segment] = scenario_training_points(cfg, 65, cfg.n_train);
obst = configure_racing_obstacles(segment, cfg.obstacle);
geom = build_track_boundary_geometry(segment, ...
    struct_field_default(cfg.track_boundary, 'n_spline_points', 400));
margin = struct_field_default(cfg.track_boundary, 'margin', 0.003);
nObs = size(obst.centers, 2);

sd = net.sd_d; mu = net.mu_d;
D = numel(mu);
nP = struct_field_default(net, 'n_points', 65);
nF = struct_field_default(net, 'n_features_per_point', D/nP);
if nF ~= 4 || D ~= nP*nF
    error('SafeFlow NN state must contain 4 features per point and have size n_points*4.');
end
use_safety = strcmp(mode, 'safeflow');

% 每条轨迹一个独立种子，与主流程 build_second_level_initial_states_from_seeds
% 同一约定（RandStream('mt19937ar','Seed',s)），因此任意一条都能单独重放。
% 种子表由 base seed 派生，也可通过 opts.trajectory_seeds 直接指定。
traj_seeds = gf('trajectory_seeds', []);
if isempty(traj_seeds)
    sstream = RandStream('mt19937ar', 'Seed', seed);
    traj_seeds = randi(sstream, [1, 2^31-1], n_gen, 1);
end
traj_seeds = traj_seeds(:);
assert(numel(traj_seeds) == n_gen, '种子数 %d 与轨迹数 %d 不符。', ...
    numel(traj_seeds), n_gen);

t_sample = tic;
z = zeros(D, n_gen);
for ig = 1:n_gen
    z(:, ig) = randn(RandStream('mt19937ar', 'Seed', traj_seeds(ig)), D, 1);
end
sample_s = toc(t_sample);
z0 = z;                       % 初始噪声 T_hat_0，存下来供复现

% u 轨迹记录：每个 (RK4 步, 样本, 轨迹点) 的二维物理空间修正量。
% 只在 CFMBF 激活后（t >= t_bar）有非零值；不记录被冻结复用的那三级，
% 因为它们与步首完全相同。
% u_per_stage=true 时每步有四组不同的 u，不落盘（该配置只用于频率对照）。
trace_on = gf('record_u_trace', true) && ~u_stage;
u_trace = []; u_trace_t = []; u_trace_slack = [];
if trace_on && use_safety
    n_act = nnz(((0:n_rk-1)*(t_max/n_rk)) >= t_bar);
    u_trace       = zeros(n_act, n_gen, nP, 2, 'single');
    u_trace_t     = zeros(n_act, 1);
    u_trace_slack = false(n_act, n_gen, nP);
end
tr_k = 0;              % 已记录的激活步数
last_slack_col = false(nP,1);              % guidance 每次调用写回
rows_xy = reshape((0:nP-1)*nF + (1:2)', [], 1);   % 260 维里的 x,y 行

dt = t_max / n_rk;
n_qp = 0; slack_hits = 0;
fmincon_opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
    'SpecifyObjectiveGradient', true, 'SpecifyConstraintGradient', true, ...
    'OptimalityTolerance', 1e-12, 'ConstraintTolerance', 1e-12, ...
    'StepTolerance', 1e-14, 'MaxIterations', 500);

t_roll = tic;
for k = 1:n_rk
    tk = (k-1)*dt;
    if u_stage
        k1 = vel(z,            tk);
        k2 = vel(z + dt/2*k1,  tk + dt/2);
        k3 = vel(z + dt/2*k2,  tk + dt/2);
        k4 = vel(z + dt*k3,    tk + dt);
    else
        % Alg.1 line 8: u_t 在步首解一次；line 10: 四级共用同一个 u_t
        v0 = raw(z, tk);
        U  = zeros(D, n_gen);
        if use_safety && tk >= t_bar
            SL = false(nP, n_gen);
            for s = 1:n_gen
                U(:,s) = guidance(z(:,s), v0(:,s), tk);
                SL(:,s) = last_slack_col;
            end
            if trace_on
                tr_k = tr_k + 1;
                u_trace_t(tr_k) = tk;
                % 反标准化回物理空间，与 QP 求解所在的空间一致
                Up = U .* sd;
                u_trace(tr_k,:,:,:) = single(permute( ...
                    reshape(Up(rows_xy, :), 2, nP, n_gen), [3 2 1]));
                u_trace_slack(tr_k,:,:) = SL.';
            end
        end
        k1 = v0 + U;
        k2 = raw(z + dt/2*k1, tk + dt/2) + U;
        k3 = raw(z + dt/2*k2, tk + dt/2) + U;
        k4 = raw(z + dt*k3,   tk + dt)   + U;
    end
    z  = z + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end
roll_s = toc(t_roll);

% ---------- 终端安全滤波（式 32），逐点二维投影 ----------
term_s = 0; term_moved = 0; term_fail = 0; term_rescued = 0;
audit_d_ours = []; audit_d_ref = []; audit_done = 0;
if term_on
    t_term = tic;
    for i = 1:n_gen
        [z(:,i), mv, fl] = terminal_filter(z(:,i));
        term_moved = term_moved + mv;  term_fail = term_fail + fl;
    end
    term_s = toc(t_term);
end
audit = struct('n', audit_done, 'dist_ours', audit_d_ours, ...
    'dist_ref', audit_d_ref);
if audit_done > 0
    gap = audit_d_ours - audit_d_ref;
    audit.max_gap = max(gap);
    audit.mean_gap = mean(gap);
    audit.max_rel_gap = max(gap ./ max(audit_d_ref, 1e-12));
    audit.n_ours_worse = nnz(gap > 1e-9);
end

Xgen = z .* sd + mu;
Fgen = permute(reshape(Xgen, nF, nP, n_gen), [2 3 1]);
Pgen = Fgen(:,:,1:2);

out = struct('points', Pgen, 'features', Fgen, 'state', z, 'mode', mode, ...
    'sample_seconds', sample_s, 'rollout_seconds', roll_s, ...
    'terminal_seconds', term_s, ...
    'total_seconds_per_traj', (sample_s+roll_s+term_s)/n_gen, ...
    'n_qp', n_qp, 'slack_active', slack_hits, ...
    'terminal_corrected', term_moved, 'terminal_failed', term_fail, ...
    'terminal_rescued_by_fmincon', term_rescued, 'terminal_audit', audit, ...
    'segment', segment, 'obstacle', obst, 'geometry', geom, ...
    'margin', margin, 'n_gen', n_gen, 'u_per_stage', u_stage, ...
    ... % ---- 复现所需的种子与初始噪声 ----
    'z0', z0, 'rollout_seed', seed, 'trajectory_seeds', traj_seeds, ...
    'seed_convention', ['每条轨迹: randn(RandStream(''mt19937ar'',''Seed'',s), ', ...
        num2str(D), ', 1); 种子表由 rollout_seed 经 randi 派生'], ...
    'data_seed', cfg.random_seed, ...
    'weight_init_seed', struct_field_default(net, 'weight_init_seed', NaN), ...
    ... % ---- u 轨迹（物理空间，仅 CFMBF 激活后的步）----
    'u_trace', u_trace, 'u_trace_t', u_trace_t, ...
    'u_trace_slack', u_trace_slack, ...
    'u_trace_layout', ['(激活步, 样本, 轨迹点, xy) 物理空间修正量; ', ...
        'u_trace_t 为对应的生成时间 t'], ...
    'n_rk_steps', n_rk, 't_max', t_max, 'activation_time', t_bar, ...
    'phi1_switch_time', eps_sw, 'phi0', phi0);

% =====================================================================
    function V = raw(zc, tc)
        V = nn_fwd(net.P, [repmat(tc,1,size(zc,2)); zc]);
    end

    function V = vel(zc, tc)
        V = raw(zc, tc);
        if ~use_safety || tc < t_bar, return; end
        for s = 1:size(zc,2)
            V(:,s) = V(:,s) + guidance(zc(:,s), V(:,s), tc);
        end
    end

    % Alg.1 步骤 6-9：物理空间求 u，再标准化加回
    function un = guidance(zc, vn, tc)
        un = zeros(D,1);
        sl_vec = false(nP,1);
        for ip = 1:nP
            base = nF*(ip-1);
            id = base + (1:2);                 % 仅 [x,y] 参与安全约束
            p  = zc(id).*sd(id) + mu(id);      % 反标准化位置
            vp = vn(id).*sd(id);               % 反标准化速度
            [a, Bm] = point_rows(p, vp, tc);   % a_j, b_j（物理空间）
            if isempty(a), continue; end
            n_qp = n_qp + 1;
            [uk, sl] = qp2d(a, Bm);            % 二维解耦 QP
            if sl, slack_hits = slack_hits + 1; end
            sl_vec(ip) = sl;
            un(id) = uk ./ sd(id);             % 标准化后加回
        end
        last_slack_col = sl_vec;
    end

    % 该点的全部 N 个约束：a_j = grad_h'*v + phi*h,  b_j = grad_h
    function [a, Bm] = point_rows(p, vp, tc)
        a = zeros(nObs+2,1); Bm = zeros(nObs+2,2); n = 0;
        for jo = 1:nObs
            [h, gp] = obstacle_level_and_gradient(p, obst, jo);
            [a,Bm,n] = add(a,Bm,n,h,gp,vp,tc);
        end
        for bi = 1:2
            [hr, gp] = evaluate_track_implicit_field(geom.implicit_fields, bi, p);
            [a,Bm,n] = add(a,Bm,n,hr-margin,gp,vp,tc);
        end
        a = a(1:n); Bm = Bm(1:n,:);
    end

    function [a,Bm,n] = add(a,Bm,n,h,gp,vp,tc)
        g = gp(:).';
        if norm(g) < 1e-9, return; end
        n = n+1;  Bm(n,:) = g;  a(n) = g*vp + phi_val(h,tc)*h;
    end

    % 式 42
    function ph = phi_val(h, tc)
        if h >= 0
            ph = phi0;
        elseif tc < eps_sw
            ph = 1 + 4*tc^3;
        else
            ph = 1 / max(1 - tc, 1e-6);
        end
    end

    % 式 30：min |u|^2 + sum d_j^2  s.t. a_j + b_j'u + d_j >= 0, d_j >= 0
    % 最优 d_j = max(0, -(a_j+b_j'u))，代回得二维严格凸分片二次目标
    %   J(u) = |u|^2 + sum_j min(0, a_j+b_j'u)^2
    % 穷举 slack 激活集 A（<=2^N 个，N<=5），每个候选
    %   u_A = -(I + B_A'B_A) \ (B_A' a_A)
    % 由 2x2 正定线性方程给出（I 项保证正定，无需伪逆）。
    %
    % 精确性：设全局最优 u*，其激活集 A* = {j : a_j+b_j'u* < 0}。在 u* 邻域
    % J 与二次函数 Q_{A*} 重合，且 grad J(u*) = 0 => u* = u_{A*}，故 u* 必在
    % 候选集中。对全部候选评估真实 J 取最小者，即为精确全局解。
    % （这与逐候选检查活跃集一致性等价，但对 a_j+b_j'u = 0 的退化情形更稳。）
    function [u, used_slack] = qp2d(a, Bm)
        N = numel(a);
        best_u = zeros(2,1);
        best_f = a.'*0 + sum(max(0,-a).^2);   % V = 空集, u = 0
        for mask = 1:(2^N - 1)
            V = bitget(mask, 1:N) > 0;
            Bv = Bm(V,:);  av = a(V);
            M  = eye(2) + (Bv.' * Bv);
            uc = M \ (-(Bv.' * av));
            r  = a + Bm*uc;
            f  = uc.'*uc + sum(max(0,-r).^2);
            if f < best_f, best_f = f; best_u = uc; end
        end
        u = best_u;
        used_slack = any(a + Bm*u < -1e-9);
    end

    % 式 32：逐点 min |p - p0|^2 s.t. h_j(p) >= 0
    %
    % 注意：h_j 非线性，且障碍物外部安全域非凸，式 32 原问题可能有多个局部
    % 极小。这里用序贯线性化：每步解一个二维线性约束子问题
    %   min |w|^2 s.t. G*w >= c
    % 该**子问题**由穷举活跃集精确求解（见 minnorm_halfspaces），但整体收敛
    % 到的是式 32 的一个 KKT 点 / 局部极小，不保证全局最优。
    % 对 Safety 指标无影响（只要求可行）；对 KL/CS/AS 的影响是投影位移可能
    % 略大于全局最优。收敛后统一校验 h_j >= 0，失败计入 terminal_failed。
    function [zf, moved, failed] = terminal_filter(z0)
        zf = z0; moved = 0; failed = 0;
        for ip = 1:nP
            base = nF*(ip-1);
            id = base + (1:2);                 % 切向特征保持终端 NN 输出不变
            p0 = z0(id).*sd(id) + mu(id);
            p  = p0;
            for it = 1:term_it
                [hv, G] = point_h(p);
                if all(hv >= -term_tol), break; end
                % min |p+d-p0|^2 s.t. h + G*d >= 0
                % 令 w = d - (p0-p): min |w|^2 s.t. G*w >= -h - G*(p0-p)
                e  = p0 - p;
                c  = -hv - G*e;
                w  = minnorm_halfspaces(G, c);
                p_new = p0 + w;
                if norm(p_new - p) < 1e-14, break; end
                p = p_new;
            end
            hv = point_h(p);
            % 兜底：序贯线性化没能收敛到可行点时，换 fmincon 多起点重解。
            % 论文只论证式 32 可行，未给求解器；这里用第二个独立求解器补一次，
            % 仍不可行才计入 terminal_failed。
            if any(hv < -1e-8)
                [p_fb, ok_fb] = fmincon_project(p0, p);
                if ok_fb
                    p = p_fb;  hv = point_h(p);
                    term_rescued = term_rescued + 1;
                end
            end
            if any(hv < -1e-8), failed = failed + 1; end
            if norm(p - p0) > 1e-12
                moved = 1;
                % 审计：与 fmincon 多起点参考解比投影位移
                if audit_done < audit_n
                    [p_ref, ok_ref] = fmincon_project(p0, p);
                    if ok_ref
                        audit_done = audit_done + 1;
                        audit_d_ours(end+1,1) = norm(p - p0);   %#ok<AGROW>
                        audit_d_ref(end+1,1)  = norm(p_ref - p0); %#ok<AGROW>
                    end
                end
            end
            zf(id) = (p - mu(id))./sd(id);
        end
    end

    % 式 32 的独立参考求解器：fmincon(sqp) + 多起点，取可行解中位移最小者。
    % 起点包含序贯线性化的结果、原始点 p0、以及沿各违反约束梯度推到边界上的
    % 投影点——非凸问题下不同盆地各给一个初值。
    function [pb, ok] = fmincon_project(p0, pstart)
        pb = pstart; ok = false; best = inf;
        cands = [pstart, p0];
        [hv0, G0] = point_h(p0);
        for j = 1:numel(hv0)
            g = G0(j,:).';
            if hv0(j) < 0 && norm(g) > 1e-9
                cands(:,end+1) = p0 - hv0(j)*g/(g.'*g);  %#ok<AGROW>
            end
        end
        for c = 1:size(cands,2)
            try
                p_try = fmincon(@obj_dist, cands(:,c), [],[],[],[],[],[], ...
                    @nlc_h, fmincon_opts);
            catch
                continue;
            end
            if any(point_h(p_try) < -1e-8), continue; end
            d = norm(p_try - p0);
            if d < best, best = d; pb = p_try; ok = true; end
        end
        function [f, g] = obj_dist(p)
            e = p - p0;  f = e.'*e;  g = 2*e;
        end
    end

    function [c, ceq, gc, gceq] = nlc_h(p)
        [hv, G] = point_h(p);
        c = -hv;  ceq = [];        % h >= 0  <=>  -h <= 0
        gc = -G.'; gceq = [];      % fmincon 要 (n_var x n_con)
    end

    function [hv, G] = point_h(p)
        hv = zeros(nObs+2,1); G = zeros(nObs+2,2);
        for jo = 1:nObs
            [h, gp] = obstacle_level_and_gradient(p, obst, jo);
            hv(jo) = h;  G(jo,:) = gp(:).';
        end
        for bi = 1:2
            [hr, gp] = evaluate_track_implicit_field(geom.implicit_fields, bi, p);
            hv(nObs+bi) = hr - margin;  G(nObs+bi,:) = gp(:).';
        end
    end
end

% =====================================================================
% min |w|^2  s.t.  G*w >= c   （二维，穷举活跃集，精确解）
function w = minnorm_halfspaces(G, c)
if all(c <= 0), w = zeros(size(G,2),1); return; end
N = size(G,1);  best_w = [];  best_n = inf;
for mask = 1:(2^N - 1)
    S  = bitget(mask, 1:N) > 0;
    Gs = G(S,:);  cs = c(S);
    Gg = Gs*Gs.';
    if rcond(Gg) < 1e-12
        lam = pinv(Gg) * cs;
    else
        lam = Gg \ cs;
    end
    wc = Gs.' * lam;
    if any(G*wc < c - 1e-9), continue; end     % 必须对全部约束可行
    nw = wc.'*wc;
    if nw < best_n, best_n = nw; best_w = wc; end
end
if isempty(best_w)
    % 退化：所有子集都不可行（约束互相矛盾），取违反最小的最小二乘解
    best_w = pinv(G) * max(c, 0);
end
w = best_w;
end

% =====================================================================
function y = nn_fwd(P, x)
sil = @(z) z./(1+exp(-z));
a1 = sil(P{1}*x  + P{5});
a2 = sil(P{2}*a1 + P{6});
a3 = sil(P{3}*a2 + P{7});
y  = P{4}*a3 + P{8};
end
