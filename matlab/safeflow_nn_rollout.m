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

% Limit the number of trajectories evaluated together.  In particular,
% rollout_batch_size=1 makes both the NN forward passes and the safety-QP
% rollout trajectory-wise, matching the memory/timing configuration used by
% the hierarchy experiments.  Seeds are derived once before splitting so the
% result is invariant to the chosen batch size.
rollout_batch_size = gf('rollout_batch_size', n_gen);
assert(isscalar(rollout_batch_size) && rollout_batch_size >= 1 && ...
    rollout_batch_size == floor(rollout_batch_size), ...
    'rollout_batch_size must be a positive integer.');
if rollout_batch_size < n_gen
    all_seeds = gf('trajectory_seeds', []);
    if isempty(all_seeds)
        batch_seed = gf('seed', 11);
        batch_stream = RandStream('mt19937ar', 'Seed', batch_seed);
        all_seeds = randi(batch_stream, [1, 2^31-1], n_gen, 1);
    end
    all_seeds = all_seeds(:);
    assert(numel(all_seeds) == n_gen, ...
        'The number of trajectory seeds must equal n_gen.');
    out = [];
    for first = 1:rollout_batch_size:n_gen
        idx = first:min(first + rollout_batch_size - 1, n_gen);
        batch_opts = opts;
        batch_opts.trajectory_seeds = all_seeds(idx);
        batch_opts.rollout_batch_size = numel(idx);
        q = safeflow_nn_rollout(net, mode, numel(idx), batch_opts);
        if isempty(out), out = q; else, out = append_rollout_batches(out, q); end
    end
    out.total_seconds_per_traj = ...
        (out.sample_seconds + out.rollout_seconds + out.terminal_seconds) / n_gen;
    out.rollout_batch_size = rollout_batch_size;
    out.options.rollout_batch_size = rollout_batch_size;
    return;
end

t_bar   = gf('activation_time', 0.5);
eps_sw  = gf('phi1_switch_time', 0.0);
phi0    = gf('phi0', 1.0);
% phi1_early_coeff: coefficient c in the paper's early branch phi1 = 1 + c*t^3.
% The paper fixes c = 4; exposing it measures where the front end becomes
% discretely unstable (phi*dt > 1 at t = phi1_switch_time).
early_c = gf('phi1_early_coeff', 4.0);
% The formal stress run uses a deliberately large prescribed-time gain
% omega/(1-t)^2 from the first guided step, exposing the pre-projection
% divergence discussed by the supervisor. 'first_order' remains available
% only for reproducing the original SafeFlow paper's 1/(T-t) gain.
%   'constant' drops the prescribed-time pole entirely and uses a plain
%   class-K rate phi1_alpha -- the 'conventional CBF' that SafeFlow's Sec. IV
%   attributes the SafeDiffuser variants' low safety rate to.
phi1_form = gf('phi1_form', 'second_order');
phi1_om   = gf('phi1_omega', 20.0);
phi1_al   = gf('phi1_alpha', 1.0);       % rate used when phi1_form='constant'
% robust_margin: require a + b'u >= rho instead of >= 0, i.e. push strictly
% inward rather than merely holding the boundary.  This is the 'robust' knob
% of the RoSD-style baseline; 0 reproduces the plain constraint.
rob_rho   = gf('robust_margin', 0.0);
% ---- SafeDiffuser-style mechanisms (Xiao et al., ICLR 2025) ----
% These are OFF by default; each one changes the constraint in the way that
% paper's corresponding variant does, which is NOT the same as changing the
% class-K gain.
%
% robust_shrink (RoSD): contract the safe set itself, h -> h - c.  The authors'
% Maze2D code uses c = 0.01.  Distinct from robust_margin above, which instead
% adds a fixed offset to the derivative inequality; that one is kept only so the
% earlier, cruder ablation stays reproducible.
rob_shrink = gf('robust_shrink', 0.0);
% gamma_mode (TVSD): a time-varying SAFE SET, h - gamma(t) >= 0, with gamma
% rising from gamma_min at t=0 to 0 at t=1 through a sigmoid.  The CBF row then
% carries the extra -gamma_dot term, so the tightening rate is accounted for.
% 'none' leaves the safe set fixed.
gamma_mode = gf('gamma_mode', 'none');
gamma_min  = gf('gamma_min', -1.0);      % gamma(0) for the obstacle rows
% gamma_min_boundary: the paper builds gamma_k per trajectory point AND per
% specification, requiring only gamma_k(N) <= b_k(x_k^N).  Sharing one value
% across constraints of different scale is not merely conservative: writing
% gamma = -c*q(t) with q(0)=1, q(1)=0, the net contribution to the constraint is
% c*(q'(t) + alpha*q(t)), which is NEGATIVE (i.e. tightening) wherever |q'| >
% alpha*q -- the whole middle of a sigmoid -- and its magnitude scales with c.
% So giving the track boundary the obstacle's c = 1 multiplies its mid-rollout
% tightening by the ratio of the two scales.  Measured at t = 0: obstacle h
% bottoms out at -0.9893, boundary h at -0.0590, a factor of 16.8.
% [] means 'same as gamma_min', which reproduces the earlier shared behaviour.
gamma_min_b = gf('gamma_min_boundary', []);
if isempty(gamma_min_b), gamma_min_b = gamma_min; end
gamma_k    = gf('gamma_steepness', 10.0);
gamma_t0   = gf('gamma_midpoint', 0.5);
% 'linear' is gamma(t) = gamma_min*(1-t): the unique schedule with a constant
% closing rate.  Since int_0^1 gamma_dot dt = |gamma_min| is fixed by the two
% endpoint conditions, spreading it uniformly minimises the peak demand, and a
% minimum-norm QP pays the square of that demand.  It also removes the steepness
% and midpoint parameters, which the paper does not specify.
assert(ismember(gamma_mode, {'none','sigmoid','linear'}), ...
    'gamma_mode must be none, sigmoid or linear');
% resd_weight (ReSD): the relaxation variable r enters as a + b'u + w(t) r >= 0
% with cost |u|^2 + |r|^2.  Eliminating r makes that identical to our penalty
% form with slack weight 1/w(t)^2, and w -> 0 turns the constraint hard.  The
% authors' Maze2D code steps the weight from 100 early to 0 late.
% [] disables the schedule and keeps the constant slack_weight.
resd_w     = gf('resd_weight_schedule', []);   % [w_early, w_late, switch_time]
% extra_hard_steps (ReSD, N_a): after the main rollout, run N_a further steps
% with the relaxation weight at 0, i.e. a hard constraint, holding the model
% time at t_max.  Theorem 3 needs these: the relaxed constraint alone does not
% end at b >= 0, and the maze experiment uses N_a = 50.  0 disables them.
n_extra    = gf('extra_hard_steps', 0);
% extra_steps_use_flow: during those N_a steps the paper holds the diffusion
% time at 0, where the denoising step is essentially the identity, so the
% nominal drift is ~0 and only the invariance correction acts.  A flow-matching
% v_theta(.,t_max) is NOT near zero, so integrating it for N_a more steps would
% carry the samples off the terminal distribution -- an artefact of the port,
% not of the method.  Default false = pure QP correction.
extra_flow = gf('extra_steps_use_flow', false);
assert(ismember(phi1_form, {'first_order','second_order','constant'}), ...
    'phi1_form must be first_order, second_order or constant');
% slack_enabled: false removes the slack variable from (30) entirely, so the
% QP is the hard min-norm problem  min|u|^2 s.t. a + B u >= 0.  Infeasible
% points are counted in slack_hits (reused as an infeasibility counter) and
% fall back to the least-violation solution.  The paper always keeps slack.
use_slack = gf('slack_enabled', true);
% qp_backend: which solver runs the per-point QP (29)-(30) and the terminal
% projection (32) subproblem.  'closed' is the exact active-set enumeration;
% 'quadprog' hands the same program to MATLAB's generic QP solver.  Solutions
% agree to solver tolerance (see verify_qp2d.m); the run time does not.
qp_backend = gf('qp_backend', 'closed');
assert(ismember(qp_backend, {'closed','quadprog'}), ...
    'qp_backend must be closed or quadprog');
n_rk    = gf('n_steps', 100);
t_max   = gf('t_max', 0.996);
term_on = gf('terminal_filter', strcmp(mode,'safeflow'));
term_it = gf('terminal_max_iter', 200);
term_tol= gf('terminal_tol', 1e-10);
% Formal runs fail immediately on a terminal projection failure.  A diagnostic
% archive may disable the immediate throw so every failed point/trajectory is
% counted and saved; this never enables an alternative solver or rescues data.
term_fail_fast = gf('terminal_failure_fail_fast', true);
seed    = gf('seed', 11);
% Algorithm 1 第 8/10 行的字面读法：u_t 每个 RK4 步只解一次（用该步的
% t_k 与 T_t），然后作为固定向量加进流场，四级只重算 v^theta。
% true 则每级重解 QP（数值更自洽，但 QP 次数与耗时约 4 倍，属偏离论文）。
u_stage = gf('u_per_stage', false);
% 式 30 中 slack 相对 |u|^2 的权重。原文为 1（两项等权）。
% 加大它等价于把安全看得比“少改动流场”更重，可用来检验
% “引导阶段安全率低是目标函数权衡所致”这一判断。
w_slack = gf('slack_weight', 1e4);
% false 时只保留障碍约束、去掉左右赛道边界行——即 SafeFlow 原文平面导航
% 实验的口径（安全集只含 3 个椭圆，迷宫墙不进 QP）。
use_boundary_rows = gf('boundary_constraints', true);
% >0 时把该点的全部安全行按 soft-min 合成一行（与本项目三层方法同一口径，
% cfg.*_joint_safety_softmin_kappa = 2000）。0 = 保持原文的多行独立约束。
kappa = gf('softmin_kappa', 0);
% obstacle_inflation: 加在障碍半轴上的膨胀量（与 make_level_variance_constraint
% 的 inflated_semi_axes 同一口径）。SafeFlow 原文为 0（h 是精确水平集）。
% 引导只把 h_inflated 驱动到 0^-，故 inflation>0 时真实 h 收敛到 +inflation^-。
obs_infl = gf('obstacle_inflation', 0);
% 终端投影审计：对最多这么多个被移动的点，另用 fmincon 多起点重解式 32，
% 记录我们的序贯线性化解与该参考解的距离差，用来量化"局部 vs 全局"的差距。
% 0 = 关闭（默认）。审计可调用 fmincon 作离线参考，但不得用它改写生成结果。
audit_n = gf('terminal_audit_n', 0);
% 式 32 只用序贯线性化求解。fmincon 兜底不是论文算法的一部分，正式
% SafeFlow 禁止用它把失败样本救回；任何终端不可行都会在下面立即报错。
% The paper states (32) as a nonlinear constrained projection without naming a
% solver, so silently switching solvers would hide a failure of this implementation.
if gf('fmincon_fallback', false)
    error('SafeFlow:FminconFallbackDisabled', ...
        ['fmincon_fallback is disabled for the formal SafeFlow run. ', ...
         'A terminal projection failure must be reported, not rescued.']);
end
use_fmincon_fb = false;
% record_path: keep the 260-d state at every RK4 step so h_min(t) can be
% plotted the way the three-level method's diagnostics do.  Off by default
% because it is 100x the memory of the terminal state alone.
record_path = gf('record_path', false);

cfg = get_config();
rng(cfg.random_seed);
[~, segment] = scenario_training_points(cfg, 65, cfg.n_train);
obst_phys = configure_racing_obstacles(segment, cfg.obstacle);
obst = obst_phys;                     % 引导/投影用的（可膨胀）几何
if obs_infl ~= 0
    obst.semi_axes = obst.semi_axes + obs_infl;
end
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
% Split by which QP was actually solved: in a soft QP a hit means the optimum
% chose to pay the penalty; in a hard QP it means the half-space intersection
% was empty and the least-violation fallback ran.  ReSD is soft early and hard
% late, so the combined count mixes two different things.
slack_hits_soft = 0; infeas_hard = 0;
fmincon_opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
    'SpecifyObjectiveGradient', true, 'SpecifyConstraintGradient', true, ...
    'OptimalityTolerance', 1e-12, 'ConstraintTolerance', 1e-12, ...
    'StepTolerance', 1e-14, 'MaxIterations', 500);

z_path = [];
if record_path
    z_path = zeros(n_rk+1, D, n_gen);
    z_path(1,:,:) = z;
end
force_hard_qp = false;
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
    if record_path, z_path(k+1,:,:) = z; end
end
% ReSD's N_a extra steps: same integrator, model time frozen at t_max, and the
% relaxation switched off so the barrier condition is enforced exactly.
for ke = 1:n_extra
    force_hard_qp = true;
    if extra_flow
        v0 = raw(z, t_max);
    else
        v0 = zeros(D, n_gen);       % diffusion time held at 0: no further drift
    end
    U = zeros(D, n_gen);
    for s = 1:n_gen
        U(:,s) = guidance(z(:,s), v0(:,s), t_max);
    end
    z = z + dt*(v0 + U);            % the flow is frozen, so RK4's four stages
    force_hard_qp = false;          % would all coincide; one Euler step is exact
    if record_path, z_path(end,:,:) = z; end
end
roll_s = toc(t_roll);

% ---------- 终端安全滤波（式 32），逐点二维投影 ----------
term_s = 0; term_moved = 0; term_fail = 0; term_rescued = 0;
term_fail_per_trajectory = zeros(n_gen, 1);
audit_d_ours = []; audit_d_ref = []; audit_done = 0;
if term_on
    t_term = tic;
    for i = 1:n_gen
        [z(:,i), mv, fl] = terminal_filter(z(:,i));
        term_moved = term_moved + mv;  term_fail = term_fail + fl;
        term_fail_per_trajectory(i) = fl;
        if fl > 0 && term_fail_fast
            error('SafeFlow:TerminalProjectionFailed', ...
                ['Terminal projection failed for trajectory %d (seed %.0f): ', ...
                 '%d point(s) remain outside the physical safe set.'], ...
                i, traj_seeds(i), fl);
        end
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
    'slack_active_soft', slack_hits_soft, 'infeasible_hard', infeas_hard, ...
    'terminal_corrected', term_moved, 'terminal_failed', term_fail, ...
    'terminal_failed_per_trajectory', term_fail_per_trajectory, ...
    'terminal_rescued_by_fmincon', term_rescued, 'terminal_audit', audit, ...
    'segment', segment, 'obstacle', obst_phys, 'geometry', geom, ...
    'obstacle_constraint', obst, 'obstacle_inflation', obs_infl, ...
    'margin', margin, 'n_gen', n_gen, 'u_per_stage', u_stage, ...
    ... % ---- 复现所需的种子与初始噪声 ----
    'z0', z0, 'z_path', z_path, 'rollout_seed', seed, 'trajectory_seeds', traj_seeds, ...
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
    'phi1_switch_time', eps_sw, 'phi0', phi0, ...
    'phi1_early_coeff', early_c, 'slack_enabled', use_slack, ...
    'phi1_form', phi1_form, 'phi1_omega', phi1_om, ...
    'qp_backend', qp_backend);
out.rollout_batch_size = rollout_batch_size;
% Complete record of every option that was actually in force, so an archive can
% regenerate this exact run.  Anything omitted here silently reverts to a
% default on replay, which is how an archive stops being reproducible.
out.options = struct( ...
    'activation_time', t_bar, 'phi1_switch_time', eps_sw, 'phi0', phi0, ...
    'phi1_early_coeff', early_c, 'phi1_form', phi1_form, 'phi1_omega', phi1_om, ...
    'phi1_alpha', phi1_al, 'robust_margin', rob_rho, ...
    'robust_shrink', rob_shrink, 'gamma_mode', gamma_mode, ...
    'gamma_min', gamma_min, 'gamma_min_boundary', gamma_min_b, ...
    'gamma_steepness', gamma_k, ...
    'gamma_midpoint', gamma_t0, 'resd_weight_schedule', resd_w, ...
    'extra_hard_steps', n_extra, 'extra_steps_use_flow', extra_flow, ...
    'n_steps', n_rk, 't_max', t_max, 'terminal_filter', term_on, ...
    'terminal_max_iter', term_it, 'terminal_tol', term_tol, 'seed', seed, ...
    'terminal_failure_fail_fast', term_fail_fast, ...
    'u_per_stage', u_stage, 'slack_weight', w_slack, 'slack_enabled', use_slack, ...
    'boundary_constraints', use_boundary_rows, 'softmin_kappa', kappa, ...
    'obstacle_inflation', obs_infl, 'terminal_audit_n', audit_n, ...
    'qp_backend', qp_backend, ...
    'fmincon_fallback', use_fmincon_fb, 'record_path', record_path);
out.options.rollout_batch_size = rollout_batch_size;

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
            [uk, sl] = qp2d(a, Bm, tc);        % 二维解耦 QP
            if sl, slack_hits = slack_hits + 1; end
            sl_vec(ip) = sl;
            un(id) = uk ./ sd(id);             % 标准化后加回
        end
        last_slack_col = sl_vec;
    end

    % 该点的全部 N 个约束：a_j = grad_h'*v + phi*h,  b_j = grad_h
    function [a, Bm] = point_rows(p, vp, tc)
        a = zeros(nObs+2,1); Bm = zeros(nObs+2,2); n = 0;
        hv = zeros(nObs+2,1); gv = zeros(nObs+2,2);
        for jo = 1:nObs
            [h, gp] = obstacle_level_and_gradient(p, obst, jo);
            [a,Bm,n] = add(a,Bm,n,h,gp,vp,tc,gamma_min);
            if n > 0, hv(n) = h; gv(n,:) = gp(:).'; end
        end
        if use_boundary_rows
            for bi = 1:2
                [hr, gp] = evaluate_track_implicit_field(geom.implicit_fields, bi, p);
                [a,Bm,n] = add(a,Bm,n,hr-margin,gp,vp,tc,gamma_min_b);
                if n > 0, hv(n) = hr-margin; gv(n,:) = gp(:).'; end
            end
        end
        a = a(1:n); Bm = Bm(1:n,:);
        if kappa > 0 && n > 1
            [a, Bm] = softmin_row(hv(1:n), gv(1:n,:), vp, tc);
        end
    end

    % h_soft = -(1/k) log sum exp(-k h_i);  grad = sum_i w_i grad h_i
    function [a1, B1] = softmin_row(hs, gs, vp, tc)
        m  = min(hs);
        e  = exp(-kappa * (hs - m));
        se = sum(e);
        hsoft = m - log(se)/kappa;
        w  = e / se;
        g  = (w.' * gs);
        if norm(g) < 1e-12, a1 = zeros(0,1); B1 = zeros(0,2); return; end
        hs = hsoft - rob_shrink - gam(tc);
        a1 = g*vp(:) - gam_dot(tc) + phi_val(hs, tc)*hs - rob_rho;
        B1 = g;
    end

    function [a,Bm,n] = add(a,Bm,n,h,gp,vp,tc,gmin)
        g = gp(:).';
        if norm(g) < 1e-9, return; end
        % RoSD contracts the safe set; TVSD moves it and contributes -gamma_dot.
        hs = h - rob_shrink - gam(tc, gmin);
        n = n+1;  Bm(n,:) = g;
        a(n) = g*vp - gam_dot(tc, gmin) + phi_val(hs,tc)*hs - rob_rho;
    end

    % 式 42
    % TVSD: gamma(t) sweeps the safe set from permissive to the true one.
    function g = gam(tc, gmin)
        if nargin < 2, gmin = gamma_min; end
        % Both forms give gam(0) = gamma_min and gam(1) = 0 exactly, which is the
        % pair of endpoint conditions the variant needs.
        switch gamma_mode
            case 'none',   g = 0;
            case 'linear', g = gmin * (1 - tc);
            otherwise,     g = gmin * (sig(1) - sig(tc)) / (sig(1) - sig(0));
        end
    end
    function gd = gam_dot(tc, gmin)
        if nargin < 2, gmin = gamma_min; end
        switch gamma_mode
            case 'none',   gd = 0;
            case 'linear', gd = -gmin;               % constant closing rate
            otherwise
                sd_ = gamma_k * sig(tc) * (1 - sig(tc));
                gd  = -gmin * sd_ / (sig(1) - sig(0));
        end
    end
    function y = sig(tc)
        y = 1 / (1 + exp(-gamma_k * (tc - gamma_t0)));
    end

    function ph = phi_val(h, tc)
        if h >= 0
            ph = phi0;
        elseif tc < eps_sw
            ph = 1 + early_c*tc^3;
        else
            switch phi1_form
                case 'second_order'
                    ph = phi1_om / max(1 - tc, 1e-6)^2;
                case 'constant'
                    ph = phi1_al;
                otherwise
                    ph = 1 / max(1 - tc, 1e-6);
            end
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
    function [u, used_slack] = qp2d(a, Bm, tc)
        % ReSD: w(t) -> 0 makes the relaxation useless, i.e. a hard constraint;
        % otherwise the equivalent penalty weight is 1/w(t)^2.
        w_slack_t = w_slack;  hard_now = ~use_slack || force_hard_qp;
        if ~isempty(resd_w)
            wt = resd_w(1);  if tc >= resd_w(3), wt = resd_w(2); end
            if wt <= 0, hard_now = true; else, w_slack_t = 1/wt^2; end
        end
        was_hard = hard_now;
        if hard_now
            % Hard constraint: no slack term in (30).
            u = minnorm_halfspaces(Bm, -a, qp_backend);
            used_slack = any(a + Bm*u < -1e-9);   % = infeasible here
            if used_slack, infeas_hard = infeas_hard + 1; end
            return;
        end
        N = numel(a);
        if strcmp(qp_backend, 'quadprog')
            % Explicit-slack form of (30):
            %   min |u|^2 + w*sum d_j^2   s.t.  a_j + b_j'u + d_j >= 0, d_j >= 0
            u = qp2d_quadprog(a, Bm, w_slack_t);
            used_slack = any(a + Bm*u < -1e-9);
            if used_slack && ~was_hard, slack_hits_soft = slack_hits_soft + 1; end
            return;
        end
        best_u = zeros(2,1);
        best_f = w_slack_t * sum(max(0,-a).^2);   % V = 空集, u = 0
        for mask = 1:(2^N - 1)
            V = bitget(mask, 1:N) > 0;
            Bv = Bm(V,:);  av = a(V);
            M  = eye(2) + w_slack_t * (Bv.' * Bv);
            uc = M \ (-w_slack_t * (Bv.' * av));
            r  = a + Bm*uc;
            f  = uc.'*uc + w_slack_t * sum(max(0,-r).^2);
            if f < best_f, best_f = f; best_u = uc; end
        end
        u = best_u;
        used_slack = any(a + Bm*u < -1e-9);
        if used_slack && ~was_hard, slack_hits_soft = slack_hits_soft + 1; end
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
                w  = minnorm_halfspaces(G, c, qp_backend);
                p_new = p0 + w;
                if norm(p_new - p) < 1e-14, break; end
                p = p_new;
            end
            hv = point_h(p);
            % 兜底：序贯线性化没能收敛到可行点时，换 fmincon 多起点重解。
            % 论文只论证式 32 可行，未给求解器；这里用第二个独立求解器补一次，
            % 仍不可行才计入 terminal_failed。
            if any(hv < -1e-8) && use_fmincon_fb
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

function a = append_rollout_batches(a, b)
a.points = cat(2, a.points, b.points);
a.features = cat(2, a.features, b.features);
a.state = cat(2, a.state, b.state);
a.z0 = cat(2, a.z0, b.z0);
a.z_path = cat(3, a.z_path, b.z_path);
a.trajectory_seeds = [a.trajectory_seeds; b.trajectory_seeds];
a.u_trace = cat(2, a.u_trace, b.u_trace);
a.u_trace_slack = cat(2, a.u_trace_slack, b.u_trace_slack);
a.terminal_failed_per_trajectory = [a.terminal_failed_per_trajectory; ...
    b.terminal_failed_per_trajectory];
sum_fields = {'sample_seconds','rollout_seconds','terminal_seconds','n_qp', ...
    'slack_active','slack_active_soft','infeasible_hard','terminal_corrected', ...
    'terminal_failed','terminal_rescued_by_fmincon','n_gen'};
for k = 1:numel(sum_fields)
    f = sum_fields{k}; a.(f) = a.(f) + b.(f);
end
end

function y = nn_fwd(P, x)
sil = @(z) z./(1+exp(-z));
a1 = sil(P{1}*x  + P{5});
a2 = sil(P{2}*a1 + P{6});
a3 = sil(P{3}*a2 + P{7});
y  = P{4}*a3 + P{8};
end

function u = qp2d_quadprog(a, Bm, w_slack)
%QP2D_QUADPROG  Explicit-slack form of (30) solved by MATLAB's QP solver.
%   min_{u,d}  |u|^2 + w*sum_j d_j^2
%   s.t.       a_j + b_j'u + d_j >= 0,   d_j >= 0
% Same program the active-set enumeration in qp2d solves after eliminating d;
% verify_qp2d.m cross-checks the two.  Falls back to the unconstrained u = 0
% only if the solver fails outright, which cannot happen for this convex QP.
persistent opts
if isempty(opts)
    opts = optimoptions('quadprog', 'Display', 'off', ...
        'Algorithm', 'interior-point-convex', ...
        'OptimalityTolerance', 1e-12, 'ConstraintTolerance', 1e-12);
end
N = numel(a);
H = 2 * diag([1; 1; w_slack * ones(N, 1)]);
A = [-Bm, -eye(N)];
lb = [-inf; -inf; zeros(N, 1)];
[x, ~, exitflag] = quadprog(H, zeros(2 + N, 1), A, a(:), [], [], lb, [], ...
    [], opts);
if exitflag <= 0 || isempty(x)
    u = zeros(2, 1);
else
    u = x(1:2);
end
end
