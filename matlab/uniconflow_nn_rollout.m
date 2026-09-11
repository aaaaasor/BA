function out = uniconflow_nn_rollout(net, n_gen, opts)
%UNICONFLOW_NN_ROLLOUT  Stage 1 of UniConFlow: PTZF-guided flow matching.
%
% Xu et al., "UniConFlow", Sec. V-C (Certified Path Planning), Eqs. (85)-(103),
% and Algorithm 2's first block.  The CEM window refinement that follows it in
% Algorithm 2 is NOT in this file.
%
% ---------------------------------------------------------------------------
% AUGMENTED STATE.  The FM model produces states only, so the paper introduces
% a virtual action flow driven purely by its own guidance (Eq. 88):
%
%   T^s = 65 points x [x, y, tx, ty]           = 260   (the FM state)
%   T^a = 64 segments x [nu_k, kappa_k]        = 128   (virtual actions)
%   d   = 388,   u_aug = [u^s; u^a]
%
%   p_{k+1} = p_k + nu_k * t_k
%   t_{k+1} = R(kappa_k) * t_k,   R = [cos -sin; sin cos]
%
% Both nu_k and kappa_k are needed.  With nu alone, t_{k+1} is not determined by
% (p_k, t_k, nu_k), so the terminal CEM could only rescale steps along fixed
% directions and could never steer around an obstacle -- that is not the
% paper's s_{k+1} = f(s_k, a_k).
%
% AGGREGATED EQUALITY (Eq. 89 form: a sum of squares, hence ONE scalar row):
%
%   g = sum_k ||p_{k+1} - p_k - nu_k t_k||^2      kinodynamic consistency
%     + sum_k ||t_{k+1} - R(kappa_k) t_k||^2      tangent transition
%     + sum_k (||t_k||^2 - 1)^2                   unit tangents
%
% The unit-norm term is not decoration: without it the QP can inflate t_k and
% shrink nu_k with no change to the first residual, a pure scale degeneracy.
%
% g_ini is deliberately ABSENT.  The paper anchors p_0 to the robot's current
% state, but this racing task has no externally fixed start -- which is also
% why the metric table carries no SD/ED columns.
%
% INEQUALITIES, in the paper's convention h_j <= 0 = safe (ours is b >= 0, so
% h = -b):
%   65 points x (3 obstacles + 2 track boundaries)          325 rows
%   64 segments x (-nu, nu - nu_max, kap - kap_max, -kap - kap_max)  256 rows
%   plus the single aggregated equality                       1 row
%                                                           --- 582 rows
%
% PTZF (Def. 6, Example 2) with T_pre = 1, c_r = 0, gamma_r(t,r) = c_g r:
%
%   rbar(t) = rbar(0) exp(-c_g t/(1-t)),   rbar'(t) = -c_g rbar(t)/(1-t)^2
%
% initialised, per the paper's appendix, at gbar(0) = 2 g(T_0) and
% hbar_j(0) = h_j(T_0).  T_0 is Gaussian noise and the virtual actions have no
% model, so T^a_0 is a fixed prior (nu = the training median, kappa = 0) and
% g(T_0) is then evaluated honestly on that pair.  Warm-starting the actions
% from an unguided FM pass would be a second sampling run, which Algorithm 2
% does not have.
%
% GUIDANCE class-K: gamma(a) = c_PT a / (1-t)^2 with c_PT > c_g (Sec. IV-D).
%
% QP.  Eq. (49): min u'Pu u + d'Pd d  s.t.  rho + eta u = d.  Eliminating the
% slack leaves an unconstrained least squares,
%
%   (Pu + eta' Pd eta) u = -eta' Pd rho,
%
% and the structure makes that cheap: every safety row touches one point's two
% position coordinates, every action row touches one action coordinate, and the
% aggregated equality is the ONLY dense row.  So the matrix is
% block-diagonal + rank one and Sherman-Morrison solves it in O(d).  That is
% our implementation choice; the paper states the closed form (51) without
% saying how to evaluate it.
%
% Only rows with rho_j > 0 enter.  The equality form would otherwise drag
% already-satisfied rows back onto their boundary.  The paper writes the guard
% as rho > 0 elementwise, which would require every row to be violated; we read
% it per row.
%
% NOT A PARAMETER-LEVEL REPRODUCTION.  P_s, P_a, P_delta, c_PT, the step count,
% and the handling of the 1/(1-t)^2 singularity are not published.

if nargin < 2 || isempty(n_gen), n_gen = 100; end
if nargin < 3, opts = struct(); end
gf = @(f,d) struct_field_default(opts, f, d);

n_steps = gf('n_steps', 100);
seed    = gf('seed', 11);
c_g     = gf('c_g', 1.0);            % PTZF decay, gamma_r(t,r) = c_g r
c_PT    = gf('c_PT', 2.0);           % guidance gain, must exceed c_g
assert(c_PT > c_g, 'c_PT must exceed c_g (Sec. IV-D)');
% Ps > Pa (Eq. 102): penalise changes to the FM state more than to the virtual
% actions, which carry no learned distribution to preserve.
P_s     = gf('P_state', 1.0);
P_a     = gf('P_action', 0.01);
% Eq. (49) permits a positive-definite P_delta matrix.  Separate weights keep
% the paper's formulation while letting state/action feasibility be enforced
% more strongly than the single aggregated consistency residual.
P_d_base   = gf('P_slack', 1000.0);
P_d_g      = gf('P_slack_equality', P_d_base);
P_d_state  = gf('P_slack_state', P_d_base);
P_d_action = gf('P_slack_action', 10*P_d_base);
nu_max  = gf('nu_max', 0.065);       % covers 100% of the training segments
kap_max = gf('kappa_max', 1.15);     % 65.9 deg, likewise
nu_init = gf('nu_init', 0.04812);    % training median segment length
eps_t   = gf('singularity_floor', 0.5);   % fallback floor, relative to current dt
adaptive_steps = gf('adaptive_steps', true);
q_max   = gf('adaptive_q_max', 0.5); % enforce c_PT*dt/(1-t)^2 <= q_max
t_end   = gf('t_end', 0.996);        % finite approximation of tau -> infinity
max_steps = gf('max_adaptive_steps', 20000);
% The transformed RK4 implementation is available for the paper-limit audit,
% but adaptive Euler remains the default: on this model it is 3.5x faster and
% produced smaller residuals at the same finite terminal time.
integration_mode = char(gf('integration_mode', 'adaptive_euler'));
transform_switch = gf('transform_switch', 0.9);
transform_step = gf('transform_step', 0.25);
terminal_tol = gf('terminal_tolerance', 1e-8);
assert(isscalar(q_max) && isfinite(q_max) && q_max > 0, ...
    'adaptive_q_max must be finite and positive');
assert(isscalar(t_end) && isfinite(t_end) && t_end > 0 && t_end < 1, ...
    't_end must lie strictly between 0 and 1');
assert(isscalar(max_steps) && max_steps >= n_steps, ...
    'max_adaptive_steps must be at least n_steps');
assert(any(strcmp(integration_mode, {'transformed_rk4','adaptive_euler'})), ...
    'integration_mode must be transformed_rk4 or adaptive_euler');
assert(transform_switch > 0 && transform_switch < t_end, ...
    'transform_switch must lie in (0,t_end)');
assert(transform_step > 0 && isfinite(transform_step), ...
    'transform_step must be finite and positive');
% debug_steps > 0 prints the per-step diagnosis for trajectory 1, including
% whether the frozen active set is self-consistent: rows that were inactive
% before the solve but violated after it prove that one round is not enough.
dbg     = gf('debug_steps', 0);
% Outer active-set passes.  1 reproduces the old single frozen solve.
max_as  = gf('max_active_set_passes', 8);
assert(isscalar(max_as) && max_as >= 1, 'max_active_set_passes must be >= 1');
record_path = gf('record_path', false);
trace_on    = gf('record_u_trace', true);

% ---------- scene ----------
cfg = get_config();
rng(cfg.random_seed);
[~, segment] = scenario_training_points(cfg, 65, cfg.n_train);
obst = configure_racing_obstacles(segment, cfg.obstacle);
geom = build_track_boundary_geometry(segment, ...
    struct_field_default(cfg.track_boundary, 'n_spline_points', 400));
nObs = size(obst.centers, 2);

sd = net.sd_d(:);  mu = net.mu_d(:);
D  = numel(mu);
nP = struct_field_default(net, 'n_points', 65);
nF = struct_field_default(net, 'n_features_per_point', D/nP);
assert(nF == 4 && D == nP*nF, 'state must be n_points x 4 features');
nSeg = nP - 1;
dAug = D + 2*nSeg;

ix = @(k) nF*(k-1) + (1:2);          % position rows of point k, 1-based
it = @(k) nF*(k-1) + (3:4);          % tangent rows of point k
inu = @(k) D + k;                    % nu_k
ika = @(k) D + nSeg + k;             % kappa_k

% ---------- seeds ----------
traj_seeds = gf('trajectory_seeds', []);
if isempty(traj_seeds)
    seed_stream = RandStream('mt19937ar', 'Seed', seed);
    traj_seeds = randi(seed_stream, [1, 2^31-1], n_gen, 1);
end
traj_seeds = traj_seeds(:);
assert(numel(traj_seeds) == n_gen, 'seed count %d vs n_gen %d', ...
    numel(traj_seeds), n_gen);

t_sample = tic;
z = zeros(D, n_gen);
for ig = 1:n_gen
    z(:, ig) = randn(RandStream('mt19937ar', 'Seed', traj_seeds(ig)), D, 1);
end
sample_s = toc(t_sample);
z0 = z;
nu  = nu_init * ones(nSeg, n_gen);
kap = zeros(nSeg, n_gen);

% Numerical grid.  The default uses the paper's proof variable
% tau=t/(1-t) (Appendix A, Eqs. 115-118).  Since dt/dtau=(1-t)^2, it cancels
% the prescribed-time blow-up in the transformed ODE.  We retain a fine
% t-grid during ordinary FM evolution, then use a uniform tau tail.  The legacy
% adaptive Euler scheme remains available only as a diagnostic ablation.
dt_base = 1 / n_steps;
if strcmp(integration_mode, 'transformed_rk4')
    tearly = (0:dt_base:transform_switch)';
    if tearly(end) < transform_switch
        tearly(end+1,1) = transform_switch;
    end
    tau0 = transform_switch/(1-transform_switch);
    tau1 = t_end/(1-t_end);
    tautail = (tau0:transform_step:tau1)';
    if tautail(end) < tau1
        tautail(end+1,1) = tau1;
    end
    ttail = tautail./(1+tautail);
    times = unique([tearly; ttail(2:end)], 'stable');
    solver_grid = times./(1-times);       % integrate w.r.t. transformed time
    n_actual = numel(times)-1;
    if n_actual > max_steps
        error('Transformed PTZF grid needs %d steps, above max_adaptive_steps=%d.', ...
            n_actual, max_steps);
    end
elseif adaptive_steps
    times = zeros(max_steps + 1, 1);
    n_actual = 0;
    while times(n_actual + 1) < t_end
        t_now = times(n_actual + 1);
        dt_now = min(dt_base, q_max * (1 - t_now)^2 / c_PT);
        dt_now = min(dt_now, t_end - t_now);
        if ~(isfinite(dt_now) && dt_now > 16*eps(max(1, abs(t_now))))
            error('Adaptive PTZF step underflow at t=%.17g.', t_now);
        end
        n_actual = n_actual + 1;
        if n_actual > max_steps
            error(['Adaptive PTZF grid exceeded max_adaptive_steps=%d before ', ...
                'reaching t_end=%g.'], max_steps, t_end);
        end
        times(n_actual + 1) = t_now + dt_now;
    end
    times = times(1:n_actual + 1);
    solver_grid = times;
else
    % Fixed-grid diagnostic uses the same terminal time for a fair comparison.
    times = linspace(0, t_end, n_steps + 1)';
    n_actual = n_steps;
    solver_grid = times;
end
dt_vec = diff(times);
solver_step = diff(solver_grid);

% ---------- PTZF initial values, from the true state at t = 0 ----------
gbar0 = zeros(1, n_gen);
hbar0 = zeros(325 + 4*nSeg, n_gen);
for s = 1:n_gen
    [gv, ~] = eq_residual(z(:,s), nu(:,s), kap(:,s));
    gbar0(s) = 2*gv;                                   % gbar(0) = 2 g(T_0)
    hbar0(:,s) = ineq_values(z(:,s), nu(:,s), kap(:,s));  % hbar_j(0) = h_j(T_0)
end

z_path = [];
if record_path
    z_path = zeros(n_actual+1, D, n_gen);  z_path(1,:,:) = z;
end
u_trace = []; u_trace_t = [];
if trace_on
    u_trace = zeros(n_actual, n_gen, nP, 2, 'single');
    u_trace_t = times(1:end-1);
end
g_trace = nan(n_actual, 1);
rho_g_trace = nan(n_actual, 1);
uS_norm_trace = nan(n_actual, 1);
uA_norm_trace = nan(n_actual, 1);
rows_xy = reshape((0:nP-1)*nF + (1:2)', [], 1);
n_solve = 0; n_active_tot = 0;
slack_count = 0; slack_max = 0; slack_sq_sum = 0;

% ---------- Algorithm 2, PTZF-guided sampling ----------
t_roll = tic;
for step_idx = 1:n_actual
    if strcmp(integration_mode, 'transformed_rk4')
        x0 = solver_grid(step_idx);  hx = solver_step(step_idx);
        t1 = x0/(1+x0);
        [f1z,f1n,f1k,US,dg_] = flow_rhs(z,nu,kap,t1);
        sc1 = (1-t1)^2;  k1z=sc1*f1z; k1n=sc1*f1n; k1k=sc1*f1k;
        x2=x0+0.5*hx; t2=x2/(1+x2);
        [f2z,f2n,f2k] = flow_rhs(z+0.5*hx*k1z,nu+0.5*hx*k1n,kap+0.5*hx*k1k,t2);
        sc2=(1-t2)^2; k2z=sc2*f2z; k2n=sc2*f2n; k2k=sc2*f2k;
        [f3z,f3n,f3k] = flow_rhs(z+0.5*hx*k2z,nu+0.5*hx*k2n,kap+0.5*hx*k2k,t2);
        k3z=sc2*f3z; k3n=sc2*f3n; k3k=sc2*f3k;
        x4=x0+hx; t4=x4/(1+x4);
        [f4z,f4n,f4k] = flow_rhs(z+hx*k3z,nu+hx*k3n,kap+hx*k3k,t4);
        sc4=(1-t4)^2; k4z=sc4*f4z; k4n=sc4*f4n; k4k=sc4*f4k;
        z   = z   + hx*(k1z+2*k2z+2*k3z+k4z)/6;
        nu  = nu  + hx*(k1n+2*k2n+2*k3n+k4n)/6;
        kap = kap + hx*(k1k+2*k2k+2*k3k+k4k)/6;
        t = t1;
    else
        t = times(step_idx);  dt = dt_vec(step_idx);
        [dzdt,dnudt,dkapdt,US,dg_] = flow_rhs(z,nu,kap,t);
        z=z+dt*dzdt; nu=nu+dt*dnudt; kap=kap+dt*dkapdt;
    end
    g_trace(step_idx)=dg_.g; rho_g_trace(step_idx)=dg_.rho_g;
    uS_norm_trace(step_idx)=dg_.uS_norm; uA_norm_trace(step_idx)=dg_.uA_norm;
    if dbg > 0 && step_idx <= dbg
        fprintf(['t=%.3f g=%-10.4g rho_g=%-10.3g |uS|=%-10.3g |uA|=%-10.3g ' ...
            'Npre=%-4d Npost=%-4d Nnew=%-4d\n'], t,dg_.g,dg_.rho_g, ...
            dg_.uS_norm,dg_.uA_norm,dg_.n_pre,dg_.n_post,dg_.n_new);
    end
    if trace_on
        for s=1:n_gen
            Up=(US(rows_xy,s)).*sd(rows_xy);
            u_trace(step_idx,s,:,:)=single(permute(reshape(Up,2,nP),[3 2 1]));
        end
    end
    if record_path, z_path(step_idx+1,:,:) = z; end
end
roll_s = toc(t_roll);

% ---------- outputs ----------
Xgen = z .* sd + mu;
Fgen = permute(reshape(Xgen, nF, nP, n_gen), [2 3 1]);
Pgen = Fgen(:,:,1:2);
gfin = zeros(1, n_gen);  hfin = zeros(1, n_gen);
for s = 1:n_gen
    gfin(s) = eq_residual(z(:,s), nu(:,s), kap(:,s));
    hfin(s) = max(ineq_values(z(:,s), nu(:,s), kap(:,s)));
end

out = struct('points', Pgen, 'features', Fgen, 'state', z, 'mode', 'uniconflow', ...
    'nu', nu, 'kappa', kap, 'g_final', gfin, 'h_max_final', hfin, ...
    'sample_seconds', sample_s, 'rollout_seconds', roll_s, ...
    'terminal_seconds', 0, ...
    'total_seconds_per_traj', (sample_s+roll_s)/n_gen, ...
    'n_qp', n_solve, 'mean_active_rows', n_active_tot/max(n_solve,1), ...
    'slack_active', slack_count, 'slack_active_soft', slack_count, ...
    'slack_max', slack_max, 'slack_l2', sqrt(slack_sq_sum), 'infeasible_hard', 0, ...
    'terminal_corrected', 0, 'terminal_failed', 0, ...
    'terminal_rescued_by_fmincon', 0, 'terminal_audit', struct(), ...
    'segment', segment, 'obstacle', obst, 'geometry', geom, ...
    'obstacle_constraint', obst, 'obstacle_inflation', 0, 'margin', 0, ...
    'normalization_mu', mu, 'normalization_sd', sd, ...
    'n_gen', n_gen, 'u_per_stage', false, ...
    'z0', z0, 'z_path', z_path, 'rollout_seed', seed, ...
    'trajectory_seeds', traj_seeds, ...
    'seed_convention', ['per trajectory: randn(RandStream(''mt19937ar'',' ...
        '''Seed'',s), ' num2str(D) ', 1)'], ...
    'data_seed', cfg.random_seed, ...
    'weight_init_seed', struct_field_default(net, 'weight_init_seed', NaN), ...
    'u_trace', u_trace, 'u_trace_t', u_trace_t, 'u_trace_slack', [], ...
    'u_trace_layout', '(step, sample, point, xy) state guidance, physical units', ...
    'n_rk_steps', n_actual, 't_max', t_end, 'activation_time', 0, ...
    'time_grid', times, 'dt_min', min(dt_vec), 'dt_max', max(dt_vec), ...
    'ptzf_trace', struct('t', times(1:end-1), 'g', g_trace, ...
        'rho_g', rho_g_trace, 'u_state_norm', uS_norm_trace, ...
        'u_action_norm', uA_norm_trace));
out.options = struct('method', 'uniconflow_stage1', 'n_steps', n_steps, ...
    'seed', seed, 'c_g', c_g, 'c_PT', c_PT, 'P_state', P_s, 'P_action', P_a, ...
    'P_slack', P_d_base, 'P_slack_equality', P_d_g, ...
    'P_slack_state', P_d_state, 'P_slack_action', P_d_action, ...
    'nu_max', nu_max, 'kappa_max', kap_max, ...
    'nu_init', nu_init, 'singularity_floor', eps_t, ...
    'adaptive_steps', adaptive_steps, 'adaptive_q_max', q_max, ...
    't_end', t_end, 'max_adaptive_steps', max_steps, ...
    'integration_mode', integration_mode, 'transform_switch', transform_switch, ...
    'transform_step', transform_step, 'terminal_tolerance', terminal_tol, ...
    'max_active_set_passes', max_as, 'record_path', record_path, ...
    'record_u_trace', trace_on);

% =====================================================================
    function [dzdt,dnudt,dkapdt,US,dg1] = flow_rhs(zq,nuq,kapq,tq)
        omq=max(1-tq,realmin);
        decayq=exp(-c_g*tq/omq);
        prateq=c_g/omq^2;
        vq=nn_fwd(net.P,[repmat(tq,1,n_gen);zq]);
        dzdt=zeros(D,n_gen); dnudt=zeros(nSeg,n_gen); dkapdt=zeros(nSeg,n_gen);
        US=zeros(D,n_gen); dg1=struct();
        for sample_idx=1:n_gen
            [usq,uaq,naq,dq]=guidance(zq(:,sample_idx),nuq(:,sample_idx), ...
                kapq(:,sample_idx),vq(:,sample_idx),gbar0(sample_idx)*decayq, ...
                hbar0(:,sample_idx)*decayq,prateq,omq);
            dzdt(:,sample_idx)=vq(:,sample_idx)+usq;
            dnudt(:,sample_idx)=uaq(1:nSeg);
            dkapdt(:,sample_idx)=uaq(nSeg+1:end); US(:,sample_idx)=usq;
            n_solve=n_solve+1; n_active_tot=n_active_tot+naq;
            slack_count=slack_count+dq.slack_count;
            slack_max=max(slack_max,dq.slack_max);
            slack_sq_sum=slack_sq_sum+dq.slack_sq;
            if sample_idx==1
                dg1=dq; dg1.uS_norm=norm(usq); dg1.uA_norm=norm(uaq);
            end
        end
    end

    function [uS, uA, n_act, dg_] = guidance(zs, nus, kaps, vs, gbar, hbar, prate, om)
        [gv, gr] = eq_residual(zs, nus, kaps);          % scalar + dense gradient
        [hv, HI, HG] = ineq_rows(zs, nus, kaps);        % values, indices, grads
        gam = @(a) c_PT * a / om^2;
        % rho = eta' * v_ext - gamma(bar - val) - dbar,  v_ext = [v; 0]
        rho_g = gr(1:D)'*vs - gam(gbar - gv) + prate*gbar;
        nH = numel(hv);
        rho_h = zeros(nH,1);
        for r = 1:nH
            gv_dot = 0;
            for c = 1:size(HI,2)
                idx = HI(r,c);
                if idx > 0 && idx <= D, gv_dot = gv_dot + HG(r,c)*vs(idx); end
            end
            rho_h(r) = gv_dot - gam(hbar(r) - hv(r)) + prate*hbar(r);
        end
        act_h = rho_h > 0;
        act_g = rho_g > 0;
        n_pre = nnz(act_h) + double(act_g);
        uS = zeros(D,1);  uA = zeros(2*nSeg,1);
        n_act = n_pre;
        dg_ = struct('g', gv, 'rho_g', rho_g, 'n_pre', n_pre, 'n_post', 0, ...
            'n_new', 0, 'passes', 0, 'slack_count', 0, 'slack_max', 0, ...
            'slack_sq', 0);
        if n_pre == 0, return; end
        % OUTER ACTIVE-SET LOOP.  A single pass freezes {rho_j > 0} and solves the
        % equality form, but the correction it produces can push rows that were
        % inactive across their own boundary, so that solve is not a KKT point of
        % the QP the paper writes (Eq. 49-51).  Here the newly violated rows are
        % added and the whole system is re-solved.  Rows are only ever ADDED,
        % never dropped, so the active set grows monotonically: the loop cannot
        % cycle and terminates in at most nH passes.  Each solve stays O(d) via
        % the block-diagonal + rank-one structure, so a few passes cost far less
        % than the one network evaluation the step already pays for.
        x = zeros(dAug,1);  post = rho_h;  post_g = rho_g;
        for pass = 1:max_as
            dg_.passes = pass;
            x = solve_active(act_h, act_g, rho_h, rho_g, HI, HG, gr);
            post = rho_h;
            for r2 = 1:nH
                idx = HI(r2,:);  gr2 = HG(r2,:);
                for c2 = 1:numel(idx)
                    if idx(c2) > 0, post(r2) = post(r2) + gr2(c2)*x(idx(c2)); end
                end
            end
            post_g = rho_g + gr'*x;
            new_h = ~act_h & (post > 0);
            new_g = ~act_g && (post_g > 0);
            % n_new keeps its old meaning: how inconsistent ONE frozen pass was.
            if pass == 1, dg_.n_new = nnz(new_h) + double(new_g); end
            if ~any(new_h) && ~new_g, break; end
            act_h = act_h | new_h;
            act_g = act_g || new_g;
        end
        uS = x(1:D);  uA = x(D+1:end);
        n_act = nnz(act_h) + double(act_g);
        dg_.n_post = nnz(post > 0) + double(post_g > 0);
        sl=[post(act_h); post_g*double(act_g)];
        if ~act_g, sl=post(act_h); end
        dg_.slack_count=nnz(abs(sl)>terminal_tol);
        if ~isempty(sl), dg_.slack_max=max(abs(sl)); dg_.slack_sq=sum(sl.^2); end
    end

    % One equality-form solve over a GIVEN active set.  Eq. (49) with the slack
    % eliminated leaves (Pu + eta' Pd eta) u = -eta' Pd rho; only active rows
    % contribute, every safety row touches one point's two position coordinates
    % and every action row a single action coordinate, so the matrix is block
    % diagonal apart from the one dense aggregated-equality row, which enters by
    % Sherman-Morrison.
    function x = solve_active(act_h, act_g, rho_h, rho_g, HI, HG, gr)
        dg = [P_s*ones(D,1); P_a*ones(2*nSeg,1)];
        blk = zeros(2,2,nP);                             % 2x2 blocks on positions
        for k = 1:nP, blk(:,:,k) = P_s*eye(2); end
        for r = find(act_h)'
            idx = HI(r,:);  gr_ = HG(r,:);
            if idx(2) > 0 && idx(2) == idx(1)+1 && idx(1) <= D
                k = (idx(1)-1)/nF + 1;                   % position pair
                wr=row_slack_weight(r);
                blk(:,:,k) = blk(:,:,k) + wr*(gr_(:)*gr_(:)');
            else
                wr=row_slack_weight(r);
                dg(idx(1)) = dg(idx(1)) + wr*gr_(1)^2;  % single action coord
            end
        end
        rhs = zeros(dAug,1);
        for r = find(act_h)'
            idx = HI(r,:);  gr_ = HG(r,:);
            for c = 1:numel(idx)
                if idx(c) > 0
                    rhs(idx(c)) = rhs(idx(c)) - row_slack_weight(r)*gr_(c)*rho_h(r);
                end
            end
        end
        if act_g, rhs = rhs - P_d_g*gr*rho_g; end
        x = dinv_apply(rhs, dg, blk);
        if act_g                                          % Sherman-Morrison
            w = dinv_apply(gr, dg, blk);
            x = x - P_d_g*(gr'*x)/(1 + P_d_g*(gr'*w)) * w;
        end
    end

    function wr=row_slack_weight(r)
        if r <= nP*(nObs+2), wr=P_d_state; else, wr=P_d_action; end
    end

    % Apply the inverse of the block-diagonal part: 2x2 blocks on each point's
    % position pair, plain diagonal everywhere else.
    function y = dinv_apply(b, dg, blk)
        y = b ./ dg;
        for k = 1:nP
            id = ix(k);
            y(id) = blk(:,:,k) \ b(id);
        end
    end

    % ---- aggregated equality and its dense gradient ----
    function [gv, gr] = eq_residual(zs, nus, kaps)
        gr = zeros(dAug,1);
        gv = 0;
        P = zeros(2,nP);  T = zeros(2,nP);
        for k = 1:nP
            P(:,k) = zs(ix(k)).*sd(ix(k)) + mu(ix(k));
            T(:,k) = zs(it(k)).*sd(it(k)) + mu(it(k));
        end
        for k = 1:nSeg
            e = P(:,k+1) - P(:,k) - nus(k)*T(:,k);       % kinodynamic residual
            gv = gv + e'*e;
            gr(ix(k+1)) = gr(ix(k+1)) + 2*e.*sd(ix(k+1));
            gr(ix(k))   = gr(ix(k))   - 2*e.*sd(ix(k));
            gr(it(k))   = gr(it(k))   - 2*nus(k)*e.*sd(it(k));
            gr(inu(k))  = gr(inu(k))  - 2*(e'*T(:,k));
            c = cos(kaps(k)); sn = sin(kaps(k));
            Rt = [c*T(1,k) - sn*T(2,k); sn*T(1,k) + c*T(2,k)];
            q  = T(:,k+1) - Rt;                          % tangent transition
            gv = gv + q'*q;
            gr(it(k+1)) = gr(it(k+1)) + 2*q.*sd(it(k+1));
            dRt_dT = [c -sn; sn c];
            gr(it(k))  = gr(it(k))  - 2*(dRt_dT'*q).*sd(it(k));
            dRt_dk = [-sn*T(1,k) - c*T(2,k); c*T(1,k) - sn*T(2,k)];
            gr(ika(k)) = gr(ika(k)) - 2*(q'*dRt_dk);
        end
        for k = 1:nP                                     % unit tangent
            n2 = T(:,k)'*T(:,k) - 1;
            gv = gv + n2^2;
            gr(it(k)) = gr(it(k)) + 4*n2*T(:,k).*sd(it(k));
        end
    end

    % ---- inequality values only (for the PTZF initial values) ----
    function hv = ineq_values(zs, nus, kaps)
        hv = ineq_rows(zs, nus, kaps);
    end

    % ---- inequality rows: values, the coordinate indices each touches, grads ----
    function [hv, HI, HG] = ineq_rows(zs, nus, kaps)
        nH = nP*(nObs+2) + 4*nSeg;
        hv = zeros(nH,1);  HI = zeros(nH,2);  HG = zeros(nH,2);
        r = 0;
        for k = 1:nP
            id = ix(k);
            p  = zs(id).*sd(id) + mu(id);
            for jo = 1:nObs
                [bv, gp] = obstacle_level_and_gradient(p, obst, jo);
                r = r+1;  hv(r) = -bv;                    % paper: h <= 0 is safe
                HI(r,:) = id;  HG(r,:) = (-gp(:).*sd(id))';
            end
            for bi = 1:2
                [bv, gp] = evaluate_track_implicit_field(geom.implicit_fields, bi, p);
                r = r+1;  hv(r) = -bv;
                HI(r,:) = id;  HG(r,:) = (-gp(:).*sd(id))';
            end
        end
        for k = 1:nSeg                                    % box bounds on actions
            r=r+1; hv(r) = -nus(k);          HI(r,:)=[inu(k) 0]; HG(r,:)=[-1 0];
            r=r+1; hv(r) =  nus(k)-nu_max;   HI(r,:)=[inu(k) 0]; HG(r,:)=[ 1 0];
            r=r+1; hv(r) =  kaps(k)-kap_max; HI(r,:)=[ika(k) 0]; HG(r,:)=[ 1 0];
            r=r+1; hv(r) = -kaps(k)-kap_max; HI(r,:)=[ika(k) 0]; HG(r,:)=[-1 0];
        end
    end
end

% =====================================================================
function y = nn_fwd(P, x)
sil = @(z) z./(1+exp(-z));
a1 = sil(P{1}*x  + P{5});
a2 = sil(P{2}*a1 + P{6});
a3 = sil(P{3}*a2 + P{7});
y  = P{4}*a3 + P{8};
end
