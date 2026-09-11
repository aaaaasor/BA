function out = pcfm_nn_rollout(net, n_gen, opts)
%PCFM_NN_ROLLOUT  Physics-Constrained Flow Matching on the racing scenario.
%
% Utkarsh et al., "Physics-Constrained Flow Matching: Sampling Generative
% Models with Hard Constraints", NeurIPS 2025 (arXiv:2506.04171), Algorithm 1.
%
% Deliberately independent of safeflow_nn_rollout: PCFM shares no machinery
% with the CBF-QP path -- no class-K alpha, no blow-up gain, no gamma(t), no
% slack variable.  The scene setup below is duplicated rather than shared so
% that later edits to the CBF rollout cannot silently change this baseline.
%
% ---------------------------------------------------------------------------
% One step tau -> tau' of Algorithm 1, as the official implementation writes it:
%
%   1. v      = v_theta(z_t, t)
%   2. z1     = z_t + (1-t) v                     forward shooting, one Euler
%                                                 step reusing v -- no extra
%                                                 network evaluation
%   3. z1proj = project(z1)                       one linearised projection
%   4. z_t'   = (1-t') z0 + t' z1proj             OT displacement interpolant,
%                                                 Eq. (5) / Prop. 3.1
%
% Step 4 does not reference z_t.  That is not a missing accumulation term: the
% paper reconstructs the state from the initial noise and the current projected
% terminal at every step, and z1proj already depends on z_t through v.  The
% straight-line path is justified in the small-step limit by Prop. 3.1.
%
% ---------------------------------------------------------------------------
% INEQUALITY CONSTRAINTS ARE AN EXTENSION, NOT THE PAPER'S METHOD.  PCFM is
% stated for equalities h(u) = 0, and its conclusion lists inequalities as
% future work.  Safety here is b(p) >= 0, so the Gauss-Newton step of Eq. (4)
% is replaced by the minimum-norm step over the linearised half-spaces,
%
%     min ||dz||^2   s.t.   b_i(p) + grad b_i(p)' dz >= 0,  i = 1..5,
%
% solved exactly by minnorm_halfspaces, which enumerates active sets.  All five
% rows are handed over every time; a satisfied row has c_i = -b_i <= 0, so it
% costs nothing and only becomes active if the correction would otherwise push
% through it.  Pre-filtering to b_i < 0 would miss exactly that case, and
% applying the equality formula to near-active rows would pull safe points back
% onto the boundary.
%
% The projection runs in the model's own standardised space, which is where the
% paper's ||u - u_hat||^2 lives.  Since b is defined on physical coordinates,
% the Jacobian carries the chain rule  grad_z b = sd .* grad_p b.  Minimising
% in physical space instead would be a different problem, because sd differs
% per dimension.
%
% lambda = 0 (no relaxed correction, Eq. 6).  That is the official default
% sampling setting; with N = 100 steps the paper's ablation also shows the
% relaxed correction contributes little.  It is not required by the method.
%
% The final projection of Eq. (7) is NOT optional -- it is what makes the
% constraint hard -- so unlike the SafeDiffuser variants, which have no such
% step in their paper, this baseline always runs it.  It is solved by sequential
% linearisation against a FIXED reference point, so each iterate is the nearest
% feasible point to the original state rather than the end of a chain of local
% steps.  The feasible set (the intersection of three ellipse exteriors with the
% track) is non-convex, so the result is a KKT point of Eq. (7), not certified
% globally nearest -- the same caveat the CBF path's terminal filter carries.

if nargin < 2 || isempty(n_gen), n_gen = 100; end
if nargin < 3, opts = struct(); end
gf = @(f,d) struct_field_default(opts, f, d);

n_steps  = gf('n_steps', 100);
% Algorithm 1 uses dtau = 1/N and ends at tau = 1, and its step 4 is an exact
% interpolation with no truncation error, so there is no reason to stop early.
% Stopping at 0.996 (what the CBF baselines use, where it limits the RK4
% truncation) would leave the output as 0.004*z0 + 0.996*z1proj -- a 0.4%%
% admixture of pure noise applied AFTER the projection, which can push points
% back out of the safe set.
t_max    = gf('t_max', 1.0);
seed     = gf('seed', 11);
% Paper has no margin and no obstacle inflation; b is the exact level set.
margin   = gf('margin', 0.0);
obs_infl = gf('obstacle_inflation', 0);
% Eq. (7): iterate the same inequality projection until feasible.
term_tol = gf('terminal_tol', 1e-8);
term_it  = gf('terminal_max_iter', 200);
% qp_backend: 'closed' = exact active-set enumeration (minnorm_halfspaces),
% 'quadprog' = same program through MATLAB's generic QP solver.
qp_backend = gf('qp_backend', 'closed');
assert(ismember(qp_backend, {'closed','quadprog'}), ...
    'qp_backend must be closed or quadprog');
lambda   = gf('lambda', 0.0);
assert(lambda == 0, ['lambda > 0 (the relaxed correction of Eq. 6) is not ' ...
    'implemented; the official default is 0.']);
record_path   = gf('record_path', false);
trace_on      = gf('record_u_trace', true);

% ---------- scene ----------
cfg = get_config();
rng(cfg.random_seed);
[~, segment] = scenario_training_points(cfg, 65, cfg.n_train);
obst_phys = configure_racing_obstacles(segment, cfg.obstacle);
obst = obst_phys;
if obs_infl ~= 0, obst.semi_axes = obst.semi_axes + obs_infl; end
geom = build_track_boundary_geometry(segment, ...
    struct_field_default(cfg.track_boundary, 'n_spline_points', 400));
nObs = size(obst.centers, 2);
nRow = nObs + 2;                      % 3 obstacles + left/right boundary

sd = net.sd_d;  mu = net.mu_d;
D  = numel(mu);
nP = struct_field_default(net, 'n_points', 65);
nF = struct_field_default(net, 'n_features_per_point', D/nP);
assert(nF == 4 && D == nP*nF, 'state must be n_points x 4 features');

% ---------- seeds and initial noise ----------
traj_seeds = gf('trajectory_seeds', []);
if isempty(traj_seeds)
    sstream = RandStream('mt19937ar', 'Seed', seed);
    traj_seeds = randi(sstream, [1, 2^31-1], n_gen, 1);
end
traj_seeds = traj_seeds(:);
assert(numel(traj_seeds) == n_gen, 'seed count %d does not match n_gen %d', ...
    numel(traj_seeds), n_gen);

t_sample = tic;
z = zeros(D, n_gen);
for ig = 1:n_gen
    z(:, ig) = randn(RandStream('mt19937ar', 'Seed', traj_seeds(ig)), D, 1);
end
sample_s = toc(t_sample);
z0 = z;

dt = t_max / n_steps;
z_path = [];
if record_path
    z_path = zeros(n_steps+1, D, n_gen);
    z_path(1,:,:) = z;
end
u_trace = []; u_trace_t = [];
if trace_on
    u_trace   = zeros(n_steps, n_gen, nP, 2, 'single');
    u_trace_t = zeros(n_steps, 1);
end

n_proj = 0;            % projections solved (one per point per step)
n_infeas = 0;          % of those, the ones with no feasible half-space point
n_moved = 0;           % projections that returned a non-zero correction
rows_xy = reshape((0:nP-1)*nF + (1:2)', [], 1);

% ---------- Algorithm 1 ----------
t_roll = tic;
for k = 0:n_steps-1
    t  = k*dt;
    tp = t + dt;
    v  = nn_fwd(net.P, [repmat(t,1,n_gen); z]);
    z1 = z + (1 - t) * v;                       % shooting, reusing v
    z1p = z1;
    for s = 1:n_gen
        for ip = 1:nP
            id = nF*(ip-1) + (1:2);
            [dz, moved] = project_point(z1(id,s), sd(id), mu(id));
            z1p(id,s) = z1(id,s) + dz;
            n_proj = n_proj + 1;
            n_moved = n_moved + moved;
        end
    end
    z_new = (1 - tp) * z0 + tp * z1p;
    if trace_on
        % Effective PCFM correction velocity, in physical units, so it is
        % directly comparable to the CBF path's u_trace.
        dzs = (z_new - z) / dt - v;
        Up  = dzs .* sd;
        u_trace(k+1,:,:,:) = single(permute( ...
            reshape(Up(rows_xy,:), 2, nP, n_gen), [3 2 1]));
        u_trace_t(k+1) = t;
    end
    z = z_new;
    if record_path, z_path(k+2,:,:) = z; end
end
roll_s = toc(t_roll);

% ---------- Eq. (7): final projection, iterated to feasibility ----------
t_term = tic;
term_moved = 0; term_fail = 0;
for s = 1:n_gen
    for ip = 1:nP
        id = nF*(ip-1) + (1:2);
        z_ref = z(id,s);              % Eq. (7) measures distance from HERE
        zk = z_ref;
        [b, ~] = rows_at(zk, sd(id), mu(id));
        if min(b) >= -term_tol, continue; end
        moved_any = false;
        for it = 1:term_it
            [b, G] = rows_at(zk, sd(id), mu(id));
            if min(b) >= -term_tol, break; end
            % Sequential linearisation of  min ||z - z_ref||^2  s.t. b(z) >= 0.
            % With w = z - z_ref and e = z_ref - zk, the linearised constraint
            % b + G*(w + e) >= 0 becomes G*w >= -b - G*e, so the reference stays
            % fixed across iterations.  Stepping from the current zk instead
            % would minimise the sum of the increments, which is a different
            % problem and lands somewhere other than the nearest feasible point.
            e  = z_ref - zk;
            w  = minnorm_halfspaces(G, -b - G*e, qp_backend);
            zk_new = z_ref + w;
            if norm(zk_new - zk) < 1e-15, break; end
            zk = zk_new;
            moved_any = true;
        end
        b = rows_at(zk, sd(id), mu(id));
        if min(b) < -term_tol, term_fail = term_fail + 1; end
        term_moved = term_moved + moved_any;
        z(id,s) = zk;
    end
end
term_s = toc(t_term);
if record_path, z_path(end,:,:) = z; end

% ---------- outputs ----------
Xgen = z .* sd + mu;
Fgen = permute(reshape(Xgen, nF, nP, n_gen), [2 3 1]);
Pgen = Fgen(:,:,1:2);

out = struct('points', Pgen, 'features', Fgen, 'state', z, 'mode', 'pcfm', ...
    'sample_seconds', sample_s, 'rollout_seconds', roll_s, ...
    'terminal_seconds', term_s, ...
    'total_seconds_per_traj', (sample_s+roll_s+term_s)/n_gen, ...
    'n_qp', n_proj, 'slack_active', n_infeas, 'projections_moved', n_moved, ...
    'terminal_corrected', term_moved, 'terminal_failed', term_fail, ...
    'terminal_rescued_by_fmincon', 0, 'terminal_audit', struct(), ...
    'slack_active_soft', 0, 'infeasible_hard', n_infeas, ...
    'segment', segment, 'obstacle', obst_phys, 'geometry', geom, ...
    'obstacle_constraint', obst, 'obstacle_inflation', obs_infl, ...
    'margin', margin, 'n_gen', n_gen, 'u_per_stage', false, ...
    'z0', z0, 'z_path', z_path, 'rollout_seed', seed, ...
    'trajectory_seeds', traj_seeds, ...
    'seed_convention', ['per trajectory: randn(RandStream(''mt19937ar'',' ...
        '''Seed'',s), ' num2str(D) ', 1); the seed table is derived from ' ...
        'rollout_seed by randi'], ...
    'data_seed', cfg.random_seed, ...
    'weight_init_seed', struct_field_default(net, 'weight_init_seed', NaN), ...
    'u_trace', u_trace, 'u_trace_t', u_trace_t, 'u_trace_slack', [], ...
    'u_trace_layout', ['(step, sample, trajectory point, xy) effective PCFM ' ...
        'correction velocity in physical units'], ...
    'n_rk_steps', n_steps, 't_max', t_max, 'activation_time', 0);
out.options = struct('method', 'pcfm', 'n_steps', n_steps, 't_max', t_max, ...
    'seed', seed, 'margin', margin, 'obstacle_inflation', obs_infl, ...
    'lambda', lambda, 'terminal_tol', term_tol, ...
    'terminal_max_iter', term_it, ...
    'record_path', record_path, 'record_u_trace', trace_on, ...
    'qp_backend', qp_backend);

% =====================================================================
    function [dz, moved] = project_point(zk, sdk, muk)
        [b, G] = rows_at(zk, sdk, muk);
        if min(b) >= 0, dz = [0;0]; moved = 0; return; end
        dz = minnorm_halfspaces(G, -b, qp_backend);
        moved = double(norm(dz) > 0);
        % minnorm_halfspaces returns pinv(G)*max(c,0) when no subset of the
        % half-spaces is feasible, and that solution satisfies nothing in
        % particular.  Detect it here rather than silently reporting zero.
        if any(G*dz < -b - 1e-9), n_infeas = n_infeas + 1; end
    end

    % b_i and their standardised-space gradients for one trajectory point.
    % grad_z b = sd .* grad_p b by the chain rule, because b lives on physical
    % coordinates while the projection metric lives on the standardised state.
    function [b, G] = rows_at(zk, sdk, muk)
        p = zk .* sdk + muk;
        b = zeros(nRow,1);  G = zeros(nRow,2);
        for jo = 1:nObs
            [bv, gp] = obstacle_level_and_gradient(p, obst, jo);
            b(jo) = bv;  G(jo,:) = (sdk(:) .* gp(:)).';
        end
        for bi = 1:2
            [hv, gp] = evaluate_track_implicit_field(geom.implicit_fields, bi, p);
            b(nObs+bi) = hv - margin;  G(nObs+bi,:) = (sdk(:) .* gp(:)).';
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
