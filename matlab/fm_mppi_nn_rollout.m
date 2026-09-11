function out = fm_mppi_nn_rollout(net, n_gen, opts)
%FM_MPPI_NN_ROLLOUT  FM-warm-started MPPI baseline for the racing scenario.
%
% UniConFlow describes FM-MPPI only as trajectories sampled from a pretrained
% FM model and used as informed priors that warm-start MPPI.  It publishes no
% action parameterisation, MPPI pseudocode, costs, sample count, covariance,
% temperature, or iteration count.  This file therefore reproduces that stated
% pipeline, while recording every missing choice as an option:
%
%   1. Sample a complete 65x4 [x,y,tx,ty] trajectory with the frozen FM model.
%   2. Convert its chords to geometric controls a_k=(nu_k,kappa_k).  The
%      conversion replays the FM POSITIONS exactly -- no clipping, so the warm
%      start really is the FM trajectory:
%          p_{k+1}=p_k+nu_k*t_k,   t_{k+1}=R(kappa_k)*t_k.
%      The reconstructed tangents are chord directions, which are NOT the FM's
%      own tangent features; the cost pulls them toward those separately.
%   3. Use those controls as the mean of an MPPI proposal distribution.
%   4. Perturb controls, roll out the known geometric dynamics, score
%      safety/reference/smoothness/action-bound costs, and apply the
%      path-integral exponential-weighted noise update.
%
% This is deliberately independent of UniConFlow's PTZF/QP/slack and of the
% SafeFlow/PCFM terminal projections, so it has no hard state-safety guarantee
% and no terminal safety filter.  The canonical variant below uses hard action
% bounds; the reconstructed variant retains the earlier soft-bound ablation.

if nargin < 2 || isempty(n_gen), n_gen = 100; end
if nargin < 3, opts = struct(); end
gf = @(f,d) struct_field_default(opts, f, d);

% FM prior.  These defaults match the project's unconstrained NN-FM baseline.
fm_steps = gf('fm_n_steps', 100);
fm_tmax  = gf('fm_t_max', 0.996);
seed     = gf('seed', 11);

% MPPI choices not published by UniConFlow.  'canonical' follows Algorithm 2
% of Williams et al., Information-Theoretic MPC (ICRA 2017): fixed lambda,
% lambda*u'*Sigma^{-1}*epsilon in the rollout cost, and hard action bounds.
% It is retained as a diagnostic because lambda and Sigma are not published for
% the UniConFlow baseline and the literal combination is ill-scaled here.
% 'reconstructed' is the default FM-warm-started, ESS-controlled, soft-bound
% weighted-sampling optimizer.  It deliberately omits the likelihood-ratio
% cross term; call it MPPI-like rather than an exact R->0 canonical limit.
variant  = lower(char(gf('mppi_variant', 'reconstructed')));
assert(ismember(variant, {'canonical','reconstructed'}), ...
    'mppi_variant must be canonical or reconstructed');
canonical = strcmp(variant, 'canonical');
n_roll   = gf('n_rollouts', 512);
n_iter   = gf('n_iterations', 15);
temp     = gf('temperature', 1e4);
% Independent segment perturbations accumulate over the 64-step open-loop
% rollout.  A local sweep on the archived FM prior found the original
% (0.012, 0.20) proposal destructive; these smaller defaults reduce cost in
% both variants.  They remain project choices because UniConFlow publishes no
% covariance for its FM-MPPI baseline.
sig_nu   = gf('sigma_nu', 0.0004);
sig_kap  = gf('sigma_kappa', 0.005);
upd      = gf('update_rate', 1.0);
% Adaptive temperature.  With a fixed lambda the exponential weights collapse
% onto a single rollout as soon as the cost spread exceeds it -- measured ESS
% 1.5 of 128 at lambda = 1e4 -- which turns the path-integral average into a
% plain argmin and throws away the other samples.  Bisecting lambda so the
% effective sample size hits a target fraction of K is standard MPPI practice;
% UniConFlow publishes no temperature for this baseline either way.
adapt_T  = gf('adaptive_temperature', ~canonical);
ess_frac = gf('ess_target_fraction', 0.10);
antithetic = gf('antithetic_noise', true);
% The exact information-theoretic cross term depends on the controller's
% control-cost matrix, which UniConFlow does not publish for this baseline.
% Keep it available for an ablation, but default to the widely used
% exponential-cost/noise-update form instead of inventing that missing matrix.
cross_on = gf('control_cross_term', canonical);
hard_bounds = gf('hard_control_bounds', canonical);

% Bounds measured from the training data, with a small support margin.
% A small positive lower bound prevents the finite-horizon planner from
% collapsing segments to zero length and is still below the training minimum
% (0.0112).  Set it to zero explicitly for a stop-capable ablation.
nu_min   = gf('nu_min', 0.005);
nu_max   = gf('nu_max', 0.065);
kap_max  = gf('kappa_max', 1.15);

% Cost weights.  Obstacle and boundary fields have different numerical scales,
% so their squared penetration weights are deliberately separate.
w_ref    = gf('reference_weight', 1.0);
w_tan    = gf('tangent_weight', 0.10);
w_term   = gf('terminal_weight', 10.0);
w_act    = gf('action_prior_weight', 0.05);
w_smooth = gf('smoothness_weight', 0.10);
w_obs    = gf('obstacle_weight', 2e3);
w_bnd    = gf('boundary_weight', 2e5);
w_hit    = gf('violation_count_weight', 100.0);
safety_tol = gf('safety_tolerance', 1e-8);
% Weight on action-bound violation.  Not published by UniConFlow; set so that a
% segment one box-width outside its bound costs about as much as a handful of
% state violations, i.e. comparable to w_hit.
w_box    = gf('action_bound_weight', 1e3);

assert(n_roll >= 2 && n_iter >= 1 && temp > 0, ...
    'MPPI needs n_rollouts>=2, n_iterations>=1, and temperature>0.');
assert(sig_nu > 0 && sig_kap > 0 && upd > 0 && upd <= 1, ...
    'Noise scales must be positive and update_rate must lie in (0,1].');
assert(nu_min >= 0 && nu_max > nu_min && kap_max > 0, ...
    'Invalid geometric-action bounds.');

% Use the same per-trajectory FM noise seeds as the other baselines.
traj_seeds = gf('trajectory_seeds', []);
if isempty(traj_seeds)
    ss = RandStream('mt19937ar', 'Seed', seed);
    traj_seeds = randi(ss, [1, 2^31-1], n_gen, 1);
end
traj_seeds = traj_seeds(:);
assert(numel(traj_seeds) == n_gen, ...
    'trajectory_seeds must contain exactly n_gen entries.');

fm_opts = struct('trajectory_seeds', traj_seeds, 'n_steps', fm_steps, ...
    't_max', fm_tmax, 'terminal_filter', false, 'record_path', false, ...
    'record_u_trace', false);
prior = safeflow_nn_rollout(net, 'fm', n_gen, fm_opts);

cfg = get_config();
rng(cfg.random_seed);
[~, segment] = scenario_training_points(cfg, 65, cfg.n_train);
obst = configure_racing_obstacles(segment, cfg.obstacle);
geom = build_track_boundary_geometry(segment, ...
    struct_field_default(cfg.track_boundary, 'n_spline_points', 400));

mu = net.mu_d; sd = net.sd_d;
D = numel(mu);
nP = struct_field_default(net, 'n_points', 65);
nF = struct_field_default(net, 'n_features_per_point', D/nP);
assert(nF == 4 && D == nP*nF, ...
    'FM-MPPI expects 65 points with [x,y,tx,ty] features.');
nSeg = nP - 1;
assert(size(prior.features,1) == nP && size(prior.features,3) == nF);

% A second deterministic stream is derived for MPPI perturbations.  This keeps
% a one-trajectory replay independent of its position in a batch.
mppi_seeds = mod(double(traj_seeds) + 104729, 2^31-2) + 1;

Pgen = zeros(nP,n_gen,2);
Tgen = zeros(nP,n_gen,2);
Nu = zeros(nSeg,n_gen);
Kap = zeros(nSeg,n_gen);
cost_initial = zeros(n_gen,1);
cost_final = zeros(n_gen,1);
ess_final = zeros(n_gen,1);
ess_hist = zeros(n_gen,n_iter);
lam_hist = zeros(n_gen,n_iter);
safe_rollouts_final = zeros(n_gen,1);

t_mppi = tic;
for is = 1:n_gen
    Fref = squeeze(prior.features(:,is,:));
    Pref = Fref(:,1:2).';
    Tref = Fref(:,3:4).';
    [Ubar, p0, t0] = controls_from_fm(Pref, Tref);
    Uref = Ubar;
    % An infeasible FM-derived sequence cannot simultaneously replay the FM
    % positions exactly and satisfy the action box.  Canonical MPPI uses the
    % projected sequence as its informed initial mean; the raw FM sequence is
    % retained as the reference prior in the state/action costs.
    if hard_bounds, Ubar = project_controls(Ubar); end
    stream = RandStream('mt19937ar', 'Seed', mppi_seeds(is));

    [J0, ~, ~] = rollout_cost(Ubar, p0, t0, Pref, Tref, Uref);
    cost_initial(is) = J0;
    for it = 1:n_iter
        E = sample_noise(stream);
        Us = Ubar + E;
        if hard_bounds
            Us = project_controls(Us);
            Eused = Us - Ubar;
        else
            Eused = E;
        end

        [J, ~, safe_mask] = rollout_cost(Us, p0, t0, Pref, Tref, Uref);
        if cross_on
            cross = squeeze(sum(sum( ...
                (Ubar(1,:)./sig_nu^2).*Eused(1,:,:) + ...
                (Ubar(2,:)./sig_kap^2).*Eused(2,:,:), 2), 1));
            J = J + temp*cross(:).';
        end
        [w, lam_used] = path_integral_weights(J);
        lam_hist(is,it) = lam_used;
        du = sum(Eused .* reshape(w,1,1,[]), 3);
        Ubar = Ubar + upd*du;
        if hard_bounds, Ubar = project_controls(Ubar); end
        ess_hist(is,it) = 1/max(sum(w.^2), eps);
        ess_final(is) = ess_hist(is,it);
        safe_rollouts_final(is) = nnz(safe_mask);
    end

    [Jf, Xf, ~] = rollout_cost(Ubar, p0, t0, Pref, Tref, Uref);
    cost_final(is) = Jf;
    Pgen(:,is,:) = permute(Xf.P, [2 3 1]);
    Tgen(:,is,:) = permute(Xf.T, [2 3 1]);
    Nu(:,is) = Ubar(1,:).';
    Kap(:,is) = Ubar(2,:).';
end
mppi_s = toc(t_mppi);

Fgen = cat(3, Pgen, Tgen);
Xgen = reshape(permute(Fgen,[3 1 2]), D, n_gen);
z = (Xgen-mu)./sd;

total_s = prior.sample_seconds + prior.rollout_seconds + mppi_s;
out = struct('points',Pgen,'features',Fgen,'state',z,'mode','fm_mppi', ...
    'nu',Nu,'kappa',Kap,'fm_prior_points',prior.points, ...
    'fm_prior_features',prior.features,'fm_prior_state',prior.state, ...
    'cost_initial',cost_initial,'cost_final',cost_final, ...
    'effective_sample_size_final',ess_final, ...
    'effective_sample_size_history',ess_hist,'temperature_history',lam_hist, ...
    'safe_rollouts_final',safe_rollouts_final, ... % backward-compatible name
    'sampled_safe_rollouts_last',safe_rollouts_final, ...
    'sample_seconds',prior.sample_seconds, ...
    'rollout_seconds',prior.rollout_seconds+mppi_s, ...
    'mppi_seconds',mppi_s,'terminal_seconds',0, ...
    'total_seconds_per_traj',total_s/n_gen, ...
    'n_qp',0,'slack_active',0,'slack_active_soft',0, ...
    'infeasible_hard',0,'terminal_corrected',0,'terminal_failed',0, ...
    'terminal_rescued_by_fmincon',0,'terminal_audit',struct(), ...
    'segment',segment,'obstacle',obst,'geometry',geom, ...
    'obstacle_constraint',obst,'obstacle_inflation',0,'margin',0, ...
    'n_gen',n_gen,'u_per_stage',false,'z0',prior.z0,'z_path',[], ...
    'rollout_seed',seed,'trajectory_seeds',traj_seeds, ...
    'mppi_seeds',mppi_seeds,'data_seed',cfg.random_seed, ...
    'weight_init_seed',struct_field_default(net,'weight_init_seed',NaN), ...
    'seed_convention',['FM uses the stored per-trajectory Gaussian seed; ' ...
        'MPPI uses mod(seed+104729,2^31-2)+1, independently per trajectory'], ...
    'u_trace',[],'u_trace_t',[],'u_trace_slack',[], ...
    'u_trace_layout','not applicable: MPPI refines action sequences', ...
    'n_rk_steps',fm_steps,'t_max',fm_tmax,'activation_time',NaN);
out.options = struct('method','fm_mppi','fm_n_steps',fm_steps, ...
    'fm_t_max',fm_tmax,'seed',seed,'n_rollouts',n_roll, ...
    'n_iterations',n_iter,'temperature',temp,'sigma_nu',sig_nu, ...
    'sigma_kappa',sig_kap,'update_rate',upd, ...
    'mppi_variant',variant,'antithetic_noise',antithetic, ...
    'control_cross_term',cross_on,'hard_control_bounds',hard_bounds, ...
    'adaptive_temperature',adapt_T,'ess_target_fraction',ess_frac, ...
    'nu_min',nu_min,'nu_max',nu_max,'kappa_max',kap_max, ...
    'reference_weight',w_ref,'tangent_weight',w_tan, ...
    'terminal_weight',w_term,'action_prior_weight',w_act, ...
    'smoothness_weight',w_smooth,'obstacle_weight',w_obs, ...
    'boundary_weight',w_bnd,'violation_count_weight',w_hit, ...
    'action_bound_weight',w_box,'safety_tolerance',safety_tol, ...
    'trajectory_seeds',traj_seeds);

% ---------------------------------------------------------------------
    function [w, lam] = path_integral_weights(J)
        % Softmax over -J/lam.  With adapt_T the temperature is bisected so the
        % effective sample size 1/sum(w^2) reaches ess_frac*K; ESS rises
        % monotonically with lam, so bisection is well posed.  Degenerate cases
        % (all costs equal, or no lam large enough) fall back to the uniform and
        % argmin weights respectively.
        d = J - min(J);
        if ~any(d > 0), w = ones(size(J))/numel(J); lam = temp; return; end
        if ~adapt_T
            lam = temp;
            w = normalise_weights(d, lam);
            return;
        end
        target = max(ess_frac*numel(J), 1.5);
        lo = eps; hi = max(temp, max(d));
        for ex = 1:60
            if ess_of(d, hi) >= target, break; end
            hi = hi*2;
        end
        if ess_of(d, hi) < target       % cannot reach the target at any lam
            lam = hi;  w = normalise_weights(d, lam);  return;
        end
        for bs = 1:60
            mid = 0.5*(lo+hi);
            if ess_of(d, mid) < target, lo = mid; else, hi = mid; end
        end
        lam = hi;
        w = normalise_weights(d, lam);
    end

    function e = ess_of(d, lam)
        w = normalise_weights(d, lam);
        e = 1/max(sum(w.^2), eps);
    end

    function E = sample_noise(stream)
        E = zeros(2,nSeg,n_roll);
        if antithetic
            nh = floor((n_roll-1)/2);
            A = zeros(2,nSeg,nh);
            A(1,:,:) = sig_nu*randn(stream,1,nSeg,nh);
            A(2,:,:) = sig_kap*randn(stream,1,nSeg,nh);
            E(:,:,2:(nh+1)) = A;
            E(:,:,(nh+2):(2*nh+1)) = -A;
            if 2*nh+1 < n_roll
                E(1,:,(2*nh+2):end) = sig_nu*randn(stream,1,nSeg,n_roll-2*nh-1);
                E(2,:,(2*nh+2):end) = sig_kap*randn(stream,1,nSeg,n_roll-2*nh-1);
            end
        else
            E(1,:,:) = sig_nu*randn(stream,1,nSeg,n_roll);
            E(2,:,:) = sig_kap*randn(stream,1,nSeg,n_roll);
            E(:,:,1) = 0; % always retain the current nominal sequence
        end
    end

    function [U,pstart,tstart] = controls_from_fm(P,T)
        chord = diff(P,1,2);
        len = sqrt(sum(chord.^2,1));
        dirs = chord ./ max(len,eps);
        bad = len <= 1e-12;
        if any(bad)
            tf = T(:,1:nSeg); tf = tf./max(sqrt(sum(tf.^2,1)),eps);
            dirs(:,bad) = tf(:,bad);
        end
        kap0 = zeros(1,nSeg);
        if nSeg > 1
            kap0(1:nSeg-1) = atan2( ...
                dirs(1,1:end-1).*dirs(2,2:end)-dirs(2,1:end-1).*dirs(1,2:end), ...
                sum(dirs(:,1:end-1).*dirs(:,2:end),1));
        end
        % No clipping here.  The bounds describe action feasibility, and the FM
        % prior violates them (13.4% of its segments exceed nu_max), so clipping
        % at this point silently replaces the warm start with a different
        % trajectory: because the rollout is sequential, one shortened segment
        % shifts every later point, and the endpoint drifted by 0.404 on a track
        % about 1.0 across, taking KL from 0.124 to 14.7 before MPPI even ran.
        % The bounds are enforced by a cost term instead, so MPPI starts from
        % the FM trajectory exactly and has to earn its way back into the box.
        U = [len; kap0];
        pstart = P(:,1); tstart = dirs(:,1);
    end

    function U = project_controls(U)
        U(1,:,:) = min(max(U(1,:,:), nu_min), nu_max);
        U(2,:,:) = min(max(U(2,:,:), -kap_max), kap_max);
    end

    function [J,X,safe] = rollout_cost(U,pstart,tstart,Pref,Tref,Uref)
        K = size(U,3);
        P = zeros(2,nP,K); T = zeros(2,nP,K);
        P(:,1,:) = repmat(pstart,1,1,K);
        T(:,1,:) = repmat(tstart,1,1,K);
        for kk = 1:nSeg
            nuk = reshape(U(1,kk,:),1,1,K);
            kak = reshape(U(2,kk,:),1,1,K);
            P(:,kk+1,:) = P(:,kk,:) + nuk.*T(:,kk,:);
            c = cos(kak); s = sin(kak);
            tx = T(1,kk,:); ty = T(2,kk,:);
            T(1,kk+1,:) = c.*tx-s.*ty;
            T(2,kk+1,:) = s.*tx+c.*ty;
        end

        Jref = zeros(1,K); Jtan = zeros(1,K);
        Jobs = zeros(1,K); Jbnd = zeros(1,K); Nh = zeros(1,K);
        for kk = 1:nP
            id0 = nF*(kk-1);
            sp = max(sd(id0+(1:2)),1e-8);
            st = max(sd(id0+(3:4)),1e-8);
            ep = (P(:,kk,:)-Pref(:,kk))./reshape(sp,2,1,1);
            et = (T(:,kk,:)-normalise_columns(Tref(:,kk)))./reshape(st,2,1,1);
            Jref = Jref + squeeze(sum(ep.^2,1)).';
            Jtan = Jtan + squeeze(sum(et.^2,1)).';
            pk = reshape(P(:,kk,:),2,K);
            for jo = 1:size(obst.centers,2)
                h = obstacle_values(pk,jo);
                Jobs = Jobs + max(0,-h).^2;
                Nh = Nh + double(h < -safety_tol);
            end
            for bi = 1:2
                h = boundary_values(pk,bi);
                Jbnd = Jbnd + max(0,-h).^2;
                Nh = Nh + double(h < -safety_tol);
            end
        end
        spf = max(sd(nF*(nP-1)+(1:2)),1e-8);
        epf = (P(:,end,:)-Pref(:,end))./reshape(spf,2,1,1);
        Jterm = squeeze(sum(epf.^2,1)).';
        enu = (U(1,:,:)-Uref(1,:))/max(nu_max-nu_min,eps);
        eka = (U(2,:,:)-Uref(2,:))/kap_max;
        Jact = squeeze(sum(enu.^2+eka.^2,2)).';
        dnu = diff(U(1,:,:),1,2)/max(nu_max-nu_min,eps);
        dka = diff(U(2,:,:),1,2)/kap_max;
        Jsm = squeeze(sum(dnu.^2+dka.^2,2)).';
        % Action feasibility, now a penalty rather than a clip.  Normalised by
        % the box width so nu and kappa violations are commensurate.
        exnu = max(0, U(1,:,:)-nu_max) + max(0, nu_min-U(1,:,:));
        exka = max(0, abs(U(2,:,:))-kap_max);
        Jbox = squeeze(sum((exnu/max(nu_max-nu_min,eps)).^2 + ...
                           (exka/kap_max).^2, 2)).';
        J = w_ref*Jref/nP + w_tan*Jtan/nP + w_term*Jterm + ...
            w_act*Jact/nSeg + w_smooth*Jsm/max(nSeg-1,1) + ...
            w_obs*Jobs + w_bnd*Jbnd + w_hit*Nh + w_box*Jbox;
        safe = Nh == 0;
        X = struct('P',P,'T',T);
    end

    function w = normalise_weights(d, lam)
        w = exp(max(-d/max(lam,realmin), -700));
        sw = sum(w);
        if ~isfinite(sw) || sw <= 0
            w = zeros(size(d));  [~,ib] = min(d);  w(ib) = 1;
        else
            w = w/sw;
        end
    end

    function t = normalise_columns(t)
        t = t./max(sqrt(sum(t.^2,1)),eps);
    end

    function h = obstacle_values(P,jo)
        c = obst.centers(:,jo);
        ax = obst.semi_axes(:,jo);
        ang = field_value(obst,'angles',jo,0);
        ex = field_value(obst,'exponents',jo,2);
        R = [cos(ang),-sin(ang);sin(ang),cos(ang)];
        q = R.'*(P-c);
        ps = sum(abs(q./ax).^ex,1);
        if ex > 2, h = ps.^(1/ex)-1; else, h = ps-1; end
    end

    function h = boundary_values(P,bi)
        f = geom.implicit_fields;
        xmax = f.x_min+f.dx*(f.n_x-1); ymax = f.y_min+f.dy*(f.n_y-1);
        pg = [min(max(P(1,:),f.x_min),xmax); min(max(P(2,:),f.y_min),ymax)];
        xc = (pg(1,:)-f.x_min)/f.dx+1; yc = (pg(2,:)-f.y_min)/f.dy+1;
        xi = min(max(floor(xc),1),f.n_x-1); yi = min(max(floor(yc),1),f.n_y-1);
        xf = xc-xi; yf = yc-yi; V = f.h{bi};
        i00 = sub2ind(size(V),yi,xi); i10 = sub2ind(size(V),yi,xi+1);
        i01 = sub2ind(size(V),yi+1,xi); i11 = sub2ind(size(V),yi+1,xi+1);
        h00=V(i00); h10=V(i10); h01=V(i01); h11=V(i11);
        h=(1-xf).*(1-yf).*h00+xf.*(1-yf).*h10+(1-xf).*yf.*h01+xf.*yf.*h11;
        gx=((1-yf).*(h10-h00)+yf.*(h11-h01))/f.dx;
        gy=((1-xf).*(h01-h00)+xf.*(h11-h10))/f.dy;
        h=h+gx.*(P(1,:)-pg(1,:))+gy.*(P(2,:)-pg(2,:));
    end

    function v = field_value(S,name,idx,default)
        if ~isfield(S,name) || isempty(S.(name)), v=default; return; end
        a=S.(name); if isscalar(a), v=a; else, v=a(idx); end
    end
end
