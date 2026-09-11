function out = uniconflow_cem_refine(r1, opts)
%UNICONFLOW_CEM_REFINE  Stage 2 of UniConFlow: CEM window refinement.
%
% Algorithm 2, lines 6-20.  Takes the output of uniconflow_nn_rollout (stage 1,
% PTZF-guided sampling) and returns a trajectory that the dynamics actually
% generate and that satisfies the state constraints, or reports where it could
% not.
%
% ---------------------------------------------------------------------------
% WHY THIS STAGE EXISTS.  Stage 1 leaves g ~ 0.06 and max h ~ 0.018: the state
% and the actions are nearly but not exactly consistent, and a few points are
% still unsafe.  Algorithm 2 says so itself -- its refinement blocks are
% "while constraints violated" loops, not optional polish.  This is the same
% division of labour SafeFlow gives its terminal projection (32) and PCFM its
% Eq. (7).
%
% Stage 1 returns both a state and a virtual-action sequence.  The paper's
% terminal refinement continues from those actions, so that is the default.
% A reconstructed-action ablation remains available through
% use_stage1_actions=false, but is not the paper-faithful branch.
%
% The rollout used everywhere is
%
%     p_{k+1} = p_k + nu_k t_k,   t_{k+1} = R(kappa_k) t_k
%
% Every CEM sample is scored by re-running that rollout from the window's entry
% state, never by moving trajectory points directly -- which is what makes this
% a kinodynamic refinement rather than a projection.
%
% ---------------------------------------------------------------------------
% WINDOWS (line 7).  Points with h > 0 are grouped into contiguous runs; each
% run is padded by n_pad on both sides to form a violation window, and the
% n_rec points that follow it form a recovery window whose job is to steer the
% state back onto the reference so the untouched remainder still applies.  What
% is left over is frozen.
%
% Frozen, violation, and recovery windows are optimized separately, followed by
% a global pass.  Actions are sampled in the hard admissible set.  Violation and
% global phases rank feasible candidates first; accepted updates cannot regress
% the best full-horizon certificate found so far.
%
% NOT A PARAMETER-LEVEL REPRODUCTION.  UniConFlow publishes no CEM population,
% elite fraction, initial variance, iteration count, n_pad or n_rec, so every
% one of them is an option here and is recorded in out.options.

if nargin < 2, opts = struct(); end
gf = @(f,d) struct_field_default(opts, f, d);

n_pop    = gf('cem_population', 256);
n_elite  = gf('cem_elite', 32);
n_iter   = gf('cem_iterations', 12);
sig_nu   = gf('cem_sigma_nu', 0.008);
sig_kap  = gf('cem_sigma_kappa', 0.10);
shrink   = gf('cem_shrink', 0.85);       % variance decay per CEM iteration
n_pad    = gf('n_pad', 2);
n_rec    = gf('n_rec', 6);
max_pass = gf('max_passes', 10);         % numerical cap for Algorithm 2's while
tol      = gf('violation_tol', 1e-8);
w_vio    = gf('violation_weight', 1e4);
w_join   = gf('rejoin_weight', 1e3);
% Swept over {1, 10, 100, 1000} on 20 trajectories sharing one stage-1 run.
% The response is single-peaked, not a safety/fidelity trade-off: 10 is best
% on all three axes at once (Safety 60%, KL 1.28, 2.10 residual violations),
% against 1 (50%, 1.39, 3.10), 100 (45%, 3.41, 3.90) and 1000 (50%, 6.92,
% 5.55).  Too small and CEM reshapes the window freely; too large and it can
% neither clear the violations nor rejoin the reference, and the unrecovered
% offset propagates downstream to the endpoint that KL measures.
w_dev    = gf('action_deviation_weight', 10.0);
w_box    = gf('action_bound_weight', 1e3);
w_smooth = gf('smoothness_weight', 1.0);
seed     = gf('seed', 4242);
use_a1   = gf('use_stage1_actions', true);
hard_bounds = gf('hard_action_bounds', true);
global_refine = gf('global_refinement', true);
rollback = gf('rollback_on_regression', true);
anchor_initial = gf('anchor_initial_state', true);
nu_min   = gf('nu_min', 0);
nu_max   = gf('nu_max', r1.options.nu_max);
kap_max  = gf('kappa_max', r1.options.kappa_max);

obst = r1.obstacle;  geom = r1.geometry;
nObs = size(obst.centers, 2);
[nP, n_gen, ~] = size(r1.points);
nSeg = nP - 1;

Pout = zeros(nP, n_gen, 2);
Tout = zeros(nP, n_gen, 2);
NuO  = zeros(nSeg, n_gen);
KapO = zeros(nSeg, n_gen);
n_win_total = zeros(n_gen, 1);
n_bad_before = zeros(n_gen, 1);
n_bad_after  = zeros(n_gen, 1);
passes_used  = zeros(n_gen, 1);
n_frozen_total = zeros(n_gen,1);
n_vio_total = zeros(n_gen,1);
n_rec_total = zeros(n_gen,1);
action_bad_before = zeros(n_gen,1);
action_bad_after = zeros(n_gen,1);

t_cem = tic;
for s = 1:n_gen
    Phat=squeeze(r1.points(:,s,:)).';
    That=squeeze(r1.features(:,s,3:4)).';
    That=That./max(sqrt(sum(That.^2,1)),eps);
    p0 = Phat(:,1);
    t0 = That(:,1);
    t0 = t0 / max(norm(t0), eps);
    if use_a1
        U0 = [r1.nu(:,s).'; r1.kappa(:,s).'];
    else
        [U0, p0, t0] = actions_from_points(Phat);
    end
    action_bad_before(s)=nnz(action_violation(U0)>tol);
    if hard_bounds, U0=project_controls(U0); end
    Uref=U0;
    [Uinv,~,tinv]=actions_from_points(Phat);
    if ~anchor_initial, t0=tinv; end
    if hard_bounds, Uinv=project_controls(Uinv); end
    % The project has no prescribed current state/heading.  Use inverse
    % dynamics only as a deterministic CEM proposal, while retaining the
    % Stage-1 actions as the fixed reference in every deviation cost.
    if ~anchor_initial && reference_error(Uinv,p0,t0,Phat,That) < reference_error(U0,p0,t0,Phat,That)
        U=Uinv;
    else
        U=U0;
    end
    n_bad_before(s) = nnz(point_violation(Phat) > tol);

    stream = RandStream('mt19937ar', 'Seed', mod(seed + s, 2^31-2) + 1);

    % Algorithm 2, lines 8-11: enforce dynamics/alignment on frozen windows
    % before repairing unsafe windows.  Unlike the old implementation, the
    % stage-1 action sequence remains the reference throughout all passes.
    [~,~,F0]=window_sets(point_violation(Phat)>tol,n_pad,n_rec,nSeg);
    n_frozen_total(s)=size(F0,1);
    for iw=1:size(F0,1)
        U=refine_span(U,F0(iw,:),Phat,That,Uref,'frozen',stream);
    end

    bestU=U; bestKey=global_key(U,p0,t0,Phat,That,Uref);
    for pass = 1:max_pass
        [P, T] = rollout(U, p0, t0);
        vio = point_violation(P) > tol;
        if ~any(vio), break; end
        passes_used(s) = pass;
        [V,R,~] = window_sets(vio,n_pad,n_rec,nSeg);
        n_win_total(s)=n_win_total(s)+size(V,1);
        n_vio_total(s)=n_vio_total(s)+size(V,1);
        for iw = 1:size(V,1)
            U=refine_span(U,V(iw,:),Phat,That,Uref,'violation',stream);
            if iw<=size(R,1) && R(iw,1)>0
                U=refine_span(U,R(iw,:),Phat,That,Uref,'recovery',stream);
                n_rec_total(s)=n_rec_total(s)+1;
            end
        end
        key=global_key(U,p0,t0,Phat,That,Uref);
        if lex_less(key,bestKey), bestU=U; bestKey=key;
        elseif rollback, U=bestU;
        end
    end

    % Algorithm 2, line 20: one final full-horizon CEM pass.  It is accepted
    % only if the lexicographic certificate (violations first, cost second)
    % does not regress.
    if global_refine
        Ug=refine_span(U,[1 nSeg],Phat,That,Uref,'global',stream);
        if lex_less(global_key(Ug,p0,t0,Phat,That,Uref), ...
                global_key(U,p0,t0,Phat,That,Uref)), U=Ug; end
    end

    [P, T] = rollout(U, p0, t0);
    n_bad_after(s) = nnz(point_violation(P) > tol);
    action_bad_after(s)=nnz(action_violation(U)>tol);
    Pout(:,s,:) = permute(P, [2 3 1]);
    Tout(:,s,:) = permute(T, [2 3 1]);
    NuO(:,s) = U(1,:).';  KapO(:,s) = U(2,:).';
end
cem_s = toc(t_cem);

Fout = cat(3, Pout, Tout);
out = r1;
out.points = Pout;  out.features = Fout;
out.nu = NuO;  out.kappa = KapO;
out.mode = 'uniconflow';
out.stage1_points = r1.points; out.stage1_features=r1.features;
out.stage1_nu=r1.nu; out.stage1_kappa=r1.kappa;
if isfield(r1,'normalization_mu') && isfield(r1,'normalization_sd')
    Xout=reshape(permute(Fout,[3 1 2]),[],n_gen);
    out.state=(Xout-r1.normalization_mu(:))./r1.normalization_sd(:);
else
    out.state=[]; % never expose the stale Stage-1 normalized state
end
gnew=zeros(1,n_gen); hnew=zeros(1,n_gen);
for s=1:n_gen
    PP=squeeze(Pout(:,s,:)).'; TT=squeeze(Tout(:,s,:)).';
    UU=[NuO(:,s).';KapO(:,s).'];
    gnew(s)=trajectory_residual(PP,TT,UU);
    hnew(s)=max([point_violation(PP),action_violation(UU)]);
end
out.g_final=gnew; out.h_max_final=hnew;
out.violations_before = n_bad_before;
out.violations_after  = n_bad_after;
out.windows_processed = n_win_total;
out.frozen_windows_processed=n_frozen_total;
out.violation_windows_processed=n_vio_total;
out.recovery_windows_processed=n_rec_total;
out.cem_passes = passes_used;
out.terminal_corrected = nnz(n_bad_before > 0);
out.state_terminal_failed = nnz(n_bad_after > 0);
out.action_violations_before=action_bad_before;
out.action_violations_after=action_bad_after;
out.action_terminal_failed=nnz(action_bad_after>0);
out.terminal_failed = nnz((n_bad_after > 0) | (action_bad_after > 0) | (gnew(:)>tol));
out.terminal_seconds = cem_s;
out.total_seconds_per_traj = (r1.sample_seconds + r1.rollout_seconds + cem_s)/n_gen;
out.terminal_audit=struct('certified', ...
    (n_bad_after==0)&(action_bad_after==0)&(gnew(:)<=tol), ...
    'state_violations',n_bad_after,'action_violations',action_bad_after, ...
    'kinodynamic_residual',gnew(:));
out.options.method='uniconflow_full';
out.options.stage2 = struct('cem_population', n_pop, 'cem_elite', n_elite, ...
    'cem_iterations', n_iter, 'cem_sigma_nu', sig_nu, ...
    'cem_sigma_kappa', sig_kap, 'cem_shrink', shrink, 'n_pad', n_pad, ...
    'n_rec', n_rec, 'max_passes', max_pass, 'violation_tol', tol, ...
    'violation_weight', w_vio, 'rejoin_weight', w_join, ...
    'action_deviation_weight', w_dev, 'action_bound_weight', w_box, ...
    'smoothness_weight',w_smooth, ...
    'seed', seed, 'nu_min', nu_min, 'nu_max', nu_max, 'kappa_max', kap_max, ...
    'use_stage1_actions', use_a1, 'hard_action_bounds', hard_bounds, ...
    'global_refinement',global_refine,'rollback_on_regression',rollback, ...
    'anchor_initial_state',anchor_initial);

% =====================================================================
    % Chord lengths and turning angles that replay the given positions exactly.
    % No clipping: the bounds are a cost inside the CEM objective, and clipping
    % here would silently substitute a different trajectory for the reference.
    function [U, pstart, tstart] = actions_from_points(P)
        ch = diff(P, 1, 2);
        len = sqrt(sum(ch.^2, 1));
        dirs = ch ./ max(len, eps);
        ns=size(P,2)-1;
        ka = zeros(1, ns);
        if ns > 1
            ka(1:ns-1) = atan2( ...
                dirs(1,1:end-1).*dirs(2,2:end) - dirs(2,1:end-1).*dirs(1,2:end), ...
                sum(dirs(:,1:end-1).*dirs(:,2:end), 1));
        end
        U = [len; ka];
        pstart = P(:,1);  tstart = dirs(:,1);
    end

    function [P, T] = rollout(U, pstart, tstart)
        P = zeros(2, nP);  T = zeros(2, nP);
        P(:,1) = pstart;  T(:,1) = tstart;
        for k = 1:nSeg
            P(:,k+1) = P(:,k) + U(1,k)*T(:,k);
            c = cos(U(2,k));  sn = sin(U(2,k));
            T(:,k+1) = [c*T(1,k) - sn*T(2,k); sn*T(1,k) + c*T(2,k)];
        end
    end

    % Worst constraint violation at each point, in the paper's h <= 0 sense.
    function v = point_violation(P)
        v = zeros(1, size(P,2));
        for k = 1:size(P,2)
            p = P(:,k);  worst = -inf;
            for jo = 1:nObs
                worst = max(worst, -obstacle_level_and_gradient(p, obst, jo));
            end
            for bi = 1:2
                worst = max(worst, ...
                    -evaluate_track_implicit_field(geom.implicit_fields, bi, p));
            end
            v(k) = worst;
        end
    end

    function Unew=refine_span(Ucur,span,Phat,That,Ufixed,mode,stream)
        ka=span(1); kb=span(2);
        if ka<1 || kb<ka || ka>nSeg, Unew=Ucur; return; end
        kb=min(kb,nSeg);
        [Pc,Tc]=rollout(Ucur,p0,t0);
        Uw=Ucur(:,ka:kb);
        Ptar=Phat(:,ka:kb+1); Ttar=That(:,ka:kb+1);
        Ucand=cem_window(Uw,Pc(:,ka),Tc(:,ka),Ptar,Ttar,Ufixed(:,ka:kb),mode,stream);
        Utrial=Ucur; Utrial(:,ka:kb)=Ucand;
        if hard_bounds, Utrial=project_controls(Utrial); end
        if strcmp(mode,'frozen')
            % Frozen enforcement precedes certification in Algorithm 2.  Its
            % purpose is to make the action rollout follow the generated state;
            % temporary downstream violations are repaired in the next phase.
            Unew=Utrial;
        elseif lex_less(global_key(Utrial,p0,t0,Phat,That,Ufixed), ...
                global_key(Ucur,p0,t0,Phat,That,Ufixed))
            Unew=Utrial;
        else
            Unew=Ucur;
        end
    end

    function Ubest = cem_window(Uw, pe, te, Ptar, Ttar, Uref, mode, stream)
        nw = size(Uw, 2);
        mu_u = project_controls(Uw);
        sg = [sig_nu*ones(1,nw); sig_kap*ones(1,nw)];
        Ubest = mu_u; [n0,v0,j0]=window_cost(Ubest,pe,te,Ptar,Ttar,Uref,mode);
        bestRank=rank_key(n0,v0,j0,mode);
        for it = 1:n_iter
            C = zeros(2, nw, n_pop);
            C(:,:,1) = mu_u;                     % always keep the current mean
            qstart=2;
            if n_pop>=2
                C(:,:,2)=tracking_proposal(pe,te,Ptar,Ttar); qstart=3;
            end
            if n_pop>=3 && any(strcmp(mode,{'frozen','global'}))
                [Uinv,~,~]=actions_from_points(Ptar);
                C(:,:,3)=project_controls(Uinv); qstart=4;
            end
            for q = qstart:n_pop
                C(:,:,q) = mu_u + sg .* randn(stream, 2, nw);
                if hard_bounds, C(:,:,q)=project_controls(C(:,:,q)); end
            end
            J = zeros(1,n_pop); Nv=J; Vm=J;
            for q = 1:n_pop
                [Nv(q),Vm(q),J(q)]=window_cost(C(:,:,q),pe,te,Ptar,Ttar,Uref,mode);
            end
            if any(strcmp(mode,{'frozen','recovery'}))
                [~,ord]=sort(J);
            else
                [~,ord]=sortrows([Nv(:),Vm(:),J(:)],[1 2 3]);
            end
            E = C(:,:,ord(1:min(n_elite, n_pop)));
            mu_u = mean(E, 3); if hard_bounds, mu_u=project_controls(mu_u); end
            sg = max(std(E, 0, 3), 1e-9) * shrink;
            q0=ord(1);
            qr=rank_key(Nv(q0),Vm(q0),J(q0),mode);
            if lex_less(qr,bestRank)
                bestRank=qr; Ubest=C(:,:,q0);
            end
        end
        [nm,vm,Jm]=window_cost(mu_u,pe,te,Ptar,Ttar,Uref,mode);
        if lex_less(rank_key(nm,vm,Jm,mode),bestRank), Ubest=mu_u; end
    end

    % Eq. (108) in window form: stay close to the stage-1 actions, remove the
    % violations, and land on the reference state so the frozen remainder still
    % applies.  Every candidate is rolled out from the window's entry state.
    function [nv,vm,J] = window_cost(Uw, pe, te, Ptar, Ttar, Uref, mode)
        nw = size(Uw, 2);
        P = zeros(2, nw+1);  T = zeros(2, nw+1);
        P(:,1) = pe;  T(:,1) = te;
        for k = 1:nw
            P(:,k+1) = P(:,k) + Uw(1,k)*T(:,k);
            c = cos(Uw(2,k));  sn = sin(Uw(2,k));
            T(:,k+1) = [c*T(1,k) - sn*T(2,k); sn*T(1,k) + c*T(2,k)];
        end
        v = point_violation(P(:,2:end));
        av=action_violation(Uw);
        vv=[max(0,v),max(0,av)]; nv=nnz(vv>tol);
        if isempty(vv), vm=0; else, vm=max(vv); end
        Jv = sum(max(0, v).^2) + 0.01*nnz(v > tol);
        Jj = sum((P(:,end)-Ptar(:,end)).^2)+sum((T(:,end)-Ttar(:,end)).^2);
        Jr=sum((P-Ptar).^2,'all')+sum((T-Ttar).^2,'all');
        Jd = sum((Uw(1,:) - Uref(1,:)).^2)/max(nu_max,eps)^2 + ...
             sum((Uw(2,:) - Uref(2,:)).^2)/kap_max^2;
        exn = max(0, Uw(1,:) - nu_max) + max(0, nu_min - Uw(1,:));
        exk = max(0, abs(Uw(2,:)) - kap_max);
        Jb = sum((exn/max(nu_max,eps)).^2) + sum((exk/kap_max).^2);
        Js=0;
        if nw>1
            Js=sum(diff(Uw(1,:)).^2)/max(nu_max,eps)^2+ ...
                sum(diff(Uw(2,:)).^2)/kap_max^2;
        end
        switch mode
            case 'frozen'
                J=w_join*Jr+w_dev*Jd/max(nw,1)+w_smooth*Js+w_box*Jb;
            case 'recovery'
                J=w_vio*Jv+w_join*(10*Jj+Jr)+w_dev*Jd/max(nw,1)+w_smooth*Js+w_box*Jb;
            otherwise % violation and final global certification
                J=w_vio*Jv+w_join*Jj+w_dev*(Jr+Jd)/max(nw,1)+w_smooth*Js+w_box*Jb;
        end
    end

    function U=project_controls(U)
        U(1,:)=min(max(U(1,:),nu_min),nu_max);
        U(2,:)=min(max(U(2,:),-kap_max),kap_max);
    end

    function av=action_violation(U)
        av=max([nu_min-U(1,:);U(1,:)-nu_max;abs(U(2,:))-kap_max],[],1);
    end

    function U=tracking_proposal(pe,te,Ptar,Ttar)
        nw=size(Ptar,2)-1; U=zeros(2,nw); p=pe; tt=te/max(norm(te),eps);
        for jj=1:nw
            d=Ptar(:,jj+1)-p;
            un=min(max(dot(d,tt),nu_min),nu_max);
            U(1,jj)=un; p=p+un*tt;
            if jj<nw, desired=Ptar(:,jj+2)-p; else, desired=Ttar(:,end); end
            if norm(desired)>eps
                desired=desired/norm(desired);
                ka=atan2(tt(1)*desired(2)-tt(2)*desired(1),dot(tt,desired));
            else
                ka=0;
            end
            ka=min(max(ka,-kap_max),kap_max); U(2,jj)=ka;
            c=cos(ka); sn=sin(ka); tt=[c*tt(1)-sn*tt(2);sn*tt(1)+c*tt(2)];
        end
    end

    function key=global_key(U,pstart,tstart,Phat,That,Ufixed)
        [Pg,Tg]=rollout(U,pstart,tstart);
        v=max(0,point_violation(Pg)); av=max(0,action_violation(U));
        allv=[v,av];
        jd=sum((Pg-Phat).^2,'all')+sum((Tg-That).^2,'all')+ ...
            0.1*sum((U-Ufixed).^2,'all');
        key=[nnz(allv>tol),max(allv),jd];
    end

    function e=reference_error(U,pstart,tstart,Phat,That)
        [Pr,Tr]=rollout(U,pstart,tstart);
        e=sum((Pr-Phat).^2,'all')+sum((Tr-That).^2,'all');
    end

    function tf=lex_less(a,b)
        tf=false;
        for kk=1:numel(a)
            sc=max([1,abs(a(kk)),abs(b(kk))]);
            if a(kk)<b(kk)-1e-12*sc, tf=true; return; end
            if a(kk)>b(kk)+1e-12*sc, return; end
        end
    end

    function key=rank_key(nv,vm,J,mode)
        if any(strcmp(mode,{'frozen','recovery'})), key=[J,0,0];
        else, key=[nv,vm,J]; end
    end

    function gv=trajectory_residual(P,T,U)
        gv=0;
        for kk=1:nSeg
            e=P(:,kk+1)-P(:,kk)-U(1,kk)*T(:,kk);
            c=cos(U(2,kk)); sn=sin(U(2,kk));
            rt=[c*T(1,kk)-sn*T(2,kk);sn*T(1,kk)+c*T(2,kk)];
            q=T(:,kk+1)-rt;
            gv=gv+e'*e+q'*q;
        end
        gv=gv+sum((sum(T.^2,1)-1).^2);
    end
end

% =====================================================================
function [V,R,F] = window_sets(vio, n_pad, n_rec, nSeg)
%WINDOW_SETS  Disjoint violation, recovery and frozen action-index spans.
% vio is a per-POINT logical mask (1..nP).  Action k moves point k to point k+1,
% so a violated point k is influenced by actions 1..k-1; the window starts
% n_pad actions before it and runs n_rec actions past the end of the run.
d = diff([false, vio(:)', false]);
starts = find(d == 1);
stops  = find(d == -1) - 1;
V = zeros(numel(starts), 2);
for i = 1:numel(starts)
    ka = max(1, starts(i) - 1 - n_pad);
    kb = min(nSeg, stops(i) + n_pad - 1);
    V(i,:) = [ka, max(ka,kb)];
end
if size(V,1) > 1
    M = V(1,:);
    for i = 2:size(V,1)
        if V(i,1) <= M(end,2)+1
            M(end,2) = max(M(end,2), V(i,2));
        else
            M(end+1,:) = V(i,:); %#ok<AGROW>
        end
    end
    V = M;
end
R=zeros(size(V));
occupied=false(1,nSeg);
for i=1:size(V,1)
    occupied(V(i,1):V(i,2))=true;
    ra=V(i,2)+1; rb=min(nSeg,V(i,2)+n_rec);
    if i<size(V,1), rb=min(rb,V(i+1,1)-1); end
    if ra<=rb, R(i,:)=[ra rb]; occupied(ra:rb)=true; end
end
F=[]; d=diff([false,~occupied,false]); fs=find(d==1); fe=find(d==-1)-1;
for i=1:numel(fs), F(end+1,:)=[fs(i),fe(i)]; end %#ok<AGROW>
end
