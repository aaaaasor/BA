function exp_tvsd_gamma_sweep(n_gen)
%EXP_TVSD_GAMMA_SWEEP  Pick a defensible gamma(t) for the TVSD-style baseline.
%
% TVSD is the only ported mechanism that reaches full obstacle safety on its
% own (100% of points, min h = +0.001, no terminal projection), but with the
% first gamma tried -- sigmoid, gamma(0) = -1, steepness 10 -- it costs KL 3.11
% against 0.11 for the others.  The paper does not state the form of gamma, so
% that number reflects an arbitrary choice, not the mechanism.
%
% What is and is not free to choose:
%
%   gamma_min   NOT free.  Thm 4 needs gamma(0) <= b(x^0) for every point.  The
%               obstacle level set h = (s-c)'Q(s-c) - 1 bottoms out at -1, so
%               gamma_min = -1 is the only value that covers all points.
%               Raising it to -0.2 would cut peak gamma_dot fivefold but drop
%               every point with b < -0.2 out of the guarantee -- and the
%               measured worst b at t = 0 is -0.81 per trajectory (median).
%
%   steepness   Free, and bounded.  int_0^1 gamma_dot dt = |gamma_min| is fixed,
%               so steepness only redistributes it in time; the peak is minimised
%               by a straight line and equals |gamma_min| there.  A minimum-norm
%               QP pays the square of the demand, so spreading strictly beats
%               spiking.  k = 10 peaks at 2.53; k = 2 already reaches 1.08,
%               against the floor of 1.00.
%
%   alpha       Free.  The constraint only bites where alpha(s) < gamma_dot, i.e.
%               s < gamma_dot/alpha, so a larger alpha narrows the band of points
%               that feel it at all.  alpha*dt = 0.0996 at alpha = 10, far below
%               the discrete-stability limit of 1, so there is budget.
%
%   midpoint    Free.  Shifting it later lets the flow carry points out of the
%               obstacles on its own before the set starts closing.
%
% The unguided FM obstacle-h profile is printed first: it is the trajectory the
% flow follows with no guidance at all, so a gamma that stays below it is one
% the CBF barely has to enforce.  That is the cheapest gamma there can be.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_gen), n_gen = 100; end
car = char([36187 36710]);
src = fullfile('outputs', [car 'safeflow_tbar05_paper']);
if ~exist(src, 'dir'), src = fullfile('outputs', [car 'safeflow']); end
N = load(fullfile(src, 'Racing_SafeFlow_NN_Net.mat'));
A = load(fullfile(src, 'Racing_SafeFlow_NN_Rollout.mat'));
af = fieldnames(A); arch = A.(af{1});

base = struct('trajectory_seeds', arch.trajectory_seeds, 'u_per_stage', false, ...
    'fmincon_fallback', false, 'record_path', true, 'activation_time', 0, ...
    'terminal_filter', false, 'phi1_form', 'constant', 'phi1_switch_time', 0);

% ---- reference: where the unguided flow's obstacle h actually is over time ----
r0 = safeflow_nn_rollout(N.net, 'fm', n_gen, base);
[tt, q01] = fm_profile(r0, N.net);
fprintf('unguided FM: 1st percentile of obstacle h over all points\n');
for k = round(linspace(1, numel(tt), 6))
    fprintf('  t=%.3f   h_p1 = %+.4f\n', tt(k), q01(k));
end

% ---- candidates ----
C = { ...
 'linear a=1',              inf, 0.50, 1; ...
 'linear a=5',              inf, 0.50, 5; ...
 'linear a=10',             inf, 0.50, 10; ...
 'k=10 mid.50 a=1 (current)', 10, 0.50, 1; ...
 'k=6  mid.50 a=1',            6, 0.50, 1; ...
 'k=4  mid.50 a=1',            4, 0.50, 1; ...
 'k=2  mid.50 a=1',            2, 0.50, 1; ...
 'k=2  mid.50 a=5',            2, 0.50, 5; ...
 'k=2  mid.50 a=10',           2, 0.50, 10; ...
 'k=4  mid.65 a=5',            4, 0.65, 5 };

nc = size(C,1);  M = cell(nc,1);  PK = zeros(nc,1);
for i = 1:nc
    k = C{i,2}; mid = C{i,3}; al = C{i,4};
    PK(i) = peak_gamma_dot(k, mid, -1.0);
    o = base;
    if isinf(k), o.gamma_mode = 'linear'; else, o.gamma_mode = 'sigmoid'; end
    o.gamma_min = -1.0;
    o.gamma_steepness = k;     o.gamma_midpoint = mid;
    o.phi1_alpha = al;         o.phi0 = al;
    ws1 = warning('off','MATLAB:nearlySingularMatrix');
    ws2 = warning('off','MATLAB:singularMatrix');
    r = safeflow_nn_rollout(N.net, 'safeflow', n_gen, o);
    warning(ws1); warning(ws2);
    m = safeflow_run_metrics(r, N.net);
    [Ho, Hb] = final_h(r);
    m.obst_safe = 100*mean(Ho(:) >= -1e-8);
    m.bound_safe = 100*mean(Hb(:) >= -1e-8);
    m.obst_min = min(Ho(:));
    M{i} = m;
    fprintf('%-26s peak gdot %5.2f | obst safe %6.2f%%  KL %7.4f  CS %.4f\n', ...
        C{i,1}, PK(i), m.obst_safe, m.kl, m.cs);
end

fprintf('\n\n%-26s %-10s %-11s %-11s %-9s %-9s %-11s %-10s\n', 'gamma', ...
    'peak gdot', 'obst safe%', 'bound safe%', 'KL', 'CS', 'obst min h', 'slack%');
for i = 1:nc
    m = M{i};
    fprintf('%-26s %-10.2f %-11.2f %-11.2f %-9.4f %-9.4f %-11.5f %-10.1f\n', ...
        C{i,1}, PK(i), m.obst_safe, m.bound_safe, m.kl, m.cs, m.obst_min, ...
        100*m.slack_active/max(m.n_qp,1));
end
fprintf(['\nReference rows from the six-baseline run (same seeds, projection off):\n' ...
    '  SafeFlow           obst safe 96.06%%  KL 0.1145  CS 0.1980\n' ...
    '  conventional CBF   obst safe 96.02%%  KL 0.1147  CS 0.2072\n' ...
    '  FM (no guidance)   obst safe 93.54%%  KL 0.1236  CS 0.2101\n']);

S = struct('candidates', {C}, 'metrics', {M}, 'peak_gamma_dot', PK, ...
    'fm_profile_t', tt, 'fm_profile_p1', q01, 'n_gen', n_gen);
save(fullfile('outputs', 'TVSD_Gamma_Sweep.mat'), 'S', '-v7.3');
fprintf('\nsaved outputs/TVSD_Gamma_Sweep.mat\n');
end

% =====================================================================
function p = peak_gamma_dot(k, mid, gmin)
if isinf(k), p = -gmin; return; end     % linear: constant rate |gamma_min|
% gamma(t) = gmin*(sig(1)-sig(t))/(sig(1)-sig(0));  the derivative peaks at the
% midpoint, where sig = 1/2.
sg = @(t) 1./(1+exp(-k*(t-mid)));
p = -gmin * k*0.25 / (sg(1) - sg(0));
end

function [tt, q01] = fm_profile(r, net)
[n_t, Dd, ng] = size(r.z_path); nF = 4; nP = Dd/nF;
tt = linspace(0, r.t_max, n_t)';  q01 = zeros(n_t,1);
for kk = 1:n_t
    X = squeeze(r.z_path(kk,:,:)) .* net.sd_d + net.mu_d;
    P = permute(reshape(X, nF, nP, ng), [2 3 1]);
    H = inf(nP, ng);
    for s = 1:ng
        for ip = 1:nP
            p = [P(ip,s,1); P(ip,s,2)]; hm = inf;
            for jo = 1:size(r.obstacle.centers,2)
                hm = min(hm, obstacle_level_and_gradient(p, r.obstacle, jo));
            end
            H(ip,s) = hm;
        end
    end
    q01(kk) = quantile(H(:), 0.01);
end
end

function [Ho, Hb] = final_h(r)
P = r.points; [nPt, ng, ~] = size(P);
Ho = zeros(nPt, ng); Hb = zeros(nPt, ng);
for s = 1:ng
    for k = 1:nPt
        p = [P(k,s,1); P(k,s,2)];
        ho = inf; hb = inf;
        for jo = 1:size(r.obstacle.centers,2)
            ho = min(ho, obstacle_level_and_gradient(p, r.obstacle, jo));
        end
        for bi = 1:2
            hb = min(hb, evaluate_track_implicit_field( ...
                r.geometry.implicit_fields, bi, p));
        end
        Ho(k,s) = ho; Hb(k,s) = hb;
    end
end
end
