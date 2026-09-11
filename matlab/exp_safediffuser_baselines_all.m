function exp_safediffuser_baselines_all(n_gen, n_show, only)
%EXP_SAFEDIFFUSER_BASELINES_ALL  SafeDiffuser variants on our QP: all outputs.
%
% Runs each configuration ONCE and produces every result from that one rollout:
% the metric table, the pointwise safety breakdown, and the trajectory figure.
% The three used to be separate scripts, which meant three rollouts per
% configuration and three sets of numbers that were only equal because the
% seeds happened to match.
%
% ---------------------------------------------------------------------------
% Mechanisms, read from the SafeDiffuser paper (Xiao et al., arXiv 2306.00148 /
% ICLR 2025), Sec. 3.1-3.3 and Sec. 4.  None of the three variants changes the
% class-K gain; each changes the CONSTRAINT.  The three variant rows share the
% linear class-K alpha(b) = b and the guidance-from-t=0 setting, and differ from
% each other only in the mechanism named below.
%
%   RoSD  (Thm 2, Eq. 8; QP Eq. 13)  db/dx*u + alpha(b) >= 0, hard.
%         min||u - (tau^j - tau^{j+1})/dtau||^2.  The paper adds no margin and no
%         safe-set contraction; "robust" names the generality of the guarantee.
%         Eq. 7 makes u the whole velocity, ours is the correction on top of the
%         nominal flow, and u_paper = v_theta + u_c maps them onto each other
%         exactly -- objective included.        -> slack_enabled false.
%
%   ReSD  (Thm 3, Eq. 9-10; QP Eq. 14)  db/dx*u + alpha(b) - w_k(j) r >= 0,
%         min||u - nominal||^2 + ||r||^2, with w decreasing to 0 as j -> 0.
%         Eliminating r gives our penalty form with slack weight 1/w(t)^2, so
%         the schedule maps over exactly.      -> resd_weight_schedule.
%         The paper's N_a extra steps are NOT used: they work in diffusion
%         because each is another draw of the Gaussian transition that may jump
%         b across 0 (the guarantee is that lottery, 1-(1-p)^j).  A deterministic
%         flow has no such draw, and under a linear class-K those steps only run
%         hdot = -alpha*h, so h decays as exp(-alpha t) and never reaches 0 in
%         finite time.  Measured: N_a = 50 moved obstacle safety from 93.54%
%         (unguided FM) to 93.97%, versus 96.08% for RoSD with no extra steps.
%
%   TVSD  (Thm 4, Eq. 11-12; QP Eq. 13)  b(x) - gamma(t) >= 0 with gamma(0) <=
%         b at t=0 and gamma(1) = 0, enforced as db/dx*u - gamma_dot +
%         alpha(b - gamma) >= 0.  A moving SAFE SET, not a moving gain.  Alone
%         among the three, Thm 4 carries no "with almost probability 1": it is
%         deterministic, which is why it is the one that works here.
%         Eq. 13 covers "RoS diffuser else (12)", so TVSD solves the same HARD
%         QP as RoSD; only ReSD gets Eq. 14's relaxation variable.
%                                              -> gamma_mode linear, slack off.
%         gamma is the straight line gamma(t) = -(1-t): constant closing rate,
%         no free shape parameters, and gamma(0) = -1 <= b everywhere because
%         the obstacle level set bottoms out at -1.  A sigmoid shape was swept
%         (steepness 2..10, midpoints 0.5/0.65) and changed KL non-monotonically
%         between 3.0 and 4.1, so the shape does not matter; alpha does.
%
% ---------------------------------------------------------------------------
% READ THE NUMBERS WITH THESE THREE CAVEATS.
%
% (a) The terminal projection (32) is OFF.  It is not part of any variant, and
%     leaving it on lets it supply the safety the CBF-QP failed to -- which is
%     exactly what these rows measure.  So they are NOT comparable to our
%     SafeFlow rows, which keep (32).
%
% (b) The hard QP has a fallback.  minnorm_halfspaces enumerates active sets and
%     returns pinv(G)*max(c,0) when no subset is feasible, which satisfies
%     nothing in particular.  A non-zero infeas count means that row did not
%     hold the paper's hard constraint everywhere.
%
% (c) Not a reproduction.  No diffusion backbone, so the stochastic transitions
%     behind "with almost probability 1" are absent; and the paper does not give
%     the forms of gamma(t) or w(t), so the linear ramp and the step are ours.
%
% Obstacle and boundary h are always reported separately: the obstacle level set
% h = (s-c)'Q(s-c) - 1 has a floor of -1 at the centre, while the boundary field
% is a clamped bilinear grid, so a min over both is not a comparable quantity.
% The boundary rows are our racing addition -- neither paper's experiment has them.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_gen),  n_gen  = 100; end
if nargin < 2 || isempty(n_show), n_show = 20;  end
% only: cellstr of row-name prefixes to run, e.g. {'RoSD','ReSD','TVSD'}.
% Omit for all rows.  FM is bit-identical to outputs/<racing>fm because mode
% 'fm' has no QP, so activation_time and terminal_filter do not reach it;
% SafeFlow here is NOT the archived run (that one is the second-order-pole
% ablation with the projection on).
if nargin < 3, only = {}; end
car = char([36187 36710]);
src = fullfile('outputs', [car 'safeflow_tbar05_paper']);
if ~exist(src, 'dir'), src = fullfile('outputs', [car 'safeflow']); end
N = load(fullfile(src, 'Racing_SafeFlow_NN_Net.mat'));
A = load(fullfile(src, 'Racing_SafeFlow_NN_Rollout.mat'));
af = fieldnames(A); arch = A.(af{1});

base = struct('trajectory_seeds', arch.trajectory_seeds, 'u_per_stage', false, ...
    'fmincon_fallback', false, 'record_path', true, 'activation_time', 0, ...
    'terminal_filter', false);
conv = struct('phi1_form', 'constant', 'phi1_alpha', 1.0, 'phi1_switch_time', 0);

V = { ...
 'FM (no guidance)',       'fm',       struct(); ...
 'SafeFlow (paper gain)',  'safeflow', struct('phi1_switch_time', 0.9); ...
 'RoSD-mech (hard QP)',    'safeflow', merge(conv, struct('slack_enabled', false)); ...
 'ReSD-mech (w(t) relax)', 'safeflow', merge(conv, ...
        struct('resd_weight_schedule', [100 0 0.9])); ...
 'TVSD-mech (moving set)', 'safeflow', merge(conv, struct('gamma_mode', 'linear', ...
        'gamma_min', -1.0, 'slack_enabled', false)) };

if ~isempty(only)
    if ischar(only) || isstring(only), only = cellstr(only); end
    keep = false(size(V,1),1);
    for a = 1:size(V,1)
        keep(a) = any(cellfun(@(z) startsWith(V{a,1}, z), only));
    end
    assert(any(keep), 'only matched no row of: %s', strjoin(V(:,1)', ', '));
    V = V(keep, :);
    fprintf('running %d of the rows: %s\n', size(V,1), strjoin(V(:,1)', ', '));
end

nv = size(V,1);
M = cell(nv,1); Dq = cell(nv,1);
for i = 1:nv
    fprintf('\n===== %s =====\n', V{i,1});
    ws1 = warning('off','MATLAB:nearlySingularMatrix');
    ws2 = warning('off','MATLAB:singularMatrix');
    r = safeflow_nn_rollout(N.net, V{i,2}, n_gen, merge(base, V{i,3}));
    warning(ws1); warning(ws2);
    M{i}  = safeflow_run_metrics(r, N.net);
    Dq{i} = pointwise(r, N.net);
    Dq{i}.points = r.points;
    Dq{i}.obstacle = r.obstacle;
    Dq{i}.segment  = r.segment;
    m = M{i};
    fprintf(['  Safety %6.2f%%  KL %.4f  CS %.4f  AS %.4g  Time %.4f\n' ...
             '  infeas/slack %d of %d QP (%.2f%%) | |u|max %.3e\n'], ...
        m.safety*100, m.kl, m.cs, m.as, m.time_seconds, ...
        m.slack_active, m.n_qp, 100*m.slack_active/max(m.n_qp,1), m.u_max);
end

tol = -1e-8;
fprintf('\n\n======== metrics, terminal projection (32) OFF ========\n');
fprintf('%-24s %-9s %-9s %-9s %-11s %-9s %-10s\n', 'row', 'Safety%', 'KL', ...
    'CS', 'AS', 'Time', 'infeas');
for i = 1:nv
    m = M{i};
    fprintf('%-24s %-9.2f %-9.4f %-9.4f %-11.4g %-9.4f %-10d\n', V{i,1}, ...
        m.safety*100, m.kl, m.cs, m.as, m.time_seconds, m.slack_active);
end
fprintf(['\ninfeas: for a hard row, the steps where the constraint could not be\n' ...
    'satisfied and the least-violation fallback ran; for a soft row, the steps\n' ...
    'where the QP chose to pay the penalty instead.\n']);

fprintf('\n======== pointwise safety ========\n');
fprintf('%-24s %-11s %-11s %-11s %-11s %-12s\n', 'row', 'obst safe%', ...
    'bound safe%', 'both safe%', 'traj safe%', 'bad pts/traj');
for i = 1:nv
    d = Dq{i};
    fprintf('%-24s %-11.2f %-11.2f %-11.2f %-11.2f %-12.1f\n', V{i,1}, ...
        100*mean(d.Ho(:) >= tol), 100*mean(d.Hb(:) >= tol), ...
        100*mean(d.Ho(:) >= tol & d.Hb(:) >= tol), ...
        100*mean(all(d.Ho >= tol & d.Hb >= tol, 1)), ...
        median(sum(d.Ho < tol | d.Hb < tol, 1)));
end

fprintf('\n======== obstacle h at the final state ========\n');
fprintf('%-24s %-11s %-11s %-11s %-11s\n', 'row', 'min', 'p1', 'median', 'worst/traj');
for i = 1:nv
    d = Dq{i};
    fprintf('%-24s %-11.5f %-11.5f %-11.5f %-11.5f\n', V{i,1}, min(d.Ho(:)), ...
        quantile(d.Ho(:), 0.01), median(d.Ho(:)), median(min(d.Ho, [], 1)));
end

fprintf('\n======== track boundary h at the final state ========\n');
fprintf('%-24s %-11s %-11s %-11s %-11s\n', 'row', 'min', 'p1', 'median', 'worst/traj');
for i = 1:nv
    d = Dq{i};
    fprintf('%-24s %-11.5f %-11.5f %-11.5f %-11.5f\n', V{i,1}, min(d.Hb(:)), ...
        quantile(d.Hb(:), 0.01), median(d.Hb(:)), median(min(d.Hb, [], 1)));
end

draw_figure(V, Dq, tol, n_show, only);

S = struct('labels', {V(:,1)}, 'modes', {V(:,2)}, 'settings', {V(:,3)}, ...
    'metrics', {M}, 'pointwise', {Dq}, 'tolerance', tol, 'n_gen', n_gen, ...
    'terminal_filter', false, ...
    'source', 'Xiao et al., SafeDiffuser, arXiv 2306.00148, Thms 2-4, Eqs 13-14', ...
    'note', ['variant mechanisms ported onto the SafeFlow QP with the terminal ' ...
             'projection OFF; not reproductions -- no diffusion backbone, and ' ...
             'the forms of gamma(t) and w(t) are our choices']);
stem = 'SafeDiffuser_Baselines_All';
if ~isempty(only), stem = ['SafeDiffuser_Baselines_' strjoin(only, '_')]; end
save(fullfile('outputs', [stem '.mat']), 'S', '-v7.3');
fprintf('\nsaved outputs/SafeDiffuser_Baselines_All.mat\n');
end

% =====================================================================
function draw_figure(V, Dq, tol, n_show, only)
nv = size(V,1);
f = figure('Visible','off','Color','w','Position',[20 20 400*nv 460]);
tiledlayout(1, nv, 'Padding','compact', 'TileSpacing','compact');
for i = 1:nv
    d = Dq{i};  P = d.points;  inside = d.Ho < tol;
    ax = nexttile; hold(ax, 'on');
    draw_track_segment(d.segment, 'HandleVisibility','off');
    draw_obstacles(d.obstacle, 'HandleVisibility','off');
    for s = 1:min(n_show, size(P,2))
        plot(ax, P(:,s,1), P(:,s,2), '-', 'Color', [0 0.45 0.74 0.55], ...
            'LineWidth', 0.9, 'HandleVisibility','off');
    end
    % Every point inside an obstacle, across all trajectories, not only drawn ones.
    [kk, ss] = find(inside);
    if ~isempty(kk)
        xs = arrayfun(@(a,b) P(a,b,1), kk, ss);
        ys = arrayfun(@(a,b) P(a,b,2), kk, ss);
        scatter(ax, xs, ys, 9, [0.85 0.10 0.10], 'filled', 'HandleVisibility','off');
    end
    axis(ax, 'equal'); grid(ax, 'on'); box(ax, 'on');
    title(ax, sprintf('%s\ninside obstacles: %.2f%% of points, worst h = %+.4f', ...
        V{i,1}, 100*mean(inside(:)), min(d.Ho(:))), 'FontSize', 9);
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 9);
end
sgtitle(['Generated trajectories, terminal projection (32) OFF; ' ...
    'red dots are points inside an obstacle'], 'FontSize', 11);
stem = 'SafeDiffuser_Baselines_Trajectories';
if ~isempty(only), stem = ['SafeDiffuser_Trajectories_' strjoin(only, '_')]; end
emf = fullfile('outputs', [stem '.emf']);
png = fullfile('outputs', [stem '.png']);
print(f, emf, '-dmeta', '-vector');
exportgraphics(f, png, 'Resolution', 130);
close(f);
fprintf('\nwrote %s\n     %s\n', emf, png);
end

% =====================================================================
function d = pointwise(r, net) %#ok<INUSD>
P = r.points; [nPt, ng, ~] = size(P);
Ho = zeros(nPt, ng); Hb = zeros(nPt, ng);
for s = 1:ng
    for k = 1:nPt
        p = [P(k,s,1); P(k,s,2)];
        ho = inf; hb = inf;
        for jo = 1:size(r.obstacle.centers, 2)
            ho = min(ho, obstacle_level_and_gradient(p, r.obstacle, jo));
        end
        for bi = 1:2
            hb = min(hb, evaluate_track_implicit_field( ...
                r.geometry.implicit_fields, bi, p));
        end
        Ho(k,s) = ho; Hb(k,s) = hb;
    end
end
d = struct('Ho', Ho, 'Hb', Hb);
end

function a = merge(a, b)
fn = fieldnames(b);
for i = 1:numel(fn), a.(fn{i}) = b.(fn{i}); end
end
