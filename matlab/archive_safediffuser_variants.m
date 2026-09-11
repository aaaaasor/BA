function archive_safediffuser_variants(n_gen, which, qp_backend)
%ARCHIVE_SAFEDIFFUSER_VARIANTS  Run RoSD/ReSD/TVSD and archive each one.
%
% Produces the same reproducible directory the two headline runs get:
%   <prefix>_Net.mat, _Rollout.mat, _Metrics.mat, _Trajectory_Seeds.csv/.mat,
%   _KL_Reference.mat, REPRODUCE.md, GIT_STATE.txt, MATLAB_ENVIRONMENT.txt,
%   run_reproduce.m, source_snapshot/, SHA256SUMS.csv, and the diagnostic EMFs.
%
% run_reproduce.m carries the full option struct, so each archive replays its
% own configuration rather than the rollout defaults -- that is checked here
% before the run is accepted.
%
% Mechanisms, from the SafeDiffuser paper (Xiao et al., arXiv 2306.00148),
% Sec. 3.1-3.3 and Sec. 4.  None of them changes the class-K gain; each changes
% the constraint, and all three share alpha(b) = b and guidance from t = 0.
%
%   RoSD  Thm 2 / Eq. 8, solved by the hard QP of Eq. 13.
%   ReSD  Thm 3 / Eq. 9-10, solved by Eq. 14: relaxation r with weight w(t)
%         falling to 0, which is our penalty form with slack weight 1/w(t)^2.
%         The paper's N_a extra steps are omitted -- they rely on the diffusion
%         noise redrawing across b = 0, which a deterministic flow does not have.
%   TVSD  Thm 4 / Eq. 11-12, solved by the same hard QP of Eq. 13.  Two rows:
%
%         The -gamma_dot term adds a constant POSITIVE rate demand to every
%         constraint row.  The two track-boundary rows have opposing gradients
%         (h_left rises exactly as h_right falls), so both can only hold when the
%         demand is non-positive:  gamma_dot(t) <= alpha * s(t),  s = h - gamma.
%         Near t = 1, gamma -> 0 and s -> h ~ 0.011 (the track is narrow), so
%         gamma_dot must vanish there too.  The critical alpha, max_t
%         gamma_dot/s, is 66.7 for a straight line but only 8.8 for a sigmoid of
%         steepness 10, which flattens at BOTH ends; the optimum over steepness
%         is 8.6 at k = 9.5.  A straight line is therefore the WORST choice here,
%         not the best -- minimising the peak of gamma_dot is the wrong criterion,
%         because what has to stay bounded is the ratio gamma_dot/s.
%
%         So TVSD alone uses alpha = 10, just above its critical 8.79; the other
%         rows keep alpha = 1.  At alpha = 1 this row is structurally infeasible
%         and 93.4%% of its QPs fell back to the least-violation solution.
%
%         gamma applies to the boundary rows as well as the obstacle rows.  A
%         single gamma_min = -1 covers both: measured at t = 0 the obstacle h
%         bottoms out at -0.9893 and the boundary h at -0.0590, so -1 satisfies
%         gamma(0) <= b for every row, if conservatively for the boundary (16.8x
%         more slack than it needs).  The paper writes gamma_k(j) per point and
%         per specification, so a per-row gamma_min would also be admissible and
%         would cut the boundary rows' gamma_dot by the same factor.
%         The paper does not state the form of gamma.
%
% The terminal projection (32) is OFF for all three: it belongs to SafeFlow, not
% to these variants, and leaving it on would let it supply the safety the CBF-QP
% failed to.  These archives are therefore NOT comparable to outputs/<racing>fm
% or outputs/<racing>safeflow, which keep it.
%
% Output directories follow the same naming as the two headline runs:
%   outputs/<track>RoSD, outputs/<track>ReSD, outputs/<track>TVSD, where
%   <track> is a different prefix from the <racing> one used by fm/safeflow.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_gen), n_gen = 100; end
if nargin < 2 || isempty(which), which = {'rosd','resd','tvsd'}; end
if ischar(which) || isstring(which), which = cellstr(which); end

car = char([36187 36710]);           % the fm / safeflow archives use this
trk = char([36187 36947]);           % the three variant archives use this one
% Two different prefixes are in use on disk; keep each pointing at what exists
% rather than renaming, so no archive is orphaned from its own directory.
src = fullfile('outputs', [car 'safeflow_tbar05_paper']);
if ~exist(src, 'dir'), src = fullfile('outputs', [car 'safeflow']); end
N = load(fullfile(src, 'Racing_SafeFlow_NN_Net.mat'));
A = load(fullfile(src, 'Racing_SafeFlow_NN_Rollout.mat'));
af = fieldnames(A); arch = A.(af{1});
net = N.net;  n_train_steps = N.n_train_steps;
cfg = get_config();

if nargin < 3 || isempty(qp_backend), qp_backend = 'closed'; end
base = struct('trajectory_seeds', arch.trajectory_seeds, 'u_per_stage', false, ...
    'fmincon_fallback', false, 'record_path', true, 'activation_time', 0, ...
    'terminal_filter', false, ...
    'phi1_form', 'constant', 'phi1_alpha', 1.0, 'phi1_switch_time', 0, ...
    'qp_backend', qp_backend);

DEF = { ...
 'rosd', [trk 'RoSD'], 'RoSD_Mech', 'RoSD-mech (hard QP)', ...
    struct('slack_enabled', false); ...
 'resd', [trk 'ReSD'], 'ReSD_Mech', 'ReSD-mech (w(t) relax)', ...
    struct('resd_weight_schedule', [100 0 0.9]); ...
 'tvsd', [trk 'TVSD'], 'TVSD_Mech', 'TVSD-mech (sigmoid, alpha=10)', ...
    struct('gamma_mode', 'sigmoid', 'gamma_min', -1.0, 'gamma_steepness', 10, ...
           'gamma_midpoint', 0.5, 'phi1_alpha', 10, 'phi0', 10, ...
           'slack_enabled', false) };

keep = ismember(DEF(:,1), lower(which));
assert(any(keep), 'which matched none of: %s', strjoin(DEF(:,1)', ', '));
DEF = DEF(keep, :);

R = struct();  OPTS = struct();
for i = 1:size(DEF,1)
    key = DEF{i,1};
    o = merge(base, DEF{i,5});
    fprintf('\n===== %s =====\n', DEF{i,4});
    ws1 = warning('off', 'MATLAB:nearlySingularMatrix');
    ws2 = warning('off', 'MATLAB:singularMatrix');
    r = safeflow_nn_rollout(net, 'safeflow', n_gen, o);
    warning(ws1); warning(ws2);
    r.metrics = safeflow_nn_metrics(r);
    m = r.metrics;
    fprintf(['  Safety %6.2f%%  KL %.4f  CS %.4f  AS %.4g  Time %.4f\n' ...
             '  infeas/slack %d of %d QP (%.2f%%) | |u|max %.3e\n'], ...
        m.safety*100, m.kl, m.cs, m.as, m.time_seconds, m.slack_active, ...
        m.n_qp, 100*m.slack_active/max(m.n_qp,1), umax(r));
    R.(key) = r;  OPTS.(key) = o;
end

specs = DEF(:, 1:4);
archive_safeflow_nn_run(R, net, cfg, n_train_steps, n_gen, DEF(:,1)', specs);

% ---- each archive must replay from its own recorded options ----
for i = 1:size(DEF,1)
    d = fullfile('outputs', DEF{i,2});
    B = load(fullfile(d, [DEF{i,3} '_Rollout.mat']));
    o2 = rmfield(OPTS.(DEF{i,1}), 'record_path');
    ws1 = warning('off', 'MATLAB:nearlySingularMatrix');
    ws2 = warning('off', 'MATLAB:singularMatrix');
    r2 = safeflow_nn_rollout(net, 'safeflow', n_gen, o2);
    warning(ws1); warning(ws2);
    dmax = max(abs(r2.points(:) - B.rollout.points(:)));
    fprintf('%-24s archive replay: max|diff| = %.3e\n', DEF{i,4}, dmax);
    assert(dmax < 1e-12, '%s does not replay from its archived options', DEF{i,4});
    add_diagnostics(d, DEF{i,3}, DEF{i,4}, R.(DEF{i,1}), net);
end
fprintf('\ndone: %s\n', strjoin(DEF(:,2)', ', '));
end

% =====================================================================
function add_diagnostics(d, prefix, name, r, net)
% Obstacle and boundary h are kept apart: the obstacle level set bottoms out at
% -1 at its centre while the boundary field is a clamped bilinear grid, so a min
% over both is not a comparable quantity.
[Ho, Hb] = h_over_time(r, net);
t = linspace(0, r.t_max, size(Ho,1))';
[HoF, HbF] = final_h(r);
tol = -1e-8;

f = figure('Visible','off','Color','w','Position',[40 40 1250 430]);
tiledlayout(1, 3, 'Padding','compact', 'TileSpacing','compact');

ax = nexttile; hold(ax,'on');
plot(ax, t, Ho, 'Color', [0 0.45 0.74 0.25], 'LineWidth', 0.6);
yline(ax, 0, '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.2);
grid(ax,'on'); box(ax,'on'); xlabel(ax,'Generation time t');
ylabel(ax,'obstacle h_{min}(t)'); title(ax,'Obstacle CBF value');

ax = nexttile; hold(ax,'on');
plot(ax, t, Hb, 'Color', [0 0.45 0.74 0.25], 'LineWidth', 0.6);
yline(ax, 0, '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.2);
grid(ax,'on'); box(ax,'on'); xlabel(ax,'Generation time t');
ylabel(ax,'boundary h_{min}(t)'); title(ax,'Track boundary CBF value');

ax = nexttile; hold(ax,'on');
draw_track_segment(r.segment, 'HandleVisibility','off');
draw_obstacles(r.obstacle, 'HandleVisibility','off');
P = r.points;
for s = 1:min(20, size(P,2))
    plot(ax, P(:,s,1), P(:,s,2), '-', 'Color', [0 0.45 0.74 0.55], 'LineWidth', 0.9);
end
[kk, ss] = find(HoF < tol);
if ~isempty(kk)
    scatter(ax, arrayfun(@(a,b) P(a,b,1), kk, ss), ...
                arrayfun(@(a,b) P(a,b,2), kk, ss), 9, [0.85 0.10 0.10], 'filled');
end
axis(ax,'equal'); grid(ax,'on'); box(ax,'on');
title(ax, sprintf('inside obstacles: %.2f%% of points', 100*mean(HoF(:) < tol)));

sgtitle(sprintf(['%s  |  terminal projection (32) OFF  |  ' ...
    'obstacle safe %.2f%%, boundary safe %.2f%%'], name, ...
    100*mean(HoF(:) >= tol), 100*mean(HbF(:) >= tol)));
emf = fullfile(d, [prefix '_Diagnostics.emf']);
print(f, emf, '-dmeta', '-vector');
close(f);

diag = struct('h_obstacle_over_time', Ho, 'h_boundary_over_time', Hb, 't', t, ...
    'h_obstacle_final', HoF, 'h_boundary_final', HbF, 'tolerance', tol, ...
    'obstacle_safe_fraction', mean(HoF(:) >= tol), ...
    'boundary_safe_fraction', mean(HbF(:) >= tol), 'options', r.options);
save(fullfile(d, [prefix '_Diagnostics.mat']), 'diag', '-v7.3');
fprintf('  wrote %s and the matching .mat\n', emf);
end

function [Ho, Hb] = h_over_time(r, net)
[n_t, Dd, ng] = size(r.z_path); nF = 4; nP = Dd/nF;
Ho = zeros(n_t, ng); Hb = zeros(n_t, ng);
for k = 1:n_t
    X = squeeze(r.z_path(k,:,:)) .* net.sd_d + net.mu_d;
    P = permute(reshape(X, nF, nP, ng), [2 3 1]);
    for s = 1:ng
        ho = inf; hb = inf;
        for ip = 1:nP
            p = [P(ip,s,1); P(ip,s,2)];
            for jo = 1:size(r.obstacle.centers,2)
                ho = min(ho, obstacle_level_and_gradient(p, r.obstacle, jo));
            end
            for bi = 1:2
                hb = min(hb, evaluate_track_implicit_field( ...
                    r.geometry.implicit_fields, bi, p));
            end
        end
        Ho(k,s) = ho; Hb(k,s) = hb;
    end
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

function u = umax(r)
if isempty(r.u_trace), u = 0; return; end
nu = sqrt(sum(double(r.u_trace).^2, 4));
u = max(nu(:));
end

function a = merge(a, b)
fn = fieldnames(b);
for i = 1:numel(fn), a.(fn{i}) = b.(fn{i}); end
end
