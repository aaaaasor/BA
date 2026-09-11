function archive_pcfm_run(n_gen, qp_backend)
%ARCHIVE_PCFM_RUN  Run PCFM and archive it reproducibly.
%
% Produces outputs/<track>PCFM with the same contents the other baselines get:
% network, rollout, metrics, per-trajectory seed table, KL reference, source
% snapshot, REPRODUCE.md, run_reproduce.m, SHA256SUMS.csv, and diagnostics.
%
% The archive replays from its own recorded options, which is asserted here
% before the run is accepted.
%
% PCFM keeps its terminal projection (Eq. 7) ON.  That is not a free choice:
% the projection is what makes the constraint hard, and the paper's guarantee
% rests on it.  The SafeDiffuser variants archived alongside have their
% projection OFF because their paper has no such step.  Safety is therefore
% comparable between PCFM and SafeFlow, but not between PCFM and RoSD/ReSD/TVSD.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_gen), n_gen = 100; end
car = char([36187 36710]);           % fm / safeflow archives
trk = char([36187 36947]);           % variant archives

src = fullfile('outputs', [car 'safeflow_tbar05_paper']);
if ~exist(src, 'dir'), src = fullfile('outputs', [car 'safeflow']); end
N = load(fullfile(src, 'Racing_SafeFlow_NN_Net.mat'));
A = load(fullfile(src, 'Racing_SafeFlow_NN_Rollout.mat'));
af = fieldnames(A); arch = A.(af{1});
net = N.net;  n_train_steps = N.n_train_steps;
cfg = get_config();

if nargin < 2 || isempty(qp_backend), qp_backend = 'closed'; end
opts = struct('trajectory_seeds', arch.trajectory_seeds, 'record_path', true, ...
    'qp_backend', qp_backend);

fprintf('===== PCFM =====\n');
r = pcfm_nn_rollout(net, n_gen, opts);
r.metrics = safeflow_nn_metrics(r);
m = r.metrics;
fprintf(['  Safety %6.2f%%  KL %.4f  CS %.4f  AS %.4g  Time %.4f\n' ...
         '  projections %d, moved %d (%.1f%%), infeasible %d (%.3f%%)\n' ...
         '  terminal: moved %d points, failed %d\n'], ...
    m.safety*100, m.kl, m.cs, m.as, m.time_seconds, ...
    r.n_qp, r.projections_moved, 100*r.projections_moved/r.n_qp, ...
    r.infeasible_hard, 100*r.infeasible_hard/r.n_qp, ...
    r.terminal_corrected, r.terminal_failed);

R = struct('pcfm', r);
specs = { 'pcfm', [trk 'PCFM'], 'PCFM_Mech', 'PCFM', ...
          'r = pcfm_nn_rollout(S.net, %d, opts);' };
archive_safeflow_nn_run(R, net, cfg, n_train_steps, n_gen, {'pcfm'}, specs);

d = fullfile('outputs', [trk 'PCFM']);

% ---- replay from the archived options ----
o2 = rmfield(opts, 'record_path');
r2 = pcfm_nn_rollout(net, n_gen, o2);
B  = load(fullfile(d, 'PCFM_Mech_Rollout.mat'));
dmax = max(abs(r2.points(:) - B.rollout.points(:)));
fprintf('\narchive replay from its own options: max|diff| = %.3e\n', dmax);
assert(dmax < 1e-12, 'the archived PCFM run does not replay from its options');

add_diagnostics(d, 'PCFM_Mech', 'PCFM', r, net);
write_archive_sha256(d);
fprintf('archived to %s\n', d);
end

% =====================================================================
function add_diagnostics(d, prefix, name, r, net)
% Obstacle and boundary h are kept apart: the obstacle level set bottoms out at
% -1 at its centre, the boundary field is a clamped bilinear grid reaching only
% about -0.06, so a min over both is not a comparable quantity.
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
ylabel(ax,'obstacle h_{min}(t)'); title(ax,'Obstacle level-set value');

ax = nexttile; hold(ax,'on');
plot(ax, t, Hb, 'Color', [0 0.45 0.74 0.25], 'LineWidth', 0.6);
yline(ax, 0, '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.2);
grid(ax,'on'); box(ax,'on'); xlabel(ax,'Generation time t');
ylabel(ax,'boundary h_{min}(t)'); title(ax,'Track boundary value');

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

sgtitle(sprintf(['%s  |  terminal projection (7) ON  |  ' ...
    'obstacle safe %.2f%%, boundary safe %.2f%%'], name, ...
    100*mean(HoF(:) >= tol), 100*mean(HbF(:) >= tol)));
emf = fullfile(d, [prefix '_Diagnostics.emf']);
print(f, emf, '-dmeta', '-vector');
close(f);

diag = struct('h_obstacle_over_time', Ho, 'h_boundary_over_time', Hb, 't', t, ...
    'h_obstacle_final', HoF, 'h_boundary_final', HbF, 'tolerance', tol, ...
    'obstacle_safe_fraction', mean(HoF(:) >= tol), ...
    'boundary_safe_fraction', mean(HbF(:) >= tol), ...
    'projections', r.n_qp, 'projections_moved', r.projections_moved, ...
    'projections_infeasible', r.infeasible_hard, ...
    'terminal_corrected', r.terminal_corrected, ...
    'terminal_failed', r.terminal_failed, 'options', r.options);
save(fullfile(d, [prefix '_Diagnostics.mat']), 'diag', '-v7.3');
fprintf('  wrote %s and the matching .mat\n', emf);
end

function [Ho, Hb] = h_over_time(r, net)
[n_t, Dd, ng] = size(r.z_path);  nF = 4;  nP = Dd/nF;
Ho = zeros(n_t, ng);  Hb = zeros(n_t, ng);
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
