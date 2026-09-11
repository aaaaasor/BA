function archive_gainform_run(omega, n_gen)
%ARCHIVE_GAINFORM_RUN  Archive the diverging gain-form run into the safeflow dir.
%
% Configuration archived: guidance active from t = 0, the second-order pole
% omega/(1-t)^2 over the whole rollout (gamma = 0), and NO slack, so the QP is
% the hard min-norm problem and the demanded correction is actually executed.
%
% This is a gain-form ablation, NOT a SafeFlow reproduction: the paper's (42)
% uses the first-order pole 1/(T-t) and (30) always carries slack.  Both
% deviations are recorded in the archive's run_reproduce.m option struct, so
% the archive replays this exact configuration instead of the defaults.
%
% The FM archive is untouched: only the 'safeflow' spec is rebuilt.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(omega), omega = 20; end
if nargin < 2 || isempty(n_gen), n_gen = 100; end
car = char([36187 36710]);
src = fullfile('outputs', [car 'safeflow_tbar05_paper']);
if ~exist(src, 'dir'), src = fullfile('outputs', [car 'safeflow']); end
N = load(fullfile(src, 'Racing_SafeFlow_NN_Net.mat'));
A = load(fullfile(src, 'Racing_SafeFlow_NN_Rollout.mat'));
af = fieldnames(A); arch = A.(af{1});
n_train_steps = N.n_train_steps;

opts = struct('trajectory_seeds', arch.trajectory_seeds, ...
    'u_per_stage', false, 'fmincon_fallback', false, 'record_path', true, ...
    'activation_time', 0, 'phi1_switch_time', 0, ...
    'phi1_form', 'second_order', 'phi1_omega', omega, 'slack_enabled', false);

ws1 = warning('off', 'MATLAB:nearlySingularMatrix');
ws2 = warning('off', 'MATLAB:singularMatrix');
r = safeflow_nn_rollout(N.net, 'safeflow', n_gen, opts);
warning(ws1); warning(ws2);
r.metrics = safeflow_nn_metrics(r);
m = r.metrics;
fprintf('Safety %.2f%%  KL %.4f  CS %.4f  AS %.6g  terminal failed %d\n', ...
    m.safety*100, m.kl, m.cs, m.as, m.terminal_failed);

% Keep the archived FM entry as it is; only the safeflow directory is rebuilt.
S = load(fullfile('outputs', 'SafeFlowNN_results.mat'), 'R', 'cfg');
R = S.R;  R.safeflow = r;  cfg = S.cfg;  net = N.net;
save(fullfile('outputs', 'SafeFlowNN_results.mat'), 'R', 'cfg', ...
    'n_train_steps', 'n_gen', '-append');
archive_safeflow_nn_run(R, net, cfg, n_train_steps, n_gen, 'safeflow');

d = fullfile('outputs', [car 'safeflow']);

% ---- extra diagnostics into the same directory ----
[~, D] = safeflow_path_stats(r, net);
ns = size(D, 1);  tD = (0:ns-1)' * r.t_max / ns;
dt = r.t_max / r.n_rk_steps;
t_unstable = max(1 - sqrt(omega*dt), 0);
onset = nan(size(D, 2), 1);
for s = 1:size(D, 2)
    k = find(D(:, s) > 1, 1);
    if ~isempty(k), onset(s) = tD(k); end
end
[Ho, Hb] = h_curves(r, net);
tH = linspace(0, r.t_max, size(Ho, 1))';

diag = struct('omega', omega, 'options', opts, 'step_displacement', D, ...
    't_step', tD, 'h_obstacle', Ho, 'h_boundary', Hb, 't_h', tH, ...
    'divergence_onset', onset, 'phi_dt_unity_time', t_unstable, ...
    'displacement_quantiles', [median(D(:)), quantile(D(:), [0.99 0.999]), max(D(:))], ...
    'n_diverged_trajectories', sum(~isnan(onset)), ...
    'note', ['gain-form ablation: second-order pole omega/(1-t)^2 from t=0, ' ...
             'hard QP (no slack). NOT the SafeFlow paper configuration.']);
save(fullfile(d, 'Racing_SafeFlow_NN_Diagnostics.mat'), 'diag', '-v7.3');

make_diag_figure(d, D, tD, Ho, Hb, tH, omega, t_unstable);

fprintf('\n=== step displacement (m) ===\n');
fprintf('  median %.5f  p99 %.4g  p99.9 %.4g  max %.4g\n', ...
    median(D(:)), quantile(D(:), 0.99), quantile(D(:), 0.999), max(D(:)));
fprintf('  %d of %d trajectories exceed 1 m; onset median t = %.3f, %d before t=0.5\n', ...
    sum(~isnan(onset)), size(D, 2), median(onset, 'omitnan'), sum(onset < 0.5));

% ---- reproduction check on the freshly written archive ----
B = load(fullfile(d, 'Racing_SafeFlow_NN_Rollout.mat'));
o2 = rmfield(opts, 'record_path');
ws1 = warning('off', 'MATLAB:nearlySingularMatrix');
ws2 = warning('off', 'MATLAB:singularMatrix');
r2 = safeflow_nn_rollout(net, 'safeflow', n_gen, o2);
warning(ws1); warning(ws2);
dmax = max(abs(r2.points(:) - B.rollout.points(:)));
fprintf('\narchive replay from its own options: max|diff| = %.3e\n', dmax);
assert(dmax < 1e-12, 'the archived run does not replay from its own options');

write_archive_sha256(d);
fprintf('archived to %s\n', d);
end

% =====================================================================
function make_diag_figure(d, D, tD, Ho, Hb, tH, omega, t_unstable)
f = figure('Visible', 'off', 'Color', 'w', 'Position', [40 40 1250 430]);
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

ax = nexttile; hold(ax, 'on');
plot(ax, tD, D, 'Color', [0 0.45 0.74 0.25], 'LineWidth', 0.6);
yline(ax, 1, '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.2);
xline(ax, t_unstable, ':', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.4);
set(ax, 'YScale', 'log'); grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'Generation time t'); ylabel(ax, 'max single-step move (m)');
title(ax, sprintf('Step displacement (dotted: phi*dt = 1 at t = %.2f)', t_unstable));

ax = nexttile; hold(ax, 'on');
plot(ax, tH, Ho, 'Color', [0 0.45 0.74 0.25], 'LineWidth', 0.6);
yline(ax, 0, '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.2);
grid(ax, 'on'); box(ax, 'on'); xlabel(ax, 'Generation time t');
ylabel(ax, 'obstacle h_{min}(t)'); title(ax, 'Obstacle CBF value');

ax = nexttile; hold(ax, 'on');
plot(ax, tH, Hb, 'Color', [0 0.45 0.74 0.25], 'LineWidth', 0.6);
yline(ax, 0, '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.2);
grid(ax, 'on'); box(ax, 'on'); xlabel(ax, 'Generation time t');
ylabel(ax, 'boundary h_{min}(t)');
title(ax, 'Track boundary CBF value (grid-clamped)');

sgtitle(sprintf(['Gain-form ablation: omega/(1-t)^2, omega = %g, hard QP, ' ...
    'active from t = 0  (NOT the SafeFlow paper setting)'], omega));
print(f, fullfile(d, 'Racing_SafeFlow_NN_Diagnostics.emf'), '-dmeta', '-vector');
close(f);
end

% =====================================================================
function [Ho, Hb] = h_curves(r, net)
% Obstacle and boundary are kept apart on purpose.  The obstacle level set
% h = (s-c)'Q(s-c) - 1 has a floor of -1 at the centre; the boundary field is a
% bilinear grid clamped at its edges and reaches only about -0.06 in practice.
% Different scales, so a min over both is not a meaningful quantity.
[n_t, Dd, ng] = size(r.z_path);  nF = 4;  nP = Dd / nF;
Ho = zeros(n_t, ng);  Hb = zeros(n_t, ng);
for k = 1:n_t
    X = squeeze(r.z_path(k, :, :)) .* net.sd_d + net.mu_d;
    P = permute(reshape(X, nF, nP, ng), [2 3 1]);
    for s = 1:ng
        ho = inf;  hb = inf;
        for ip = 1:nP
            p = [P(ip, s, 1); P(ip, s, 2)];
            for jo = 1:size(r.obstacle.centers, 2)
                ho = min(ho, obstacle_level_and_gradient(p, r.obstacle, jo));
            end
            for bi = 1:2
                hb = min(hb, evaluate_track_implicit_field( ...
                    r.geometry.implicit_fields, bi, p));
            end
        end
        Ho(k, s) = ho;  Hb(k, s) = hb;
    end
end
end
