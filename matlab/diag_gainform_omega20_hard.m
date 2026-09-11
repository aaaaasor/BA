function diag_gainform_omega20_hard(omega, n_gen)
%DIAG_GAINFORM_OMEGA20_HARD  Diagnostics for the diverging gain-form ablation.
%
% Configuration: guidance from t = 0, the second-order pole omega/(1-t)^2 over
% the whole rollout (gamma = 0), and NO slack -- the QP is the hard min-norm
% problem, so the demanded correction is actually executed.
%
% This is a gain-form ablation, NOT SafeFlow.  The paper's (42) uses the
% first-order pole 1/(T-t) and (30) always carries slack; both are replaced
% here to isolate what makes a prescribed-time CBF diverge.
%
% Obstacle and boundary h are reported separately on purpose: the boundary
% field is clamped to its grid and saturates near -1, while the obstacle level
% set keeps decreasing, so a min over both is not a meaningful quantity.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(omega), omega = 20; end
if nargin < 2 || isempty(n_gen), n_gen = 100; end
car = char([36187 36710]);
src = fullfile('outputs', [car 'safeflow_tbar05_paper']);
if ~exist(src, 'dir'), src = fullfile('outputs', [car 'safeflow']); end
N = load(fullfile(src, 'Racing_SafeFlow_NN_Net.mat'));
A = load(fullfile(src, 'Racing_SafeFlow_NN_Rollout.mat'));
af = fieldnames(A); arch = A.(af{1});

out = fullfile('outputs', sprintf('GainForm_Omega%g_Hard', omega));
if ~exist(out, 'dir'), mkdir(out); end
dt = 0.996/100;
t_unstable = max(1 - sqrt(omega*dt), 0);
fprintf('omega = %g, hard QP, pole from t = 0\n', omega);
fprintf('phi*dt exceeds 1 from t = %.3f onward\n\n', t_unstable);

ws1 = warning('off','MATLAB:nearlySingularMatrix');
ws2 = warning('off','MATLAB:singularMatrix');
r = safeflow_nn_rollout(N.net, 'safeflow', n_gen, struct( ...
    'trajectory_seeds', arch.trajectory_seeds, 'u_per_stage', false, ...
    'fmincon_fallback', false, 'record_path', true, 'activation_time', 0, ...
    'phi1_switch_time', 0, 'phi1_form', 'second_order', ...
    'phi1_omega', omega, 'slack_enabled', false));
warning(ws1); warning(ws2);

m = safeflow_run_metrics(r, N.net);
[~, D] = safeflow_path_stats(r, N.net);          % (step-1 x trajectory)
ns = size(D,1); tD = (0:ns-1)' * r.t_max / ns;

fprintf('=== metrics ===\n');
fprintf('  Safety %.2f%%  KL %.4f  CS %.4f  AS %.4f  terminal failed %d\n', ...
    m.safety*100, m.kl, m.cs, m.as, m.terminal_failed);

v = D(:);
fprintf('\n=== single-step displacement (m), %d point-steps ===\n', numel(v));
fprintf('  median %.5f  p99 %.5f  p99.9 %.5f  max %.4g\n', ...
    median(v), quantile(v,0.99), quantile(v,0.999), max(v));
for th = [0.05 0.1 1 100]
    fprintf('  > %-6g m: %6d steps (%6.2f%%) in %3d of %d trajectories\n', ...
        th, sum(v>th), 100*mean(v>th), sum(any(D>th,1)), size(D,2));
end

% When does each trajectory first blow past 1 m?  This is the quantity that
% says whether the front half is genuinely unstable.
first = nan(size(D,2),1);
for s = 1:size(D,2)
    k = find(D(:,s) > 1, 1);
    if ~isempty(k), first(s) = tD(k); end
end
nd = sum(~isnan(first));
fprintf('\n=== onset of divergence (first step > 1 m) ===\n');
fprintf('  %d of %d trajectories diverge\n', nd, size(D,2));
if nd > 0
    fprintf('  onset time: min %.3f  median %.3f  max %.3f\n', ...
        min(first), median(first,'omitnan'), max(first));
    fprintf('  diverging before t=0.5: %d | before the phi*dt=1 line (%.3f): %d\n', ...
        sum(first < 0.5), t_unstable, sum(first < t_unstable));
end

% h over time, obstacle and boundary kept apart.
[Ho, Hb] = h_curves(r, N.net);
tH = linspace(0, r.t_max, size(Ho,1))';
fprintf('\n=== h_min over time (median across trajectories) ===\n');
fprintf('%-8s %-14s %-14s\n', 't', 'obstacle', 'boundary');
for k = round(linspace(1, numel(tH), 6))
    fprintf('%-8.3f %-14.4f %-14.4f\n', tH(k), median(Ho(k,:)), median(Hb(k,:)));
end

f = figure('Visible','off','Color','w','Position',[40 40 1250 430], ...
    'Renderer','painters');
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

ax = nexttile; hold(ax,'on');
plot(ax, tD, D, 'Color', [0 0.45 0.74 0.25], 'LineWidth', 0.6);
yline(ax, 1, '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.2);
xline(ax, t_unstable, ':', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.4);
set(ax,'YScale','log'); grid(ax,'on'); box(ax,'on');
xlabel(ax,'Generation time t'); ylabel(ax,'max single-step move (m)');
title(ax, sprintf('Step displacement (dotted: phi\Deltat=1 at t=%.2f)', t_unstable));

ax = nexttile; hold(ax,'on');
plot(ax, tH, Ho, 'Color', [0 0.45 0.74 0.25], 'LineWidth', 0.6);
yline(ax, 0, '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.2);
grid(ax,'on'); box(ax,'on'); xlabel(ax,'Generation time t');
ylabel(ax,'obstacle h_{min}(t)'); title(ax,'Obstacle CBF value');

ax = nexttile; hold(ax,'on');
plot(ax, tH, Hb, 'Color', [0 0.45 0.74 0.25], 'LineWidth', 0.6);
yline(ax, 0, '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.2);
grid(ax,'on'); box(ax,'on'); xlabel(ax,'Generation time t');
ylabel(ax,'boundary h_{min}(t)'); title(ax,'Track boundary CBF value (grid-clamped)');

sgtitle(sprintf(['Gain-form ablation: second-order pole \omega/(1-t)^2, \omega = %g, ' ...
    'hard QP, active from t=0  (NOT SafeFlow)'], omega));
emf = fullfile(out, sprintf('GainForm_Omega%g_Hard_Diagnostics.emf', omega));
print(f, emf, '-dmeta', '-painters'); close(f);
fprintf('\nwrote %s\n', emf);

S = struct('omega', omega, 'hard', true, 'metrics', m, 'D', D, 't_D', tD, ...
    'h_obstacle', Ho, 'h_boundary', Hb, 't_h', tH, 'onset', first, ...
    't_unstable', t_unstable, 'n_gen', n_gen, 'trajectory_seeds', r.trajectory_seeds);
save(fullfile(out, sprintf('GainForm_Omega%g_Hard.mat', omega)), 'S', '-v7.3');
fprintf('wrote %s\n', fullfile(out, sprintf('GainForm_Omega%g_Hard.mat', omega)));
end

% =====================================================================
function [Ho, Hb] = h_curves(r, net)
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
