function T = exp_safeflow_activation_time(t_bars, n_gen)
%EXP_SAFEFLOW_ACTIVATION_TIME  Effect of the guidance activation time t_bar.
%
% The paper's navigation experiment starts the CBF-QP guidance at t_bar = 0.5;
% the archived SafeFlow run uses that value.  This sweeps t_bar down to 0, so
% the QP is active from the very first RK4 step, and reports what changes.
%
% Everything else is held at the archived paper settings: same network, same
% per-trajectory seeds, margin 0, no obstacle inflation, no fmincon fallback,
% one u per RK4 step, phi1_switch_time 0.9, phi0 1.
%
% The t_bar = 0.5 row is asserted to reproduce the archived terminal state
% bit-exactly, so any difference in the other rows is caused by t_bar alone.
%
% Metrics are computed with the same code path as safeflow_nn_demo
% (safety tolerance -1e-8 on the true boundary, no margin).

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(t_bars), t_bars = [0, 0.1, 0.25, 0.5]; end
if nargin < 2 || isempty(n_gen),  n_gen  = 100; end

car = char([36187 36710]);
d   = fullfile('outputs', [car 'safeflow']);
N   = load(fullfile(d, 'Racing_SafeFlow_NN_Net.mat'));
A   = load(fullfile(d, 'Racing_SafeFlow_NN_Rollout.mat'));
af  = fieldnames(A); arch = A.(af{1});

nb = numel(t_bars);
rows = cell(nb, 1);
R    = cell(nb, 1);
for i = 1:nb
    tb = t_bars(i);
    fprintf('\n===== activation_time = %.2f =====\n', tb);
    r = safeflow_nn_rollout(N.net, 'safeflow', n_gen, struct( ...
        'activation_time', tb, ...
        'trajectory_seeds', arch.trajectory_seeds, ...
        'u_per_stage', false, 'fmincon_fallback', false, ...
        'record_path', true));
    if abs(tb - 0.5) < 1e-12
        dmax = max(abs(r.points(:) - arch.points(:)));
        fprintf('  reproduces archive: max|diff| = %.3e\n', dmax);
        assert(dmax < 1e-12, 'baseline t_bar=0.5 does not reproduce the archive');
    end
    m = safeflow_run_metrics(r, N.net);
    R{i} = r;
    rows{i} = m;
    fprintf(['  Safety %6.2f%%  KL %.4f  CS %.4f  AS %.4f  Time %.4f\n' ...
             '  QP %d, slack %d, terminal failed %d\n' ...
             '  |u| mean %.3e  p95 %.3e  max %.3e\n' ...
             '  h_min over the whole path: %+.5f (worst point, any step)\n' ...
             '  max single-step physical displacement: %.4f m\n'], ...
        m.safety*100, m.kl, m.cs, m.as, m.time_seconds, ...
        m.n_qp, m.slack_active, m.terminal_failed, ...
        m.u_mean, m.u_p95, m.u_max, m.h_path_min, m.max_step_disp);
end

T = struct2table([rows{:}]);
T = addvars(T, t_bars(:), 'Before', 1, 'NewVariableNames', 'ActivationTime');
out = fullfile('outputs', 'SafeFlow_ActivationTime_Sweep.csv');
writetable(T(:, {'ActivationTime','safety','kl','cs','as','time_seconds', ...
    'n_qp','slack_active','terminal_failed','u_mean','u_p95','u_max', ...
    'h_path_min','max_step_disp'}), out);
fprintf('\nwrote %s\n', out);

fprintf('\n%-8s %-9s %-9s %-9s %-9s %-9s %-7s %-7s %-11s %-11s\n', ...
    't_bar', 'Safety%', 'KL', 'CS', 'AS', 'Time', 'QP', 'fail', '|u|max', 'h_path_min');
for i = 1:nb
    m = rows{i};
    fprintf('%-8.2f %-9.2f %-9.4f %-9.4f %-9.4f %-9.4f %-7d %-7d %-11.3e %+-11.5f\n', ...
        t_bars(i), m.safety*100, m.kl, m.cs, m.as, m.time_seconds, ...
        m.n_qp, m.terminal_failed, m.u_max, m.h_path_min);
end

S = struct('t_bars', t_bars, 'rows', {rows}, 'n_gen', n_gen);
save(fullfile('outputs', 'SafeFlow_ActivationTime_Sweep.mat'), 'S', '-v7.3');

% h_min(t) for every t_bar, one panel each, so the front end is visible.
plot_paths(R, t_bars, N.net, fullfile('outputs', 'SafeFlow_ActivationTime_hmin.emf'));
end

% =====================================================================
function plot_paths(R, t_bars, net, emf)
nb = numel(R);
f = figure('Visible','off','Color','w','Position',[40 40 420*nb 420], ...
    'Renderer','painters');
tiledlayout(1, nb, 'Padding','compact', 'TileSpacing','compact');
lo = inf; hi = -inf; Hs = cell(nb,1);
for i = 1:nb
    Hs{i} = safeflow_path_stats(R{i}, net);
    lo = min(lo, min(Hs{i}(:))); hi = max(hi, max(Hs{i}(:)));
end
for i = 1:nb
    ax = nexttile; hold(ax,'on');
    times = linspace(0, R{i}.t_max, size(Hs{i},1));
    plot(ax, times, Hs{i}, 'Color', [0.00 0.45 0.74], 'LineWidth', 0.7);
    yline(ax, 0, '--', 'Color', [0.85 0.20 0.20], 'LineWidth', 1.2);
    xline(ax, t_bars(i), ':', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.2);
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'t'); ylabel(ax,'h_{min}(t)');
    title(ax, sprintf('activation time = %.2f', t_bars(i)));
    ylim(ax, [lo - 0.05*(hi-lo), hi + 0.05*(hi-lo)]);
    set(ax,'FontName','Times New Roman','FontSize',10);
end
sgtitle('SafeFlow: worst CBF value along the rollout, by guidance activation time');
print(f, emf, '-dmeta', '-painters'); close(f);
fprintf('wrote %s\n', emf);
end
