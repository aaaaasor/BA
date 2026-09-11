function exp_safeflow_gamma_variants(n_gen)
%EXP_SAFEFLOW_GAMMA_VARIANTS  Three readings of "activate the CBF from t = 0".
%
% The supervisor asked for the SafeFlow reproduction to constrain from the
% start instead of following the paper's schedule.  That admits three
% readings, which differ in whether the blow-up branch also starts at 0:
%
%   paper : activation 0.5, gamma 0.9   phi1 = 1+4t^3 then 1/(T-t)
%   A     : activation 0.0, gamma 0.9   guidance from the first step, paper gains
%   B     : activation 0.0, gamma 0.0   blow-up function 1/(T-t) over the whole
%                                        rollout
%   C     : activation 0.5, gamma 0.0   blow-up throughout, but only after 0.5
%
% Everything else is the archived paper setting: same network, same
% per-trajectory seeds, margin 0, no inflation, no fmincon fallback, one u per
% RK4 step.  The paper row is asserted to reproduce the archive bit-exactly.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_gen), n_gen = 100; end
car = char([36187 36710]);
d   = fullfile('outputs', [car 'safeflow_tbar05_paper']);
if ~exist(d, 'dir'), d = fullfile('outputs', [car 'safeflow']); end
N   = load(fullfile(d, 'Racing_SafeFlow_NN_Net.mat'));
A   = load(fullfile(d, 'Racing_SafeFlow_NN_Rollout.mat'));
af  = fieldnames(A); arch = A.(af{1});

V = { 'paper', 0.5, 0.9; ...
      'A',     0.0, 0.9; ...
      'B',     0.0, 0.0; ...
      'C',     0.5, 0.0 };
M = cell(size(V,1),1);
for i = 1:size(V,1)
    fprintf('\n===== %s: activation %.2f, gamma %.2f =====\n', V{i,1}, V{i,2}, V{i,3});
    r = safeflow_nn_rollout(N.net, 'safeflow', n_gen, struct( ...
        'trajectory_seeds', arch.trajectory_seeds, ...
        'u_per_stage', false, 'fmincon_fallback', false, 'record_path', true, ...
        'activation_time', V{i,2}, 'phi1_switch_time', V{i,3}));
    if strcmp(V{i,1}, 'paper')
        dmax = max(abs(r.points(:) - arch.points(:)));
        fprintf('  reproduces archive: max|diff| = %.3e\n', dmax);
        assert(dmax < 1e-12, 'paper row does not reproduce the archive');
    end
    m = safeflow_run_metrics(r, N.net);
    M{i} = m;
    fprintf(['  Safety %6.2f%%  KL %.4f  CS %.4f  AS %.4f  Time %.4f\n' ...
             '  terminal failed %d | slack %d of %d QP | |u| mean %.3e max %.3e\n' ...
             '  max single-step move %.4f m\n'], ...
        m.safety*100, m.kl, m.cs, m.as, m.time_seconds, m.terminal_failed, ...
        m.slack_active, m.n_qp, m.u_mean, m.u_max, m.max_step_disp);
end

fprintf('\n\n%-8s %-8s %-8s %-9s %-9s %-9s %-9s %-9s %-6s %-11s\n', ...
    'variant', 'act', 'gamma', 'Safety%', 'KL', 'CS', 'AS', 'Time', 'fail', 'maxStep(m)');
for i = 1:size(V,1)
    m = M{i};
    fprintf('%-8s %-8.2f %-8.2f %-9.2f %-9.4f %-9.4f %-9.4f %-9.4f %-6d %-11.4f\n', ...
        V{i,1}, V{i,2}, V{i,3}, m.safety*100, m.kl, m.cs, m.as, ...
        m.time_seconds, m.terminal_failed, m.max_step_disp);
end
S = struct('variants', {V}, 'metrics', {M}, 'n_gen', n_gen);
save(fullfile('outputs', 'SafeFlow_Gamma_Variants.mat'), 'S', '-v7.3');
fprintf('\nsaved outputs/SafeFlow_Gamma_Variants.mat\n');
end
