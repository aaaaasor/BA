function exp_safeflow_hard_qp_gain(n_gen)
%EXP_SAFEFLOW_HARD_QP_GAIN  Is phi*dt > 1 sufficient once slack is removed?
%
% The early-coefficient sweep showed phi*dt = 290 without divergence, because
% (30)'s slack variable absorbs the infeasible demand: the gain is requested
% but never executed.  phi*dt > 1 is therefore necessary, not sufficient --
% the constraint must also be active as an equality.
%
% This crosses the two knobs.  If the large-gain runs diverge only in the
% no-slack column, the causal chain is closed: divergence needs a large
% phi*dt AND an enforced constraint.
%
% Divergence criterion is the project's usual one: a single-step physical
% displacement above 1 m, or a non-finite state.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_gen), n_gen = 100; end
car = char([36187 36710]);
d   = fullfile('outputs', [car 'safeflow']);
N   = load(fullfile(d, 'Racing_SafeFlow_NN_Net.mat'));
A   = load(fullfile(d, 'Racing_SafeFlow_NN_Rollout.mat'));
af  = fieldnames(A); arch = A.(af{1});

cs   = [4, 400, 4000, 40000];
slk  = [true, false];
dt   = 0.996/100; gam = 0.9;
M    = cell(numel(cs), numel(slk));

for j = 1:numel(slk)
    for i = 1:numel(cs)
        fprintf('\n-- c = %g, slack %s  (phi*dt at t=0.9 = %.3f) --\n', ...
            cs(i), tern(slk(j),'on','OFF'), (1+cs(i)*gam^3)*dt);
        r = safeflow_nn_rollout(N.net, 'safeflow', n_gen, struct( ...
            'trajectory_seeds', arch.trajectory_seeds, ...
            'u_per_stage', false, 'fmincon_fallback', false, ...
            'record_path', true, 'activation_time', 0, ...
            'phi1_early_coeff', cs(i), 'slack_enabled', slk(j)));
        m = safeflow_run_metrics(r, N.net);
        M{i,j} = m;
        div = ~isfinite(m.max_step_disp) || m.max_step_disp > 1;
        fprintf(['   Safety %6.2f%%  KL %.4f  |u|max %.3e\n' ...
                 '   max single-step move %.4f m  %s\n' ...
                 '   slack/infeasible %d of %d QP (%.1f%%)\n'], ...
            m.safety*100, m.kl, m.u_max, m.max_step_disp, ...
            tern(div, '*** DIVERGED ***', ''), ...
            m.slack_active, m.n_qp, 100*m.slack_active/max(m.n_qp,1));
    end
end

fprintf('\n\n======== max single-step displacement (m) ========\n');
fprintf('%-10s %-14s %-14s %-16s\n', 'c', 'phi*dt', 'slack on', 'slack OFF');
for i = 1:numel(cs)
    fprintf('%-10g %-14.3f %-14.4f %-16.4f\n', cs(i), (1+cs(i)*gam^3)*dt, ...
        M{i,1}.max_step_disp, M{i,2}.max_step_disp);
end
fprintf('\n======== Safety (%%) / KL ========\n');
fprintf('%-10s %-18s %-18s\n', 'c', 'slack on', 'slack OFF');
for i = 1:numel(cs)
    fprintf('%-10g %6.2f / %-10.4f %6.2f / %-10.4f\n', cs(i), ...
        M{i,1}.safety*100, M{i,1}.kl, M{i,2}.safety*100, M{i,2}.kl);
end

S = struct('early_coeffs', cs, 'slack_enabled', slk, 'metrics', {M}, ...
    'n_gen', n_gen, 'dt', dt, 'gamma', gam);
save(fullfile('outputs', 'SafeFlow_HardQP_Gain_Cross.mat'), 'S', '-v7.3');
fprintf('\nsaved outputs/SafeFlow_HardQP_Gain_Cross.mat\n');
end

function v = tern(c, a, b), if c, v = a; else, v = b; end, end
