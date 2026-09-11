function exp_safeflow_slack_and_early_gain(n_gen)
%EXP_SAFEFLOW_SLACK_AND_EARLY_GAIN  Two questions about SafeFlow's front end.
%
% (A) What does the slack variable in (30) actually do?  Sweeping its weight
%     from the paper's 1 up to 1e6 prices it out; slack_enabled=false removes
%     it, leaving the hard min-norm QP  min|u|^2 s.t. a + B u >= 0.
%
% (B) Does a larger early-branch gain diverge?  The paper fixes phi1 = 1+4t^3
%     for t < gamma.  phi*dt at the worst point t->gamma=0.9 is
%     (1 + c*0.729)*dt, so with dt = 0.00996 the discrete-stability line
%     phi*dt = 1 is crossed at c = 136.  This checks that prediction.
%
% Both are run at activation_time 0, where the front end is actually active,
% and the paper row (c=4, slack weight 1, t_bar=0.5) is asserted to reproduce
% the archive bit-exactly first, so the two new options are proven inert.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_gen), n_gen = 100; end
car = char([36187 36710]);
d   = fullfile('outputs', [car 'safeflow']);
N   = load(fullfile(d, 'Racing_SafeFlow_NN_Net.mat'));
A   = load(fullfile(d, 'Racing_SafeFlow_NN_Rollout.mat'));
af  = fieldnames(A); arch = A.(af{1});
base = struct('trajectory_seeds', arch.trajectory_seeds, ...
              'u_per_stage', false, 'fmincon_fallback', false);

fprintf('=== control: paper settings must still reproduce the archive ===\n');
r0 = run_one(N.net, base, struct('activation_time', 0.5), n_gen);
dmax = max(abs(r0.r.points(:) - arch.points(:)));
fprintf('  max|diff| = %.3e\n', dmax);
assert(dmax < 1e-12, 'the new options changed the default behaviour');

dt = 0.996/100; gam = 0.9;
fprintf('\n=== (B) early-branch coefficient c in phi1 = 1 + c*t^3 ===\n');
fprintf('predicted stability line: c = %.1f  (phi*dt = 1 at t = %.2f)\n', ...
    (1/dt - 1)/gam^3, gam);
cs = [4, 40, 136, 400, 4000, 40000];
Bres = cell(numel(cs),1);
for i = 1:numel(cs)
    pdt = (1 + cs(i)*gam^3)*dt;
    fprintf('\n-- c = %g   (phi*dt at t=0.9 is %.3f) --\n', cs(i), pdt);
    Bres{i} = run_one(N.net, base, struct('activation_time', 0, ...
        'phi1_early_coeff', cs(i)), n_gen);
    report(Bres{i});
end

fprintf('\n=== (A) slack weight, and no slack at all ===\n');
ws = [1, 10, 100, 1e4, 1e6];
Ares = cell(numel(ws)+1,1);
for i = 1:numel(ws)
    fprintf('\n-- slack weight = %g --\n', ws(i));
    Ares{i} = run_one(N.net, base, struct('activation_time', 0, ...
        'slack_weight', ws(i)), n_gen);
    report(Ares{i});
end
fprintf('\n-- no slack (hard constraint) --\n');
Ares{end} = run_one(N.net, base, struct('activation_time', 0, ...
    'slack_enabled', false), n_gen);
report(Ares{end});

fprintf('\n\n============ (B) early gain, summary ============\n');
fprintf('%-9s %-9s %-9s %-9s %-9s %-9s %-11s %-11s\n', 'c', 'phi*dt', ...
    'Safety%', 'KL', 'CS', 'AS', '|u|max', 'maxStep(m)');
for i = 1:numel(cs)
    m = Bres{i}.m; pdt = (1 + cs(i)*gam^3)*dt;
    fprintf('%-9g %-9.3f %-9.2f %-9.4f %-9.4f %-9.4f %-11.3e %-11.4f\n', ...
        cs(i), pdt, m.safety*100, m.kl, m.cs, m.as, m.u_max, m.max_step_disp);
end
fprintf('\n============ (A) slack, summary ============\n');
fprintf('%-12s %-9s %-9s %-9s %-9s %-11s %-11s %-9s\n', 'slack', ...
    'Safety%', 'KL', 'CS', 'AS', '|u|max', 'maxStep(m)', 'infeas/QP%');
lbl = [arrayfun(@(w) sprintf('w=%g', w), ws, 'UniformOutput', false), {'none (hard)'}];
for i = 1:numel(Ares)
    m = Ares{i}.m;
    fprintf('%-12s %-9.2f %-9.4f %-9.4f %-9.4f %-11.3e %-11.4f %-9.1f\n', ...
        lbl{i}, m.safety*100, m.kl, m.cs, m.as, m.u_max, m.max_step_disp, ...
        100*m.slack_active/max(m.n_qp,1));
end

S = struct('early_coeffs', cs, 'early', {cellfun(@(x) x.m, Bres, 'UniformOutput', false)}, ...
    'slack_weights', ws, 'slack', {cellfun(@(x) x.m, Ares, 'UniformOutput', false)}, ...
    'slack_labels', {lbl}, 'n_gen', n_gen, 'dt', dt);
save(fullfile('outputs', 'SafeFlow_Slack_EarlyGain_Sweep.mat'), 'S', '-v7.3');
fprintf('\nsaved outputs/SafeFlow_Slack_EarlyGain_Sweep.mat\n');
end

% =====================================================================
function out = run_one(net, base, extra, n_gen)
o = base;
f = fieldnames(extra);
for i = 1:numel(f), o.(f{i}) = extra.(f{i}); end
o.record_path = true;
r = safeflow_nn_rollout(net, 'safeflow', n_gen, o);
out.r = r;
out.m = safeflow_run_metrics(r, net);
end

function report(x)
m = x.m;
fprintf(['   Safety %6.2f%%  KL %.4f  CS %.4f  AS %.4f  Time %.4f\n' ...
         '   |u| mean %.3e  max %.3e | slack/infeas %d of %d QP\n' ...
         '   worst h along path %+.5f | max single-step move %.4f m | terminal failed %d\n'], ...
    m.safety*100, m.kl, m.cs, m.as, m.time_seconds, m.u_mean, m.u_max, ...
    m.slack_active, m.n_qp, m.h_path_min, m.max_step_disp, m.terminal_failed);
if ~isfinite(m.max_step_disp) || m.max_step_disp > 1
    fprintf('   *** DIVERGED (step displacement > 1 m) ***\n');
end
end
