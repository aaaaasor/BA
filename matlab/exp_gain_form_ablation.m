function exp_gain_form_ablation(n_gen)
%EXP_GAIN_FORM_ABLATION  Pole order and slack price, inside one fixed QP.
%
% This is NOT a SafeFlow baseline.  SafeFlow's (42) only ever uses the
% first-order pole 1/(T-t); the second-order rows below substitute this
% project's own prescribed-time gain omega/(1-t)^2 into an otherwise identical
% QP, so that the two pole orders can be compared with everything else held
% fixed.  Labelling any of these rows "SafeFlow" would be a straw man.
%
% What it tests: divergence of a prescribed-time CBF needs BOTH a large
% phi*dt AND a constraint that is actually enforced.  The slack weight
% controls the second condition -- at the paper's w = 1 the QP prefers to pay
% the penalty rather than exert the demanded correction.
%
% phi*dt = omega*dt/(1-t)^2, so with dt = 0.00996 the discrete-stability line
% phi*dt = 1 is crossed at t = 0.25 for omega > 56, and at t = 0.5 for
% omega > 25.  Large omega should therefore destabilise the FRONT half.
%
% Guidance is active from t = 0 and the pole is used from t = 0 (gamma = 0),
% so the front end is genuinely exercised.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_gen), n_gen = 100; end
car = char([36187 36710]);
d   = fullfile('outputs', [car 'safeflow_tbar05_paper']);
if ~exist(d, 'dir'), d = fullfile('outputs', [car 'safeflow']); end
N   = load(fullfile(d, 'Racing_SafeFlow_NN_Net.mat'));
A   = load(fullfile(d, 'Racing_SafeFlow_NN_Rollout.mat'));
af  = fieldnames(A); arch = A.(af{1});

omegas = [0.5, 50, 500];
wslack = [1, 1e4];
dt = 0.996/100;
rows = {};
for wi = 1:numel(wslack)
    for oi = 1:numel(omegas)
        om = omegas(oi); w = wslack(wi);
        % first t at which phi*dt exceeds 1
        t_unstable = 1 - sqrt(om*dt);
        fprintf('\n===== omega = %g, slack weight = %g =====\n', om, w);
        fprintf('  phi*dt crosses 1 at t = %.3f\n', t_unstable);
        r = safeflow_nn_rollout(N.net, 'safeflow', n_gen, struct( ...
            'trajectory_seeds', arch.trajectory_seeds, ...
            'u_per_stage', false, 'fmincon_fallback', false, ...
            'record_path', true, 'activation_time', 0, ...
            'phi1_switch_time', 0, 'phi1_form', 'second_order', ...
            'phi1_omega', om, 'slack_weight', w));
        m = safeflow_run_metrics(r, N.net);
        [~, D] = safeflow_path_stats(r, N.net);      % (step-1 x trajectory)
        [mx, li] = max(D(:));
        [ks, ~] = ind2sub(size(D), li);
        t_at_max = (ks-1) * r.t_max / size(D,1);
        % where in time does the motion get large?
        per_step = max(D, [], 2);
        first_big = find(per_step > 0.05, 1);
        t_first_big = NaN;
        if ~isempty(first_big), t_first_big = (first_big-1)*r.t_max/size(D,1); end
        div = ~isfinite(mx) || mx > 1;
        fprintf(['  Safety %6.2f%%  KL %.4f  |u|max %.3e  terminal failed %d\n' ...
                 '  max single-step move %.4f m at t = %.3f  %s\n' ...
                 '  first step exceeding 0.05 m at t = %s\n'], ...
            m.safety*100, m.kl, m.u_max, m.terminal_failed, mx, t_at_max, ...
            tern(div,'*** DIVERGED ***',''), num2str(t_first_big));
        rows{end+1} = struct('omega', om, 'w', w, 'safety', m.safety, ...
            'kl', m.kl, 'umax', m.u_max, 'maxstep', mx, 't_at_max', t_at_max, ...
            't_first_big', t_first_big, 't_unstable', t_unstable, ...
            'diverged', div, 'terminal_failed', m.terminal_failed); %#ok<AGROW>
    end
end

fprintf('\n\n%-8s %-8s %-12s %-9s %-9s %-11s %-12s %-10s\n', 'omega', 'w', ...
    'unstable@t', 'Safety%', 'KL', 'maxStep(m)', 'at t', 'diverged');
for i = 1:numel(rows)
    R = rows{i};
    fprintf('%-8g %-8g %-12.3f %-9.2f %-9.4f %-11.4f %-12.3f %-10s\n', ...
        R.omega, R.w, R.t_unstable, R.safety*100, R.kl, R.maxstep, ...
        R.t_at_max, tern(R.diverged,'YES','no'));
end
S = struct('rows', {rows}, 'omegas', omegas, 'wslack', wslack, 'dt', dt, 'n_gen', n_gen);
save(fullfile('outputs', 'SafeFlow_GainForm_Ablation.mat'), 'S', '-v7.3');
fprintf('\nsaved outputs/SafeFlow_GainForm_Ablation.mat\n');
end

function v = tern(c,a,b), if c, v=a; else, v=b; end, end
