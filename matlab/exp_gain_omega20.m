function exp_gain_omega20(omegas, wslack, n_gen, hard)
%EXP_GAIN_OMEGA20  Fill in the omega = 20 cells, split front half vs back half.
%
% Gain-form ablation, not SafeFlow: the second-order pole omega/(1-t)^2 is this
% project's gain substituted into SafeFlow's QP, with guidance and the pole
% both active from t = 0.
%
% The previous sweep only reported the maximum displacement over the whole
% rollout, which always lands on the last step.  To say anything about the
% FRONT half the two halves have to be reported separately, which is what this
% adds.  Divergence criterion is unchanged: a single-step move above 1 m.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(omegas), omegas = 20; end
if nargin < 2 || isempty(wslack), wslack = [1, 1e4]; end
if nargin < 3 || isempty(n_gen),  n_gen  = 100; end
% hard = true removes the slack variable from (30) entirely, so the QP is the
% hard min-norm problem.  That is the clean 'constraint fully enforced' limit:
% a large slack weight approximates it but makes I + w*B'*B ill-conditioned,
% so its divergences cannot be separated from solver breakdown.
if nargin < 4 || isempty(hard), hard = false; end
if hard, wslack = 1; end   % weight is unused when there is no slack term
car = char([36187 36710]);
d   = fullfile('outputs', [car 'safeflow_tbar05_paper']);
if ~exist(d, 'dir'), d = fullfile('outputs', [car 'safeflow']); end
N   = load(fullfile(d, 'Racing_SafeFlow_NN_Net.mat'));
A   = load(fullfile(d, 'Racing_SafeFlow_NN_Rollout.mat'));
af  = fieldnames(A); arch = A.(af{1});
dt  = 0.996/100;

fprintf('%-7s %-8s %-11s %-9s %-9s %-13s %-13s %-9s\n', 'omega', 'w', ...
    'unstable@t', 'Safety%', 'KL', 'maxStep t<0.5', 'maxStep t>=0.5', 'diverged');
out = {};
for om = omegas(:)'
    for w = wslack(:)'
        ws = warning('off', 'MATLAB:nearlySingularMatrix');
        ws2 = warning('off', 'MATLAB:singularMatrix');
        lastwarn('');
        r = safeflow_nn_rollout(N.net, 'safeflow', n_gen, struct( ...
            'trajectory_seeds', arch.trajectory_seeds, ...
            'u_per_stage', false, 'fmincon_fallback', false, ...
            'record_path', true, 'activation_time', 0, ...
            'phi1_switch_time', 0, 'phi1_form', 'second_order', ...
            'phi1_omega', om, 'slack_weight', w, 'slack_enabled', ~hard));
        [~, wid] = lastwarn;
        warning(ws); warning(ws2);
        m = safeflow_run_metrics(r, N.net);
        [~, D] = safeflow_path_stats(r, N.net);       % (step-1 x trajectory)
        ns = size(D,1);
        tt = (0:ns-1)' * r.t_max / ns;
        front = max(D(tt <  0.5, :), [], 'all');
        back  = max(D(tt >= 0.5, :), [], 'all');
        div = ~isfinite(max(D(:))) || max(D(:)) > 1;
        t_unstable = max(1 - sqrt(om*dt), 0);
        wlab = w; if hard, wlab = NaN; end
        fprintf('%-7g %-8g %-11.3f %-9.2f %-9.4f %-13.4g %-13.4g %-9s%s\n', ...
            om, wlab, t_unstable, m.safety*100, m.kl, front, back, ...
            tern(div,'YES','no'), tern(isempty(wid),'',' [ill-conditioned]'));
        out{end+1} = struct('omega',om,'w',w,'hard',hard,'safety',m.safety,'kl',m.kl, ...
            'front_max',front,'back_max',back,'diverged',div, ...
            't_unstable',t_unstable,'terminal_failed',m.terminal_failed, ...
            'umax',m.u_max,'ill_conditioned',~isempty(wid)); %#ok<AGROW>
    end
end
S = struct('rows', {out}, 'n_gen', n_gen, 'dt', dt);
save(fullfile('outputs',tern(hard,'SafeFlow_GainForm_HardQP.mat','SafeFlow_GainForm_Omega20.mat')),'S','-v7.3');
fprintf('\nsaved outputs/%s\n', tern(hard, 'SafeFlow_GainForm_HardQP.mat', 'SafeFlow_GainForm_Omega20.mat'));
end

function v = tern(c,a,b), if c, v=a; else, v=b; end, end
