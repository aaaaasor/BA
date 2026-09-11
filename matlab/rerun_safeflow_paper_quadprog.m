function rerun_safeflow_paper_quadprog(n_gen)
%RERUN_SAFEFLOW_PAPER_QUADPROG  Re-run the published SafeFlow archive
% (outputs/<car>safeflow_tbar05_paper -- the configuration the comparison
% table reports, activation_time 0.5 / slack_weight 1) with the generic QP
% solver, and overwrite that directory.
if nargin < 1 || isempty(n_gen), n_gen = 100; end
root = 'C:\Users\JieJi\BA\matlab'; cd(root);
car = char([36187 36710]);
d = [car 'safeflow_tbar05_paper'];
N = load(fullfile('outputs', d, 'Racing_SafeFlow_NN_Net.mat'));
A = load(fullfile('outputs', d, 'Racing_SafeFlow_NN_Rollout.mat'), 'rollout');
o = A.rollout.options;
o.trajectory_seeds = A.rollout.trajectory_seeds(1:n_gen);
o.qp_backend = 'quadprog';
ws1 = warning('off','MATLAB:nearlySingularMatrix');
ws2 = warning('off','MATLAB:singularMatrix');
r = safeflow_nn_rollout(N.net, 'safeflow', n_gen, o);
warning(ws1); warning(ws2);
r.metrics = safeflow_nn_metrics(r);
m = r.metrics;
fprintf('  SafeFlow(paper) Safety %6.2f%%  KL %.4f  CS %.4f  AS %.4g  Time %.4f s/traj\n', ...
    100*m.safety, m.kl, m.cs, m.as, m.time_seconds);
fprintf('  n_qp %d  slack %d  terminal moved %d failed %d\n', ...
    r.n_qp, r.slack_active, r.terminal_corrected, r.terminal_failed);
% Compare against the closed-form archive before it is overwritten.
fprintf('  vs closed-form archive: max|dpoints| = %.3e\n', ...
    max(abs(r.points(:) - A.rollout.points(:))));
R = struct('safeflow', r);
specs = { 'safeflow', d, 'Racing_SafeFlow_NN', 'SafeFlow (NN)' };
archive_safeflow_nn_run(R, N.net, get_config(), N.n_train_steps, n_gen, ...
    {'safeflow'}, specs);
end
