function rerun_nn_quadprog(n_gen)
%RERUN_NN_QUADPROG  Re-run every NN baseline except UniConFlow with the
% generic QP solver instead of the closed-form active-set enumeration, and
% overwrite each archive directory.
%
% Each run replays its own archived option set with qp_backend = 'quadprog'
% added, so the trajectory seeds, network weights and every method parameter
% are unchanged; only the solver differs.
%
% SafeFlow is run last because the other scripts read its archive for the
% network and the trajectory seeds.
if nargin < 1 || isempty(n_gen), n_gen = 100; end
root = 'C:\Users\JieJi\BA\matlab'; cd(root);
car = char([36187 36710]);
src = fullfile('outputs', [car 'safeflow']);

N = load(fullfile(src, 'Racing_SafeFlow_NN_Net.mat'));
net = N.net; n_train_steps = N.n_train_steps;
cfg = get_config();
SF = load(fullfile(src, 'Racing_SafeFlow_NN_Rollout.mat'));
FM = load(fullfile('outputs', [car 'fm'], 'Racing_FM_NN_Rollout.mat'));
seeds = SF.rollout.trajectory_seeds(1:n_gen);
sf_opts = SF.rollout.options;
fm_opts = FM.rollout.options;

ws1 = warning('off','MATLAB:nearlySingularMatrix');
ws2 = warning('off','MATLAB:singularMatrix');
cleanup = onCleanup(@() restore_warnings(ws1, ws2));

t_all = tic;
fprintf('\n########## PCFM ##########\n');
archive_pcfm_run(n_gen, 'quadprog');

fprintf('\n########## FM-MPPI ##########\n');
archive_fm_mppi_run(n_gen, struct('qp_backend','quadprog'));

fprintf('\n########## RoSD / ReSD / TVSD ##########\n');
archive_safediffuser_variants(n_gen, {'rosd','resd','tvsd'}, 'quadprog');

fprintf('\n########## FM ##########\n');
o = prep(fm_opts, seeds);
r_fm = safeflow_nn_rollout(net, 'fm', n_gen, o);
r_fm.metrics = safeflow_nn_metrics(r_fm);
report('FM', r_fm);

fprintf('\n########## SafeFlow ##########\n');
o = prep(sf_opts, seeds);
r_sf = safeflow_nn_rollout(net, 'safeflow', n_gen, o);
r_sf.metrics = safeflow_nn_metrics(r_sf);
report('SafeFlow', r_sf);

R = struct('fm', r_fm, 'safeflow', r_sf);
archive_safeflow_nn_run(R, net, cfg, n_train_steps, n_gen);
save(fullfile('outputs','SafeFlowNN_results.mat'), 'R', 'net', 'cfg', ...
    'n_train_steps', 'n_gen', '-v7.3');
fprintf('\nALL DONE in %.1f s\n', toc(t_all));
end

function o = prep(o, seeds)
o.qp_backend = 'quadprog';
o.trajectory_seeds = seeds;
o.record_path = true;
end

function report(name, r)
m = r.metrics;
fprintf('  %-10s Safety %6.2f%%  KL %.4f  CS %.4f  AS %.4g  Time %.4f s/traj\n', ...
    name, m.safety*100, m.kl, m.cs, m.as, m.time_seconds);
fprintf('             n_qp %d  slack %d  infeas_hard %d  terminal failed %d\n', ...
    r.n_qp, r.slack_active, r.infeasible_hard, r.terminal_failed);
end

function restore_warnings(ws1, ws2)
warning(ws1); warning(ws2);
end
