function archive_fm_mppi_run(n_gen, opts)
%ARCHIVE_FM_MPPI_RUN  Run and reproducibly archive the FM-MPPI baseline.
%
% The UniConFlow paper supplies only the high-level baseline definition (FM
% trajectories warm-start MPPI).  fm_mppi_nn_rollout.m records the concrete
% action parameterisation and all unpublished MPPI choices in rollout.options.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_gen), n_gen = 100; end
if nargin < 2, opts = struct(); end

car = char([36187 36710]);
trk = char([36187 36947]);
src = fullfile('outputs', [car 'safeflow_tbar05_paper']);
if ~exist(src,'dir'), src = fullfile('outputs',[car 'safeflow']); end
N = load(fullfile(src,'Racing_SafeFlow_NN_Net.mat'));
A = load(fullfile(src,'Racing_SafeFlow_NN_Rollout.mat'));
af = fieldnames(A); arch = A.(af{1});
net = N.net; n_train_steps = N.n_train_steps; cfg = get_config();

if ~isfield(opts,'trajectory_seeds')
    assert(numel(arch.trajectory_seeds) >= n_gen, ...
        'Reference archive contains fewer than n_gen trajectory seeds.');
    opts.trajectory_seeds = arch.trajectory_seeds(1:n_gen);
end

fprintf('===== FM-MPPI (%d trajectories) =====\n',n_gen);
r = fm_mppi_nn_rollout(net,n_gen,opts);
r.metrics = safeflow_nn_metrics(r);
m = r.metrics;
fprintf(['  Safety %6.2f%%  KL %.4f  CS %.4f  AS %.4g  Time %.4f s/traj\n' ...
    '  MPPI K=%d iter=%d, mean cost %.4g -> %.4g, mean final ESS %.2f\n'], ...
    100*m.safety,m.kl,m.cs,m.as,m.time_seconds, ...
    r.options.n_rollouts,r.options.n_iterations, ...
    mean(r.cost_initial),mean(r.cost_final), ...
    mean(r.effective_sample_size_final));

R = struct('fm_mppi',r);
specs = {'fm_mppi',[trk 'FM-MPPI'],'FM_MPPI_Mech','FM-MPPI', ...
    'r = fm_mppi_nn_rollout(S.net, %d, opts);'};
archive_safeflow_nn_run(R,net,cfg,n_train_steps,n_gen,{'fm_mppi'},specs);

d = fullfile('outputs',[trk 'FM-MPPI']);
o2 = r.options;
r2 = fm_mppi_nn_rollout(net,n_gen,o2);
B = load(fullfile(d,'FM_MPPI_Mech_Rollout.mat'));
dmax = max(abs(r2.points(:)-B.rollout.points(:)));
fprintf('archive replay: max|diff| = %.3e\n',dmax);
assert(dmax < 1e-12,'Archived FM-MPPI rollout is not exactly reproducible.');
write_archive_sha256(d);
fprintf('archived to %s\n',d);
end
