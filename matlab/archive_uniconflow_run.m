function archive_uniconflow_run(n_gen, opts)
%ARCHIVE_UNICONFLOW_RUN  Run, score and archive adapted UniConFlow end to end.
root='C:\Users\JieJi\BA\matlab'; cd(root);
if nargin<1 || isempty(n_gen), n_gen=100; end
if nargin<2, opts=struct(); end
src=fullfile('outputs',[char([36187 36710]) 'safeflow_tbar05_paper']);
if ~exist(src,'dir'), src=fullfile('outputs',[char([36187 36710]) 'safeflow']); end
N=load(fullfile(src,'Racing_SafeFlow_NN_Net.mat'));
A=load(fullfile(src,'Racing_SafeFlow_NN_Rollout.mat'));
af=fieldnames(A); ref=A.(af{1});
if ~isfield(opts,'trajectory_seeds')
    opts.trajectory_seeds=ref.trajectory_seeds(1:n_gen);
end
r=uniconflow_full_rollout(N.net,n_gen,opts);
r.metrics=safeflow_nn_metrics(r);
r.uniconflow_metrics=uniconflow_metrics(r);
u=r.uniconflow_metrics;
fprintf(['UniConFlow SR-S %.2f%% SR-A %.2f%% AR %.2f%% TSR %.2f%% ' ...
    'KC-F %.3g Time %.3f s/traj\n'],100*u.sr_s,100*u.sr_a,100*u.ar, ...
    100*u.tsr,u.kc_f,u.time_seconds);
R=struct('uniconflow',r);
trk=char([36187 36947]);
specs={'uniconflow',[trk 'UniConFlow'],'UniConFlow_Mech','UniConFlow', ...
    'r = uniconflow_full_rollout(S.net, %d, opts);'};
archive_safeflow_nn_run(R,N.net,get_config(),N.n_train_steps,n_gen, ...
    {'uniconflow'},specs);
write_archive_sha256(fullfile('outputs',[trk 'UniConFlow']));
end
