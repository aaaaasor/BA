function summary = archive_uniconflow_paper_100(n_gen, opts)
%ARCHIVE_UNICONFLOW_PAPER_100 Reproducible, resumable paper reconstruction.
%
% Each trajectory is generated from its own recorded MT19937 seed and is
% committed to cache immediately.  Stage-1 stores the complete PTZF control
% u(t), state path, constraint traces and QP diagnostics.  The final archive
% contains native 101-state trajectories plus a clearly labelled 65-point
% comparison representation used for KL/CS/AS.
if nargin<1||isempty(n_gen),n_gen=100;end
if nargin<2,opts=struct();end
root=fileparts(mfilename('fullpath'));
tag='uniconflow-paper-strict-cem-cache-v6';
out_dir=local_default(opts,'output_dir',fullfile(root,'outputs', ...
    '赛道UniConFlow_Paper_StrictCEM_100'));
cache_dir=fullfile(out_dir,'cache');diag_dir=fullfile(out_dir,'diagnostics');
src_dir=fullfile(out_dir,'source_snapshot');
input_dir=fullfile(out_dir,'inputs');
if ~exist(out_dir,'dir'),mkdir(out_dir);end
if ~exist(cache_dir,'dir'),mkdir(cache_dir);end
if ~exist(diag_dir,'dir'),mkdir(diag_dir);end
if ~exist(src_dir,'dir'),mkdir(src_dir);end
if ~exist(input_dir,'dir'),mkdir(input_dir);end

% A resume/aggregation-only invocation must not replace the wall time of the
% original generation batch with the much shorter cache-loading time.
prior_summary_file=fullfile(out_dir,'UniConFlow_Paper_Reconstructed_100_Summary.mat');
prior_batch_wall=nan;
if exist(prior_summary_file,'file')
    P=load(prior_summary_file,'summary');
    if isfield(P,'summary')&&isfield(P.summary,'metrics')&& ...
            isfield(P.summary.metrics,'batch_wall_seconds')
        prior_batch_wall=P.summary.metrics.batch_wall_seconds;
    end
end

source_input_dir=local_default(opts,'source_input_dir', ...
    fullfile(root,'outputs','赛道UniConFlow_Paper_Reconstructed'));
archive_inputs(root,source_input_dir,input_dir);
safety_tol=local_default(opts,'violation_tol',5e-4);

seeds=local_seeds(root,n_gen,opts);
run_config=struct('method','UniConFlow paper-strict CEM reconstruction', ...
    'cache_version',tag,'n_gen',n_gen,'trajectory_seeds',seeds, ...
    'safety_tolerance',safety_tol, ...
    'created_at',char(datetime('now','TimeZone','local')), ...
    'matlab_version',version,'computer',computer,'options',opts, ...
    'state_layout','101 x [x,y,theta,v] in metric coordinates', ...
    'action_layout','100 x [delta,tau], Section VI-C1 bounds', ...
    'comparison_layout',['65 points interpolated uniformly in normalized ' ...
        'trajectory phase solely for comparison with the project table'], ...
    'parameter_exact',false);
save(fullfile(out_dir,'Run_Config.mat'),'run_config','-v7.3');
writetable(table((1:n_gen).',seeds,repmat("pending",n_gen,1),nan(n_gen,1), ...
    false(n_gen,1),'VariableNames',{'index','seed','status','seconds','certified'}), ...
    fullfile(out_dir,'Trajectory_Status.csv'));
snapshot_sources(root,src_dir);

log_path=fullfile(out_dir,'run.log');diary(log_path);cleanup=onCleanup(@()diary('off'));
fprintf('\n===== UniConFlow reconstructed batch: %d trajectories =====\n',n_gen);
fprintf('Started %s\n',char(datetime('now','TimeZone','local')));
status=readtable(fullfile(out_dir,'Trajectory_Status.csv'),'TextType','string');
force=local_default(opts,'force_rerun',false);
batch_clock=tic;
if local_default(opts,'two_stage_parallel',false)
    run_two_stage_parallel(root,out_dir,cache_dir,input_dir,seeds,tag,opts,status);
else
for q=1:n_gen
    cache_file=fullfile(cache_dir,sprintf('trajectory_%03d_seed_%010u.mat',q,uint32(seeds(q))));
    ropts=struct('seed',double(seeds(q)),'trajectory_seeds',double(seeds(q)), ...
        'record_path',true,'record_control_trace',true,'save_result',false, ...
        'input_dir',input_dir);
    if isfield(opts,'initial_trajectory_index'),ropts.initial_trajectory_index=opts.initial_trajectory_index;end
    if isfield(opts,'stage1'),ropts.stage1=opts.stage1;end
    if isfield(opts,'cem_stages'),ropts.cem_stages=opts.cem_stages;
    else,ropts.cem_stages=default_cem_stages(seeds(q),safety_tol, ...
        local_default(opts,'stage2_max_seconds',60), ...
        local_default(opts,'stage2_max_passes',[]));end
    if exist(cache_file,'file')&&~force
        C=load(cache_file,'cache');
        assert(strcmp(C.cache.cache_version,tag)&&C.cache.seed==seeds(q)&& ...
            isequaln(C.cache.run_options,ropts), ...
            'Incompatible cache at trajectory %d. Use a new output_dir.',q);
        status.status(q)="cached";status.seconds(q)=C.cache.elapsed_seconds;
        status.certified(q)=C.cache.out.metrics.tsr==1;
        writetable(status,fullfile(out_dir,'Trajectory_Status.csv'));
        fprintf('[%3d/%3d] cached seed=%u certified=%d\n',q,n_gen,uint32(seeds(q)),status.certified(q));
        continue
    end
    fprintf('[%3d/%3d] running seed=%u ...\n',q,n_gen,uint32(seeds(q)));
    tq=tic;out=uniconflow_paper_project_run(1,ropts);elapsed=toc(tq);
    cache=struct('cache_version',tag,'index',q,'seed',seeds(q), ...
        'elapsed_seconds',elapsed,'run_options',ropts,'out',out);
    tmp=[cache_file '.tmp'];save(tmp,'cache','-v7.3');movefile(tmp,cache_file,'f');
    status.status(q)="complete";status.seconds(q)=elapsed;
    status.certified(q)=out.metrics.tsr==1;
    writetable(status,fullfile(out_dir,'Trajectory_Status.csv'));
    fprintf('          %.3f s, certified=%d, stages=%d\n',elapsed,status.certified(q),numel(out.stage_history));
end
end

% The serial (rollout batch 1) branch never writes Stage1_All_Trajectories.mat,
% but make_three_panel and the Stage-4 homotopy both read it.  Rebuild it from
% the per-trajectory caches so both paths produce the same archive.
ensure_stage1_file(out_dir,cache_dir,seeds,tag);

% ---- Stage 4: feasible-donor homotopy for whatever is still uncertified ----
if local_default(opts,'feasible_homotopy',true)
    hom=apply_uniconflow_paper_homotopy_cache(out_dir);
    if ~isempty(hom)
        fprintf('Stage 4: homotopy certified %d residual trajectories.\n',numel(hom));
    end
end

summary=aggregate_cache(root,out_dir,cache_dir,seeds,tag);
resume_wall=toc(batch_clock);
if isfinite(prior_batch_wall)
    summary.metrics.batch_wall_seconds=prior_batch_wall;
    summary.metrics.resume_aggregation_wall_seconds=resume_wall;
else
    summary.metrics.batch_wall_seconds=resume_wall;
end
summary.metrics.throughput_seconds_per_trajectory=summary.metrics.batch_wall_seconds/n_gen;
summary.metrics.mean_compute_seconds_per_trajectory=summary.metrics.time_seconds;
if local_default(opts,'two_stage_parallel',false)
    summary.metrics.time_seconds=summary.metrics.throughput_seconds_per_trajectory;
end
save(fullfile(out_dir,'UniConFlow_Paper_Reconstructed_100_Summary.mat'), ...
    'summary','run_config','-v7.3');
writetable(summary.per_trajectory,fullfile(out_dir,'Per_Trajectory_Metrics.csv'));
write_summary_text(out_dir,summary);
make_diagnostics(out_dir,diag_dir,summary);
write_reproduce_script(out_dir,n_gen,seeds);
write_archive_sha256(out_dir);
fprintf('Finished %s\n',char(datetime('now','TimeZone','local')));
fprintf('Safety %.2f%% KL %.4f CS %.4f AS %.4f Time %.4f s/traj\n', ...
    100*summary.metrics.safety,summary.metrics.kl,summary.metrics.cs, ...
    summary.metrics.as,summary.metrics.time_seconds);
end

function stages=default_cem_stages(seed,tol,max_s,max_p)
% max_s caps Stage 2.  paper_strict leaves max_passes at inf, so without a
% finite cap both CEM loops are unbounded and a trajectory the CEM cannot
% certify never returns -- which also means Stages 3 and 4, which run only
% after CEM exits, never get their turn.  The default is therefore finite:
% a hard trajectory is handed to the bounded stages instead of stalling the
% batch.  Pass opts.stage2_max_seconds=inf for the old unbounded behaviour.
if nargin<3||isempty(max_s),max_s=60;end
% max_p caps the number of CEM passes.  Leaving it empty keeps paper_strict's
% repeat-until-certified rule; max_p=1 makes Stage 2 a single repair attempt
% and hands anything it cannot fix to the bounded Stage 3 and Stage 4.
if nargin<4,max_p=[];end
base=double(mod(uint64(seed),uint64(2^31-10000)));stages=cell(1,1);
for j=1:numel(stages)
    stages{j}=struct('cem_population',512,'cem_elite',32, ...
        'cem_iterations',20,'n_pad',10,'n_rec',20, ...
        'cem_sigma',[0.15;2],'lambda_state',1e5,'seed',base+6200+j, ...
        'paper_strict',true,'max_seconds_per_trajectory',max_s, ...
        'violation_tol',tol);
    if ~isempty(max_p), stages{j}.max_passes=max_p; end
end
end

function run_two_stage_parallel(~,out_dir,cache_dir,input_dir,seeds,tag,opts,status)
n=numel(seeds);stage_file=fullfile(out_dir,'Stage1_All_Trajectories.mat');
scene=uniconflow_paper_project_scene();
force=local_default(opts,'force_rerun',false);
if exist(stage_file,'file')&&~force
    L=load(stage_file,'stage_cache');stage_cache=L.stage_cache;
    assert(strcmp(stage_cache.cache_version,tag)&& ...
        isequal(stage_cache.trajectory_seeds,seeds)&& ...
        isequaln(stage_cache.user_options,opts), ...
        'Stage-1 cache does not match this run. Use a new output_dir.');
    fprintf('Stage 1: loaded batch cache (%.3f s).\n',stage_cache.elapsed_seconds);
else
    N=load(fullfile(input_dir,'UniConFlow_Paper_Reconstructed_Net.mat'),'net');
    D=load(fullfile(input_dir,'UniConFlow_Paper_Reconstructed_Dataset.mat'),'data');
    idx=local_default(opts,'initial_trajectory_index',1);scur=D.data.states(:,1,idx);
    % This branch builds its own Stage-1 options instead of going through
    % uniconflow_paper_project_run, so the layout and sub-step count have to be
    % repeated here.  Both are read off the archived network and dataset so the
    % parallel and serial paths cannot drift apart.
    n_states_net=uniconflow_paper_layout(numel(N.net.mu_d));
    n_sub_net=local_default(opts,'n_sub',1);
    if isfield(D.data,'spec')&&isfield(D.data.spec,'n_sub')
        n_sub_net=local_default(opts,'n_sub',D.data.spec.n_sub);
    end
    s1=struct('n_states',n_states_net,'n_sub',n_sub_net, ...
        'dt',D.data.dt,'n_gen',n,'n_steps',100,'t_end',0.995, ...
        'transform_switch',0.9,'transform_step',0.5,'seed',double(seeds(1)), ...
        'trajectory_seeds',seeds,'state_constraint',scene.state_constraint, ...
        'P_delta',local_default(opts,'P_delta',1e4),'P_u',1,'c_PT',3,'c_g',1, ...
        'record_path',true,'record_control_trace',true);
    if isfield(opts,'stage1'),s1=merge_local(s1,opts.stage1);end
    fprintf('Stage 1: batching %d independent seeds ...\n',n);t1=tic;
    stage1=uniconflow_paper_guided_sample(N.net,scur,s1);
    stage1_elapsed=toc(t1);
    stage1_violations=count_violations(stage1.states,scene, ...
        local_default(opts,'violation_tol',5e-4));
    stage_cache=struct('cache_version',tag,'trajectory_seeds',seeds, ...
        'elapsed_seconds',stage1_elapsed,'violations',stage1_violations, ...
        'user_options',opts,'out',stage1);
    tmp=[stage_file '.tmp'];save(tmp,'stage_cache','-v7.3');movefile(tmp,stage_file,'f');
    fprintf('Stage 1 complete: %.3f s total (%.3f s/trajectory).\n', ...
        stage1_elapsed,stage1_elapsed/n);
end
stage1=stage_cache.out;stage1_share=stage_cache.elapsed_seconds/n;

run_mask=true(n,1);run_options=cell(n,1);
for q=1:n
    run_options{q}=one_run_options(input_dir,seeds(q),opts);
    file=cache_name(cache_dir,q,seeds(q));
    if exist(file,'file')&&~force
        C=load(file,'cache');
        assert(strcmp(C.cache.cache_version,tag)&&C.cache.seed==seeds(q)&& ...
            isequaln(C.cache.run_options,run_options{q}), ...
            'Incompatible Stage-2 cache at trajectory %d.',q);
        run_mask(q)=false;
    end
end
todo=find(run_mask);workers=min(local_default(opts,'parallel_workers',6),6);
fprintf('Stage 2: %d cached, %d to run on at most %d process workers.\n', ...
    n-numel(todo),numel(todo),workers);
if ~isempty(todo)
    pool=gcp('nocreate');
    if isempty(pool)||pool.NumWorkers~=workers
        if ~isempty(pool),delete(pool);end
        parpool('Processes',workers);
    end
    dq=parallel.pool.DataQueue;afterEach(dq,@progress_message);
    parfor ii=1:numel(todo)
    q=todo(ii);seedq=seeds(q); %#ok<PFBNS>
    one=slice_stage1(stage1,q,stage_cache.violations(q)); %#ok<PFBNS>
    ropts=run_options{q};t2=tic; %#ok<PFBNS>
    stages=ropts.cem_stages;
    for j=1:numel(stages)
        if isfield(one,'certified')&&all(one.certified),break;end
        stages{j}.state_constraint=scene.state_constraint; %#ok<PFBNS>
        stages{j}.state_constraint_batch=scene.state_constraint_batch;
        stages{j}.state_constraint_rows_batch=scene.state_constraint_rows_batch;
        tj=tic;one=uniconflow_paper_cem_refine(one,stages{j});
        one.stage_history(end+1)=struct('name',sprintf('cem_stage_%d',j), ...
            'elapsed_seconds',toc(tj),'state_violations', ...
            one.state_violations_after(:).','certified',all(one.certified));
    end
    % Stage 3.  This branch runs Stages 1 and 2 itself instead of going
    % through uniconflow_paper_project_run, so the certifier has to be invoked
    % here too -- otherwise the parallel path silently stops after CEM while
    % the serial path certifies, and the two produce different results from
    % the same options.
    if local_default(opts,'deterministic_certifier',true) && ...
            (~isfield(one,'certified')||~all(one.certified))
        t3=tic;one=uniconflow_paper_certify(one,scene);
        one.stage_history(end+1)=struct('name','deterministic_certifier', ...
            'elapsed_seconds',toc(t3),'state_violations', ...
            one.state_violations_after(:).','certified',all(one.certified));
    end
    stage2_elapsed=toc(t2);tol=local_default(opts,'violation_tol',5e-4);
    one.metrics=uniconflow_paper_metrics(one,scene,tol);
    one.total_seconds=stage1_share+stage2_elapsed;
    one.total_seconds_per_trajectory=one.total_seconds;
    one.options.project_run=ropts;
    cache=struct('cache_version',tag,'index',q,'seed',seedq, ...
        'stage1_seconds_share',stage1_share,'stage2_elapsed_seconds',stage2_elapsed, ...
        'elapsed_seconds',one.total_seconds,'run_options',ropts,'out',one);
    save_cache(cache_name(cache_dir,q,seedq),cache);
        send(dq,[q,one.metrics.tsr,stage2_elapsed]);
    end
end
for q=1:n
    file=cache_name(cache_dir,q,seeds(q));C=load(file,'cache');
    status.status(q)="complete";status.seconds(q)=C.cache.elapsed_seconds;
    status.certified(q)=C.cache.out.metrics.tsr==1;
end
writetable(status,fullfile(out_dir,'Trajectory_Status.csv'));
end

function ropts=one_run_options(input_dir,seed,opts)
ropts=struct('seed',double(seed),'trajectory_seeds',double(seed), ...
    'record_path',true,'record_control_trace',true,'save_result',false, ...
    'input_dir',input_dir);
if isfield(opts,'initial_trajectory_index'),ropts.initial_trajectory_index=opts.initial_trajectory_index;end
if isfield(opts,'stage1'),ropts.stage1=opts.stage1;end
if isfield(opts,'cem_stages')
    ropts.cem_stages=opts.cem_stages;
else
    ropts.cem_stages=default_cem_stages(seed, ...
        local_default(opts,'violation_tol',5e-4), ...
        local_default(opts,'stage2_max_seconds',60), ...
        local_default(opts,'stage2_max_passes',[]));
end
end

function one=slice_stage1(all,q,n_bad)
one=all;
one.trajectory=all.trajectory(:,q);one.states=all.states(:,:,q);one.actions=all.actions(:,:,q);
one.normalized_state=all.normalized_state(:,q);one.z0=all.z0(:,q);
one.z_path=all.z_path(:,q,:);one.u_trace=all.u_trace(:,q,:);
one.g_final=all.g_final(q);one.h_max_final=all.h_max_final(q);
one.g_trace=all.g_trace(:,q);one.h_trace=all.h_trace(:,q);
one.active_trace=all.active_trace(:,q);one.rho_max_trace=all.rho_max_trace(:,q);
one.qp_residual_trace=all.qp_residual_trace(:,q);one.trajectory_seeds=all.trajectory_seeds(q);
one.options.n_gen=1;
one.stage_history=struct('name','stage1','elapsed_seconds',nan, ...
    'state_violations',n_bad,'certified',false);
end

function n=count_violations(S,scene,tol)
ng=size(S,3);n=zeros(1,ng);
for q=1:ng
    for k=1:size(S,2),n(q)=n(q)+any(scene.state_constraint(S(:,k,q),k-1)>tol);end
end
end

function p=cache_name(d,q,seed)
p=fullfile(d,sprintf('trajectory_%03d_seed_%010u.mat',q,uint32(seed)));
end
function save_cache(file,cache)
tmp=[tempname(fileparts(file)) '.mat'];save(tmp,'cache','-v7.3');movefile(tmp,file,'f');
end
function progress_message(x)
fprintf('  Stage 2 [%3d] certified=%d, worker time %.3f s\n',x(1),logical(x(2)),x(3));
end
function out=merge_local(a,b)
out=a;f=fieldnames(b);for k=1:numel(f),out.(f{k})=b.(f{k});end
end

function seeds=local_seeds(root,n,opts)
if isfield(opts,'trajectory_seeds')&&~isempty(opts.trajectory_seeds)
    seeds=double(opts.trajectory_seeds(:));
else
    car=char([36187 36710]);src=fullfile(root,'outputs',[car 'safeflow_tbar05_paper']);
    if ~exist(src,'dir'),src=fullfile(root,'outputs',[car 'safeflow']);end
    A=load(fullfile(src,'Racing_SafeFlow_NN_Rollout.mat'));
    f=fieldnames(A);r=A.(f{1});seeds=double(r.trajectory_seeds(:));
end
assert(numel(seeds)>=n,'Need at least n_gen trajectory seeds.');seeds=seeds(1:n);
assert(all(seeds>=1&seeds<=2^31-2&seeds==floor(seeds)),'Invalid trajectory seed.');
end

function summary=aggregate_cache(root,out_dir,cache_dir,seeds,tag)
n=numel(seeds);scene=uniconflow_paper_project_scene(struct('aggregate',false));
% Size the aggregation from the first cache rather than the paper's 101, so a
% 65-state run aggregates without silently truncating.
C0=load(cache_name(cache_dir,1,seeds(1)),'cache');
ns0=size(C0.cache.out.states,2); H0=size(C0.cache.out.actions,2);
states=zeros(4,ns0,n);actions=zeros(2,H0,n);points101=zeros(ns0,n,2);
points65=zeros(65,n,2);seconds=zeros(n,1);cert=false(n,1);stages=zeros(n,1);
srs=false(n,1);ars=false(n,1);kcf=zeros(n,1);
stage1_bad=zeros(n,1);final_bad=zeros(n,1);cs=zeros(n,1);as=zeros(n,1);
first_file=fullfile(cache_dir,sprintf('trajectory_%03d_seed_%010u.mat',1,uint32(seeds(1))));
C0=load(first_file,'cache');t=C0.cache.out.u_trace_t(:);nt=numel(t);
unorm=zeros(nt,n);umax=unorm;active=unorm;gtrace=unorm;htrace=unorm;
rhomax=unorm;qpres=unorm;
for q=1:n
    file=fullfile(cache_dir,sprintf('trajectory_%03d_seed_%010u.mat',q,uint32(seeds(q))));
    C=load(file,'cache');assert(strcmp(C.cache.cache_version,tag));o=C.cache.out;
    states(:,:,q)=o.states(:,:,1);actions(:,:,q)=o.actions(:,:,1);
    p=(o.states(1:2,:,1).*scene.segment.transform.scale+scene.segment.transform.offset(:)).';
    ns_q=size(p,1);
    points101(:,q,:)=reshape(p,ns_q,1,2);
    % The 65-point comparison representation is what the racing table uses.
    % When the run already produces 65 states this is the identity, so it must
    % not resample -- interpolating a curve onto its own grid would otherwise
    % be a silent no-op only by luck of the phase grid matching.
    if ns_q==65
        p65=p;
    else
        p65=interp1(linspace(0,1,ns_q),p,linspace(0,1,65),'linear');
    end
    points65(:,q,:)=reshape(p65,65,1,2);
    [cs(q),as(q)]=trajectory_smoothness(p65);
    seconds(q)=C.cache.elapsed_seconds;stages(q)=numel(o.stage_history);
    % Safety is state safety only, matching how every other method in the
    % racing table is scored -- they generate plain 2-D curves and have no
    % dynamics to be consistent with.  Action feasibility and the kinodynamic
    % residual stay as their own columns instead of being folded in.
    srs(q)=o.metrics.sr_s==1; ars(q)=o.metrics.ar==1;
    kcf(q)=o.metrics.kc_f; cert(q)=o.metrics.tsr==1;
    stage1_bad(q)=o.stage_history(1).state_violations;
    if isfield(o,'state_violations_after'),final_bad(q)=o.state_violations_after;end
    U=squeeze(o.u_trace(:,1,:));
    unorm(:,q)=sqrt(sum(double(U).^2,1)).';umax(:,q)=max(abs(double(U)),[],1).';
    active(:,q)=o.active_trace(:,1);gtrace(:,q)=o.g_trace(:,1);htrace(:,q)=o.h_trace(:,1);
    rhomax(:,q)=o.rho_max_trace(:,1);qpres(:,q)=o.qp_residual_trace(:,1);
end
final_xy=reshape(points65(end,:,:),n,2);
[kl,kld]=kl_from_archive(final_xy,fullfile(out_dir,'inputs','Racing_KL_Reference.mat'));
metrics=struct('safety',mean(srs),'action_feasible',mean(ars), ...
    'kc_f',mean(kcf),'tsr',mean(cert), ...
    'kl',kl,'kl_details',kld,'cs',mean(cs), ...
    'as',mean(as),'time_seconds',mean(seconds),'time_total_seconds',sum(seconds), ...
    'n_gen',n,'native_state_count',size(states,2),'comparison_point_count',65);
per_trajectory=table((1:n).',seeds,seconds,srs,ars,kcf,cert,stage1_bad,final_bad,stages,cs,as, ...
    'VariableNames',{'index','seed','seconds','state_safe','action_feasible', ...
    'kc_f','certified','stage1_violations','final_violations','stages_run','cs_65','as_65'});
summary=struct('metrics',metrics,'per_trajectory',per_trajectory,'states',states, ...
    'actions',actions,'points_101',points101,'comparison_points_65',points65, ...
    'comparison_note','65 points are uniform trajectory-phase interpolation from the native 101-state car rollout.', ...
    'diagnostics',struct('t',t,'u_l2',unorm,'u_max_abs',umax, ...
    'active_rows',active,'g',gtrace,'h_max',htrace,'rho_max',rhomax, ...
    'qp_residual',qpres),'source_root',root,'archive_dir',out_dir);
end

function make_diagnostics(out_dir,diag_dir,s)
d=s.diagnostics;t=d.t;
f=figure('Visible','off','Color','w','Position',[40 40 1250 820]);
tl=tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
bandplot(nexttile,t,d.u_l2,'||u(t)||_2');bandplot(nexttile,t,d.u_max_abs,'max |u_i(t)|');
bandplot(nexttile,t,d.active_rows,'active constraint rows');
bandplot(nexttile,t,d.h_max,'max h(t), safe when h <= 0');yline(gca,0,'--r');
title(tl,'UniConFlow Stage-1 PTZF/QP diagnostics, 100 recorded seeds');
export_pair(f,diag_dir,'Stage1_u_and_constraints');close(f);

scene=uniconflow_paper_project_scene();f=figure('Visible','off','Color','w','Position',[40 40 1050 780]);
hold on;draw_track_segment(scene.segment,'HandleVisibility','off');
draw_obstacles(scene.obstacle,'HandleVisibility','off');P=s.comparison_points_65;
for q=1:size(P,2)
    if s.per_trajectory.certified(q),c=[0 0.45 0.74 0.35];else,c=[0.85 0.1 0.1 0.8];end
    plot(P(:,q,1),P(:,q,2),'-','Color',c,'LineWidth',0.7);
end
axis equal;grid on;box on;title(sprintf('Final trajectories: %.2f%% certified',100*s.metrics.safety));
export_pair(f,diag_dir,'Final_trajectories');close(f);

f=figure('Visible','off','Color','w','Position',[40 40 1200 420]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
nexttile;histogram(s.per_trajectory.seconds);xlabel('seconds');title('Runtime per trajectory');grid on;
nexttile;scatter(s.per_trajectory.stage1_violations,s.per_trajectory.final_violations,25,'filled');
xlabel('Stage-1 violations');ylabel('final violations');grid on;title('CEM repair');
nexttile;bar([mean(s.per_trajectory.stage1_violations),mean(s.per_trajectory.final_violations)]);
set(gca,'XTickLabel',{'Stage 1','Final'});ylabel('mean violating states');grid on;
export_pair(f,diag_dir,'Runtime_and_repair');close(f);

make_three_panel(out_dir,diag_dir,s,scene);
end

function make_three_panel(out_dir,diag_dir,s,scene)
% Match the project's canonical target/source/rollout ThreePanel figure.
D=load(fullfile(out_dir,'inputs', ...
    'UniConFlow_Paper_Reconstructed_Dataset.mat'),'data');
target_states=D.data.states;
target_xy=permute(target_states(1:2,:,:),[2 3 1]);
target_xy=target_xy.*scene.segment.transform.scale+ ...
    reshape(scene.segment.transform.offset(:),1,1,2);

L=load(fullfile(out_dir,'Stage1_All_Trajectories.mat'),'stage_cache');
[source_states,~]=uniconflow_paper_pack('unpack',L.stage_cache.out.z0);
% As in plot_results.m, the source panel displays raw standardized noise.
source_xy=permute(source_states(1:2,:,:),[2 3 1]);
rollout_xy=s.points_101;

cmp=[reshape(target_xy,[],2);reshape(rollout_xy,[],2); ...
    scene.segment.left;scene.segment.right];
cmp=cmp(all(isfinite(cmp),2),:);
xl=[min(cmp(:,1)),max(cmp(:,1))];yl=[min(cmp(:,2)),max(cmp(:,2))];
xl=xl+0.05*max(diff(xl),eps)*[-1 1];
yl=yl+0.05*max(diff(yl),eps)*[-1 1];

f=figure('Color','w','Position',[60 60 1500 520],'Visible','off');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile;hold on;
draw_track_segment(scene.segment,'HandleVisibility','off');
draw_obstacles(scene.obstacle,'HandleVisibility','off');
for q=1:size(target_xy,2)
    plot(target_xy(:,q,1),target_xy(:,q,2),'.-','LineWidth',0.8, ...
        'HandleVisibility','off');
end
grid on;axis equal;xlim(xl);ylim(yl);xlabel('x');ylabel('y');
title(sprintf('Target Trajectory Data (2D, %d Points)',size(target_xy,1)));

nexttile;hold on;
for q=1:size(source_xy,2)
    plot(source_xy(:,q,1),source_xy(:,q,2),'--','LineWidth',0.9, ...
        'HandleVisibility','off');
end
grid on;axis equal;xlabel('x');ylabel('y');title('ODE Source Trajectories');

nexttile;hold on;rollout_track=scene.segment;
if isfield(rollout_track,'raceline'),rollout_track.raceline(:)=nan;end
draw_track_segment(rollout_track,'HandleVisibility','off');
draw_obstacles(scene.obstacle,'HandleVisibility','off');
colors=lines(max(size(rollout_xy,2),1));
for q=1:size(rollout_xy,2)
    if q==1
        plot(rollout_xy(:,q,1),rollout_xy(:,q,2),'-','Color',colors(q,:), ...
            'LineWidth',1.1,'DisplayName','Generated curves');
    else
        plot(rollout_xy(:,q,1),rollout_xy(:,q,2),'-','Color',colors(q,:), ...
            'LineWidth',1.1,'HandleVisibility','off');
    end
end
grid on;axis equal;xlim(xl);ylim(yl);xlabel('x');ylabel('y');
title(sprintf('UniConFlow Rollout (%d curves, Safety %.2f%%)', ...
    size(rollout_xy,2),100*s.metrics.safety));
lgd=legend('Location','southoutside');lgd.FontSize=8;
lgd.ItemTokenSize=[14 8];
export_pair(f,diag_dir,'UniConFlow_ThreePanel');close(f);
end

function bandplot(ax,x,Y,label)
hold(ax,'on');lo=pct(Y,5);md=pct(Y,50);hi=pct(Y,95);
fill(ax,[x;flipud(x)],[lo;flipud(hi)],[0.75 0.86 0.95], ...
    'EdgeColor','none','FaceAlpha',0.7);plot(ax,x,md,'Color',[0 0.35 0.7],'LineWidth',1.2);
grid(ax,'on');box(ax,'on');xlabel(ax,'generation time t');ylabel(ax,label);legend(ax,{'5--95%','median'});
end
function y=pct(X,p)
X=sort(X,2);k=1+(size(X,2)-1)*p/100;a=floor(k);b=ceil(k);
if a==b,y=X(:,a);else,y=X(:,a)+(k-a)*(X(:,b)-X(:,a));end
end
function export_pair(f,d,name)
exportgraphics(f,fullfile(d,[name '.png']),'Resolution',220);
print(f,fullfile(d,[name '.emf']),'-dmeta','-vector');
end

function write_summary_text(out_dir,s)
fid=fopen(fullfile(out_dir,'RESULTS.txt'),'w');c=onCleanup(@()fclose(fid));
fprintf(fid,'UniConFlow paper-reconstructed on project racing data\n');
fprintf(fid,'n = %d\nSafety = %.8f\nKL = %.8f\nCS_65 = %.8f\nAS_65 = %.8f\n', ...
    s.metrics.n_gen,s.metrics.safety,s.metrics.kl,s.metrics.cs,s.metrics.as);
fprintf(fid,'Reported parallel throughput seconds/trajectory = %.8f\n', ...
    s.metrics.time_seconds);
fprintf(fid,'Primary batch wall seconds = %.8f\n',s.metrics.batch_wall_seconds);
fprintf(fid,'Mean compute seconds/trajectory = %.8f\n', ...
    s.metrics.mean_compute_seconds_per_trajectory);
fprintf(fid,'Total compute seconds = %.8f\n',s.metrics.time_total_seconds);
fprintf(fid,'NOTE: This is a reconstruction from public equations, not a parameter-exact official reproduction.\n');
end

function [kl,det]=kl_from_archive(samples,ref_path)
L=load(ref_path,'kl_reference');ref=L.kl_reference;
[gx,gy]=meshgrid(ref.x_grid,ref.y_grid);grid_xy=[gx(:),gy(:)];
q=zeros(size(grid_xy,1),1);
for k=1:size(samples,1)
    dv=(grid_xy-samples(k,:))./ref.bandwidth;
    q=q+exp(-0.5*sum(dv.^2,2));
end
q=max(q/max(size(samples,1),1)/(2*pi*prod(ref.bandwidth)),realmin('double'));
p=ref.dataset_density(:);step=[ref.x_grid(2)-ref.x_grid(1),ref.y_grid(2)-ref.y_grid(1)];
ii=p>0;kl=max(0,sum(p(ii).*(log(p(ii))-log(q(ii))))*prod(step));
outside=samples(:,1)<ref.x_grid(1)|samples(:,1)>ref.x_grid(end)| ...
    samples(:,2)<ref.y_grid(1)|samples(:,2)>ref.y_grid(end);
det=struct('bandwidth',ref.bandwidth,'grid_step',step, ...
    'out_of_grid_rate',mean(outside),'out_of_grid_count',nnz(outside), ...
    'reference_path',ref_path,'reference_basis',ref.basis);
end

function write_reproduce_script(out_dir,n,seeds)
fid=fopen(fullfile(out_dir,'run_reproduce.m'),'w');c=onCleanup(@()fclose(fid));
fprintf(fid,"root = fileparts(fileparts(fileparts(mfilename('fullpath'))));\n");
fprintf(fid,"addpath(root);\nload(fullfile(fileparts(mfilename('fullpath')),'Run_Config.mat'),'run_config');\n");
fprintf(fid,"opts = run_config.options;\nopts.trajectory_seeds = run_config.trajectory_seeds;\n");
fprintf(fid,"opts.source_input_dir = fullfile(fileparts(mfilename('fullpath')),'inputs');\n");
fprintf(fid,"opts.output_dir = fullfile(fileparts(mfilename('fullpath')),'replay');\n");
fprintf(fid,"archive_uniconflow_paper_100(%d, opts);\n",n);
assert(numel(seeds)==n);
end

function archive_inputs(root,src,d)
names={'UniConFlow_Paper_Reconstructed_Net.mat', ...
    'UniConFlow_Paper_Reconstructed_Dataset.mat'};
for k=1:numel(names)
    target=fullfile(d,names{k});
    if ~exist(target,'file'),copyfile(fullfile(src,names{k}),target);end
end
ref=fullfile(src,'Racing_KL_Reference.mat');
if ~exist(ref,'file'),ref=fullfile(root,'outputs','Racing_KL_Reference.mat');end
if ~exist(fullfile(d,'Racing_KL_Reference.mat'),'file')
    copyfile(ref,fullfile(d,'Racing_KL_Reference.mat'));
end
end

function snapshot_sources(root,d)
files=dir(fullfile(root,'uniconflow_paper_*.m'));
for k=1:numel(files),copyfile(fullfile(files(k).folder,files(k).name),fullfile(d,files(k).name));end
extra={'archive_uniconflow_paper_100.m','trajectory_smoothness.m','safeflow_nn_kl.m', ...
    'get_config.m','write_archive_sha256.m'};
for k=1:numel(extra),p=fullfile(root,extra{k});if exist(p,'file'),copyfile(p,fullfile(d,extra{k}));end,end
end
function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end

% =====================================================================
function ensure_stage1_file(out_dir,cache_dir,seeds,tag)
%ENSURE_STAGE1_FILE Rebuild Stage1_All_Trajectories.mat from the caches.
% The two-stage parallel branch writes this file itself.  The serial branch
% does not, so reconstruct the two fields the downstream consumers use:
% out.z0 (604 x n, the Stage-1 source noise) and out.states (4 x 101 x n, the
% Stage-1 state reference the homotopy measures its motion against).
stage_file=fullfile(out_dir,'Stage1_All_Trajectories.mat');
if exist(stage_file,'file'), return; end
n=numel(seeds); z0=[]; S=[]; viol=zeros(1,n); el=0;
for q=1:n
    f=fullfile(cache_dir,sprintf('trajectory_%03d_seed_%010u.mat',q,uint32(seeds(q))));
    if ~exist(f,'file'), return; end          % incomplete run: leave it alone
    C=load(f,'cache'); o=C.cache.out;
    assert(isfield(o,'stage1_states'), ...
        'Cache %d has no stage1_states; cannot rebuild the Stage-1 file.',q);
    if isempty(z0)
        z0=zeros(size(o.z0,1),n); S=zeros(size(o.stage1_states,1), ...
            size(o.stage1_states,2),n);
    end
    z0(:,q)=o.z0(:,1); S(:,:,q)=o.stage1_states(:,:,1); %#ok<AGROW>
    viol(q)=C.cache.out.state_violations_before(1);
    el=el+C.cache.elapsed_seconds;
end
stage_cache=struct('cache_version',tag,'trajectory_seeds',seeds, ...
    'user_options',struct(),'elapsed_seconds',el,'violations',viol, ...
    'out',struct('z0',z0,'states',S), ...
    'rebuilt_from_caches',true);
tmp=[stage_file '.tmp']; save(tmp,'stage_cache','-v7.3'); movefile(tmp,stage_file,'f');
fprintf('Stage 1 file rebuilt from %d per-trajectory caches.\n',n);
end
