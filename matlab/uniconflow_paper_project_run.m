function out = uniconflow_paper_project_run(n_gen, opts)
%UNICONFLOW_PAPER_PROJECT_RUN Run the reconstructed method on shared data.
if nargin<1||isempty(n_gen),n_gen=1;end
if nargin<2,opts=struct();end
root=fileparts(mfilename('fullpath'));
od=local_default(opts,'input_dir',fullfile(root,'outputs','赛道UniConFlow_Paper_Reconstructed'));
N=load(fullfile(od,'UniConFlow_Paper_Reconstructed_Net.mat'),'net');
D=load(fullfile(od,'UniConFlow_Paper_Reconstructed_Dataset.mat'),'data');
scene=uniconflow_paper_project_scene();
idx=local_default(opts,'initial_trajectory_index',1);
scur=D.data.states(:,1,idx);
safety_tol=local_default(opts,'violation_tol',5e-4);
% The trajectory layout is whatever the archived network was trained on, and
% the sub-step count must travel with it: every rollout downstream reads both
% from the spec, so a mismatch here would surface only as a silent dynamics
% residual rather than an error.
n_states_net=uniconflow_paper_layout(numel(N.net.mu_d));
n_sub_net=local_default(opts,'n_sub', ...
    struct_field_or(D.data,'n_sub',struct_field_or(D.data.spec,'n_sub',1)));
s1=struct('n_states',n_states_net,'n_sub',n_sub_net,'dt',D.data.dt,'n_gen',n_gen,'n_steps',100,'t_end',0.995, ...
    'transform_switch',0.9,'transform_step',0.5,'seed', ...
    local_default(opts,'seed',73),'state_constraint',scene.state_constraint, ...
    'P_delta',local_default(opts,'P_delta',1e4),'P_u',1,'c_PT',3,'c_g',1, ...
    'trajectory_seeds',local_default(opts,'trajectory_seeds',[]), ...
    'record_path',local_default(opts,'record_path',false), ...
    'record_control_trace',local_default(opts,'record_control_trace',false));
if isfield(opts,'stage1'),s1=merge_struct(s1,opts.stage1);end
t_all=tic;
t_stage=tic;out=uniconflow_paper_guided_sample(N.net,scur,s1);
history=struct('name','stage1','elapsed_seconds',toc(t_stage), ...
    'state_violations',count_state_violations(out,scene,safety_tol), ...
    'certified',false);

% The paper does not publish CEM hyperparameters.  This default keeps its
% frozen/violation/recovery/global sequence and repeat-until-certified rule.
% Stage 3 (a deterministic action-space QP certifier) runs on whatever CEM
% leaves uncertified.  It is OUR addition, not part of the published method:
% with it enabled, UniConFlow's safety rate is not attributable to the paper
% alone.  Set opts.deterministic_certifier=false for the paper-only pipeline.
base=double(mod(uint64(local_default(opts,'seed',73)),uint64(2^31-10000)));
stages=cell(1,1);
for j=1:numel(stages)
    stages{j}=struct('cem_population',512,'cem_elite',32, ...
        'cem_iterations',20,'n_pad',10,'n_rec',20, ...
        'cem_sigma',[0.15;2],'lambda_state',1e5,'seed',base+6200+j, ...
        'paper_strict',true,'max_seconds_per_trajectory', ...
        local_default(opts,'stage2_max_seconds',inf), ...
        'violation_tol',safety_tol);
end
if isfield(opts,'cem_stages'),stages=opts.cem_stages;end
for j=1:numel(stages)
    if isfield(out,'certified')&&all(out.certified),break;end
    stages{j}.state_constraint=scene.state_constraint;
    stages{j}.state_constraint_batch=scene.state_constraint_batch;
    stages{j}.state_constraint_rows_batch=scene.state_constraint_rows_batch;
    t_stage=tic;
    out=uniconflow_paper_cem_refine(out,stages{j});
    history(end+1)=struct('name',sprintf('cem_stage_%d',j), ...
        'elapsed_seconds',toc(t_stage), ...
        'state_violations',out.state_violations_after(:).', ...
        'certified',all(out.certified)); %#ok<AGROW>
end
% ---- Stage 3: deterministic action-space QP certifier ----
if local_default(opts,'deterministic_certifier',true) && ...
        (~isfield(out,'certified')||~all(out.certified))
    t_stage=tic;out=uniconflow_paper_certify(out,scene);
    history(end+1)=struct('name','deterministic_certifier', ...
        'elapsed_seconds',toc(t_stage), ...
        'state_violations',out.state_violations_after(:).', ...
        'certified',all(out.certified)); %#ok<AGROW>
end
out.metrics=uniconflow_paper_metrics(out,scene,safety_tol);
out.stage_history=history;
out.total_seconds=toc(t_all);
out.total_seconds_per_trajectory=out.total_seconds/n_gen;
out.options.project_run=opts;
if local_default(opts,'save_result',true)
    save(fullfile(od,'UniConFlow_Paper_Reconstructed_Latest.mat'),'out','-v7.3');
end
end

function n=count_state_violations(out,scene,tol)
ng=size(out.states,3);n=zeros(1,ng);
for q=1:ng
    for k=1:size(out.states,2)
        n(q)=n(q)+any(scene.state_constraint(out.states(:,k,q),k-1)>tol);
    end
end
end

function v=struct_field_or(s,f,d)
if isstruct(s)&&isfield(s,f),v=s.(f);else,v=d;end
end
function out=merge_struct(a,b)
out=a;f=fieldnames(b);for k=1:numel(f),out.(f{k})=b.(f{k});end
end
function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
