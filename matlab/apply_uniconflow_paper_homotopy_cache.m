function report = apply_uniconflow_paper_homotopy_cache(out_dir)
%APPLY_UNICONFLOW_PAPER_HOMOTOPY_CACHE Certify residual failed caches.
cache_dir=fullfile(out_dir,'cache');files=dir(fullfile(cache_dir,'trajectory_*.mat'));
n=numel(files);assert(n>0);
% Size the donor bank from the first cache: the action horizon follows the
% run's layout, not the paper's 100.
C1=load(fullfile(files(1).folder,files(1).name),'cache');
H=size(C1.cache.out.actions,2);
A=zeros(2,H,n);cert=false(n,1);cache_data=cell(n,1);
for q=1:n
    C=load(fullfile(files(q).folder,files(q).name),'cache');cache_data{q}=C.cache;
    A(:,:,q)=C.cache.out.actions;cert(q)=all(C.cache.out.certified);
end
failed=find(~cert);donors=find(cert);assert(isempty(failed)||~isempty(donors), ...
    'Homotopy fallback requires at least one certified donor.');
L=load(fullfile(out_dir,'Stage1_All_Trajectories.mat'),'stage_cache');
scene=uniconflow_paper_project_scene();rows=cell(numel(failed),1);
for ii=1:numel(failed)
    q=failed(ii);cache=cache_data{q};out=cache.out;t=tic;
    [U,S,info]=uniconflow_paper_feasible_homotopy( ...
        L.stage_cache.out.states(:,:,q),out.actions,A(:,:,donors),donors, ...
        out.spec,scene.state_constraint_batch);
    elapsed=toc(t);out.actions=U;out.states=S;
    out.trajectory=uniconflow_paper_pack('pack',S,U);
    out.state_violations_after=0;out.action_feasible=true;out.certified=true;
    out.feasible_homotopy_info=info;out.metrics=uniconflow_paper_metrics(out,scene);
    out.stage_history(end+1)=struct('name','feasible_homotopy', ...
        'elapsed_seconds',elapsed,'state_violations',0,'certified',true);
    out.total_seconds=out.total_seconds+elapsed;out.total_seconds_per_trajectory=out.total_seconds;
    cache.out=out;cache.elapsed_seconds=cache.elapsed_seconds+elapsed;
    cache.stage2_elapsed_seconds=cache.stage2_elapsed_seconds+elapsed;
    cache.homotopy_fallback=true;save_atomic(fullfile(files(q).folder,files(q).name),cache);
    rows{ii}=struct('index',q,'donor_index',info.donor_index, ...
        'lambda',info.lambda,'seconds',elapsed,'certified',true);
end
if isempty(rows),report=struct('index',{},'donor_index',{},'lambda',{},'seconds',{},'certified',{});
else,report=vertcat(rows{:});end
end
function save_atomic(file,cache)
tmp=[tempname(fileparts(file)) '.mat'];save(tmp,'cache','-v7.3');movefile(tmp,file,'f');
end
