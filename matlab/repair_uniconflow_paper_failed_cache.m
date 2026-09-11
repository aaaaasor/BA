function report = repair_uniconflow_paper_failed_cache(out_dir,indices,workers)
%REPAIR_UNICONFLOW_PAPER_FAILED_CACHE Resume only uncertified trajectories.
if nargin<2||isempty(indices)
    T=readtable(fullfile(out_dir,'Trajectory_Status.csv'));
    indices=T.index(~logical(T.certified));
end
if nargin<3||isempty(workers),workers=min(6,numel(indices));end
cache_dir=fullfile(out_dir,'cache');scene=uniconflow_paper_project_scene();
files=dir(fullfile(cache_dir,'trajectory_*.mat'));assert(numel(files)==100);
pool=gcp('nocreate');
if isempty(pool)||pool.NumWorkers~=workers
    if ~isempty(pool),delete(pool);end
    parpool('Processes',workers);
end
dq=parallel.pool.DataQueue;afterEach(dq,@show_progress);
results=cell(numel(indices),1);
parfor ii=1:numel(indices)
    q=indices(ii);file=fullfile(files(q).folder,files(q).name); %#ok<PFBNS>
    C=load(file,'cache');cache=C.cache;out=cache.out;tall=tic;
    attempts=struct('name',{},'seconds',{},'violations',{},'certified',{});
    configs=repair_configs(cache.seed);
    for j=1:numel(configs)
        if out.certified,break;end
        configs{j}.state_constraint=scene.state_constraint; %#ok<PFBNS>
        configs{j}.state_constraint_batch=scene.state_constraint_batch;
        configs{j}.reconstruct_initial_actions=false;t=tic;
        out=uniconflow_paper_cem_refine(out,configs{j});
        out=uniconflow_paper_certify(out,scene,filter_config(j));
        attempts(end+1)=struct('name',sprintf('repair_%d',j), ...
            'seconds',toc(t),'violations',out.state_violations_after, ...
            'certified',all(out.certified)); %#ok<AGROW>
    end
    extra=toc(tall);out.metrics=uniconflow_paper_metrics(out,scene);
    out.repair_history=attempts;out.total_seconds=out.total_seconds+extra;
    out.total_seconds_per_trajectory=out.total_seconds;
    cache.out=out;cache.elapsed_seconds=cache.elapsed_seconds+extra;
    cache.stage2_elapsed_seconds=cache.stage2_elapsed_seconds+extra;
    cache.repaired=true;cache.repair_elapsed_seconds=extra;
    save_atomic(file,cache);results{ii}=struct('index',q,'seconds',extra, ...
        'certified',all(out.certified),'violations',out.state_violations_after);
    send(dq,[q,all(out.certified),out.state_violations_after,extra]);
end
report=vertcat(results{:});
end

function C=repair_configs(seed)
base=double(mod(uint64(seed),uint64(2^31-20000)));
sigmas={[0.15;2],[0.25;4],[0.08;1],[0.35;6],[0.04;0.5],[0.20;3]};
C=cell(size(sigmas));
for j=1:numel(C)
    C{j}=struct('cem_population',768,'cem_elite',48,'cem_iterations',25, ...
        'max_passes',15,'n_pad',min(10+2*j,25),'n_rec',min(20+3*j,40), ...
        'cem_sigma',sigmas{j},'lambda_state',1e6,'seed',base+12000+j);
end
end
function c=filter_config(j)
if mod(j,2)==1
    c=struct('trust_region',[0.15;3],'activation_margin',0.05, ...
        'max_iterations',60,'slack_penalty',1e8);
else
    c=struct('trust_region',[0.05;1],'activation_margin',0.10, ...
        'max_iterations',100,'slack_penalty',1e9);
end
end
function save_atomic(file,cache)
tmp=[tempname(fileparts(file)) '.mat'];save(tmp,'cache','-v7.3');movefile(tmp,file,'f');
end
function show_progress(x)
fprintf('repair [%3d] certified=%d violations=%d time=%.3f s\n', ...
    x(1),logical(x(2)),x(3),x(4));
end
