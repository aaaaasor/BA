function report = run_uniconflow_cem_until_safe(indices, opts)
%RUN_UNICONFLOW_CEM_UNTIL_SAFE Test the paper's CEM-only repeat-until-safe loop.
% No deterministic QP certifier or feasible-donor homotopy is used.
if nargin<1||isempty(indices),indices=1:100;end
if nargin<2,opts=struct();end
gf=@(f,d)local_default(opts,f,d);
root=fileparts(mfilename('fullpath'));
formal=gf('formal_dir',fullfile(root,'outputs', ...
    '赛道UniConFlow_Paper_FastCertified_100'));
out_dir=gf('output_dir',fullfile(root,'outputs', ...
    '赛道UniConFlow_Paper_CEMUntilSafe_Test'));
if ~exist(out_dir,'dir'),mkdir(out_dir);end
cache_dir=fullfile(out_dir,'cache');if ~exist(cache_dir,'dir'),mkdir(cache_dir);end

L=load(fullfile(formal,'Stage1_All_Trajectories.mat'),'stage_cache');
all_stage1=L.stage_cache.out;seeds=L.stage_cache.trajectory_seeds(:);
scene=uniconflow_paper_project_scene();indices=indices(:);
assert(all(indices>=1&indices<=numel(seeds)),'Invalid trajectory index.');
selected_seeds=seeds(indices);
stage1_spec=all_stage1.spec;
state_constraint=scene.state_constraint;
state_constraint_batch=scene.state_constraint_batch;
workers=min(gf('parallel_workers',6),6);
max_restarts=gf('max_restarts',50);
max_seconds=gf('max_seconds_per_trajectory',600);

rows=cell(numel(indices),1);dq=parallel.pool.DataQueue;
afterEach(dq,@show_progress);pool=gcp('nocreate');
if isempty(pool)||pool.NumWorkers~=workers
    if ~isempty(pool),delete(pool);end
    parpool('Processes',workers);
end
parfor ii=1:numel(indices)
    result=struct(); %#ok<NASGU>
    one=struct();run_needed=false;restart=0;elapsed_before=0;
    history=struct('restart',{},'seconds',{},'violations',{},'certified',{});
    q=indices(ii);seedq=selected_seeds(ii);file=fullfile(cache_dir,sprintf( ...
        'trajectory_%03d_seed_%010u.mat',q,uint32(seedq)));
    if exist(file,'file')
        C=load(file,'result');result=C.result;
        run_needed=~result.certified;
        if run_needed
            one=struct('states',result.states,'actions',result.actions, ...
                'spec',stage1_spec,'trajectory', ...
                uniconflow_paper_pack('pack',result.states,result.actions), ...
                'mode','uniconflow_paper_reconstructed_full', ...
                'state_violations_after',result.violations);
            history=result.history;restart=result.restarts;
            elapsed_before=result.seconds;
        end
    else
        run_needed=true;one=minimal_stage1(all_stage1,q);restart=0;
        elapsed_before=0;
    end
    if run_needed
        clock=tic;restart_limit=restart+max_restarts;
        while restart<restart_limit&&toc(clock)<max_seconds
            restart=restart+1;base=double(mod(uint64(seedq),uint64(2^31-10000)));
            copt=struct('cem_population',512,'cem_elite',32, ...
                'cem_iterations',20,'max_passes',12,'n_pad',10,'n_rec',20, ...
                'cem_sigma',[0.15;2],'lambda_state',1e5, ...
                'seed',base+6200+restart, ...
                'state_constraint',state_constraint, ...
                'state_constraint_batch',state_constraint_batch);
            tr=tic;one=uniconflow_paper_cem_refine(one,copt);dt=toc(tr);
            history(end+1)=struct('restart',restart,'seconds',dt, ...
                'violations',one.state_violations_after, ...
                'certified',all(one.certified)); %#ok<AGROW>
            if all(one.certified),break;end
        end
        result=struct('index',q,'seed',seedq,'certified',all(one.certified), ...
            'violations',one.state_violations_after,'restarts',restart, ...
            'seconds',elapsed_before+toc(clock),'history',history,'states',one.states, ...
            'actions',one.actions,'stopped_by_watchdog', ...
            ~all(one.certified)&&(restart>=restart_limit||toc(clock)>=max_seconds));
        save_atomic(file,result);
    end
    rows{ii}=rmfield(result,{'history','states','actions'});
    send(dq,[q,result.certified,result.restarts,result.seconds,result.violations]);
end
report=struct2table(vertcat(rows{:}));
writetable(report,fullfile(out_dir,'CEM_Until_Safe_Report.csv'));
save(fullfile(out_dir,'CEM_Until_Safe_Report.mat'),'report','indices','opts','-v7.3');
fprintf('CEM-only certified %d/%d = %.2f%%; wall-accounted mean %.3f s.\n', ...
    nnz(report.certified),height(report),100*mean(report.certified),mean(report.seconds));
end

function one=minimal_stage1(all,q)
one=struct('states',all.states(:,:,q),'actions',all.actions(:,:,q), ...
    'spec',all.spec,'trajectory',all.trajectory(:,q),'mode',all.mode);
end
function save_atomic(file,result)
tmp=[tempname(fileparts(file)) '.mat'];save(tmp,'result','-v7.3');movefile(tmp,file,'f');
end
function show_progress(x)
fprintf('q=%3d certified=%d restarts=%d seconds=%.2f violations=%d\n', ...
    x(1),logical(x(2)),x(3),x(4),x(5));
end
function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
