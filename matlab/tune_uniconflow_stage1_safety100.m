function report = tune_uniconflow_stage1_safety100(stage1_file, workers)
%TUNE_UNICONFLOW_STAGE1_SAFETY100 Screen PTZF settings on failed Stage-1 seeds.
if nargin < 1 || isempty(stage1_file)
    stage1_file = 'C:\Users\JieJi\BA\matlab\outputs\赛道UniConFlow\Stage1_All_Trajectories.mat';
end
if nargin < 2, workers = 6; end
root = 'C:\Users\JieJi\BA\matlab';
input_dir = fullfile(root,'outputs','赛道UniConFlow','inputs');
N = load(fullfile(input_dir,'UniConFlow_Paper_Reconstructed_Net.mat'),'net');
D = load(fullfile(input_dir,'UniConFlow_Paper_Reconstructed_Dataset.mat'),'data');
L = load(stage1_file,'stage_cache');
scene = uniconflow_paper_project_scene();
tol = 5e-4;
R = scene.state_constraint_rows_batch(L.stage_cache.out.states);
old_max = squeeze(max(R,[],[1 2]));
failed = find(old_max > tol);
seeds = L.stage_cache.trajectory_seeds(failed);

cfg = { ...
    struct('name','base','P_delta',1e4,'c_PT',3,'c_g',1,'t_end',0.995,'step',0.5), ...
    struct('name','cpt4','P_delta',1e4,'c_PT',4,'c_g',1,'t_end',0.995,'step',0.5), ...
    struct('name','cpt5','P_delta',1e4,'c_PT',5,'c_g',1,'t_end',0.995,'step',0.5), ...
    struct('name','pd3e4_cpt3','P_delta',3e4,'c_PT',3,'c_g',1,'t_end',0.995,'step',0.5), ...
    struct('name','pd1e5_cpt3','P_delta',1e5,'c_PT',3,'c_g',1,'t_end',0.995,'step',0.5), ...
    struct('name','t997_cpt3','P_delta',1e4,'c_PT',3,'c_g',1,'t_end',0.997,'step',0.5), ...
    struct('name','t997_cpt4','P_delta',1e4,'c_PT',4,'c_g',1,'t_end',0.997,'step',0.5), ...
    struct('name','t997_pd3e4','P_delta',3e4,'c_PT',3,'c_g',1,'t_end',0.997,'step',0.5), ...
    struct('name','t999_cpt3','P_delta',1e4,'c_PT',3,'c_g',1,'t_end',0.999,'step',0.5)};

workers = min(workers,6);
pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= workers
    if ~isempty(pool), delete(pool); end
    parpool('Processes',workers);
end
rows = cell(numel(cfg),1);
dq = parallel.pool.DataQueue;
afterEach(dq,@show_row);
parfor i = 1:numel(cfg)
    c = cfg{i};
    opts = struct('dt',D.data.dt,'n_gen',numel(seeds),'n_steps',100, ...
        't_end',c.t_end,'transform_switch',0.9,'transform_step',c.step, ...
        'trajectory_seeds',seeds,'state_constraint',scene.state_constraint, ...
        'P_delta',c.P_delta,'P_u',1,'c_PT',c.c_PT,'c_g',c.c_g, ...
        'record_path',false,'record_control_trace',false);
    clock = tic;
    o = uniconflow_paper_guided_sample(N.net,D.data.states(:,1,1),opts);
    elapsed = toc(clock);
    rr = scene.state_constraint_rows_batch(o.states);
    qmax = squeeze(max(rr,[],[1 2]));
    point_bad = squeeze(sum(max(rr,[],1)>tol,2));
    rows{i} = struct('name',string(c.name),'P_delta',c.P_delta, ...
        'c_PT',c.c_PT,'c_g',c.c_g,'t_end',c.t_end, ...
        'transform_step',c.step,'failed_seed_safety',mean(qmax<=tol), ...
        'remaining_failed',sum(qmax>tol),'point_safety', ...
        1-sum(point_bad)/(101*numel(seeds)),'median_max_h',median(qmax), ...
        'worst_max_h',max(qmax),'seconds',elapsed);
    send(dq,[i,rows{i}.remaining_failed,rows{i}.worst_max_h,elapsed]);
end
report = struct2table(vertcat(rows{:}));
report = sortrows(report,{'remaining_failed','worst_max_h','seconds'}, ...
    {'ascend','ascend','ascend'});
out_dir = fileparts(stage1_file);
writetable(report,fullfile(out_dir,'Stage1_Safety_Screen_19Failed.csv'));
save(fullfile(out_dir,'Stage1_Safety_Screen_19Failed.mat'), ...
    'report','cfg','failed','seeds','old_max','tol','-v7.3');
disp(report);
end

function show_row(x)
fprintf('config %d: remaining %d, worst h %.6g, %.1f s\n', ...
    x(1),x(2),x(3),x(4));
end
