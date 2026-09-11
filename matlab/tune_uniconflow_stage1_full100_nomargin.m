function report=tune_uniconflow_stage1_full100_nomargin(out_dir,workers)
%TUNE_UNICONFLOW_STAGE1_FULL100_NOMARGIN Exact 100-seed, no-margin sweep.
if nargin<1||isempty(out_dir),out_dir='C:\Users\JieJi\BA\matlab\outputs\赛道UniConFlow';end
if nargin<2,workers=6;end
N=load(fullfile(out_dir,'inputs','UniConFlow_Paper_Reconstructed_Net.mat'),'net');
D=load(fullfile(out_dir,'inputs','UniConFlow_Paper_Reconstructed_Dataset.mat'),'data');
C=load(fullfile(out_dir,'Run_Config_Stage1_Projected.mat'),'run_config');
seeds=C.run_config.trajectory_seeds(:);scene=uniconflow_paper_project_scene();tol=5e-4;
cfg={ ...
 struct('name','step_0p4','P_delta',1e4,'c_PT',3,'step',0.4), ...
 struct('name','step_0p3','P_delta',1e4,'c_PT',3,'step',0.3), ...
 struct('name','Pdelta_1p5e4','P_delta',1.5e4,'c_PT',3,'step',0.5), ...
 struct('name','Pdelta_2e4','P_delta',2e4,'c_PT',3,'step',0.5), ...
 struct('name','cpt_2p8','P_delta',1e4,'c_PT',2.8,'step',0.5), ...
 struct('name','cpt_3p2','P_delta',1e4,'c_PT',3.2,'step',0.5)};
workers=min(workers,6);p=gcp('nocreate');
if isempty(p)||p.NumWorkers~=workers,if ~isempty(p),delete(p);end,parpool('Processes',workers);end
rows=cell(numel(cfg),1);dq=parallel.pool.DataQueue;afterEach(dq,@show_row);
net=N.net;data=D.data;state_constraint=scene.state_constraint;
state_constraint_rows_batch=scene.state_constraint_rows_batch;
parfor j=1:numel(cfg)
 c=cfg{j};op=struct('dt',data.dt,'n_gen',100,'n_steps',100,'t_end',0.997, ...
  'transform_switch',0.9,'transform_step',c.step,'trajectory_seeds',seeds, ...
  'state_constraint',state_constraint,'P_delta',c.P_delta,'P_u',1, ...
  'c_PT',c.c_PT,'c_g',1,'record_path',false,'record_control_trace',false);
 clock=tic;o=uniconflow_paper_guided_sample(net,data.states(:,1,1),op);elapsed=toc(clock);
 R=state_constraint_rows_batch(o.states);
 qmax=squeeze(max(R,[],[1 2]));bad=squeeze(sum(max(R,[],1)>tol,2));
 rows{j}=struct('name',string(c.name),'P_delta',c.P_delta,'c_PT',c.c_PT, ...
  'transform_step',c.step,'trajectory_safety',mean(qmax<=tol), ...
  'remaining_failed',sum(qmax>tol),'violating_points',sum(bad), ...
  'point_safety',1-sum(bad)/10100,'worst_max_h',max(qmax),'seconds',elapsed);
 send(dq,[j,rows{j}.remaining_failed,rows{j}.violating_points,rows{j}.worst_max_h,elapsed]);
end
report=struct2table(vertcat(rows{:}));
report=sortrows(report,{'violating_points','remaining_failed','worst_max_h'}, ...
 {'ascend','ascend','ascend'});
writetable(report,fullfile(out_dir,'Stage1_Full100_NoMargin_Sweep.csv'));
save(fullfile(out_dir,'Stage1_Full100_NoMargin_Sweep.mat'),'report','cfg','seeds','tol','-v7.3');
disp(report);
end
function show_row(x)
fprintf('cfg%d failed=%d points=%d worst=%.6g sec=%.1f\n',x(1),x(2),x(3),x(4),x(5));
end
