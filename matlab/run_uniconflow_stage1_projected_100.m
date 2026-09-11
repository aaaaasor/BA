function summary = run_uniconflow_stage1_projected_100(out_dir)
%RUN_UNICONFLOW_STAGE1_PROJECTED_100 Reproduce the archived 100-seed result.
if nargin<1||isempty(out_dir)
    out_dir='C:\Users\JieJi\BA\matlab\outputs\赛道UniConFlow';
end
N=load(fullfile(out_dir,'inputs','UniConFlow_Paper_Reconstructed_Net.mat'),'net');
D=load(fullfile(out_dir,'inputs','UniConFlow_Paper_Reconstructed_Dataset.mat'),'data');
C=load(fullfile(out_dir,'Run_Config_Stage1_Projected.mat'),'run_config');
op=C.run_config.options;seeds=C.run_config.trajectory_seeds(:);op.trajectory_seeds=seeds;
op.n_gen=numel(seeds);op.record_path=true;op.record_control_trace=true;
scene=uniconflow_paper_project_scene();op.state_constraint=scene.state_constraint;
clock=tic;
out=uniconflow_paper_guided_sample(N.net,D.data.states(:,1,1),op);
stage1_seconds=toc(clock);
scene5=uniconflow_paper_project_scene(struct('aggregate',false));
[Sproj,pdiag]=uniconflow_terminal_project_points(out.states,scene5, ...
    C.run_config.projection);
projection_seconds=toc(clock)-stage1_seconds;
out.mode='uniconflow_stage1_t0997_terminal_point_projection';
out.states_before_projection=out.states;
out.trajectory_before_projection=out.trajectory;
out.h_max_final_before_projection=out.h_max_final;
out.states=Sproj;out.trajectory=uniconflow_paper_pack('pack',Sproj,out.actions);
sd=N.net.sd_d(:);sd(sd==0)=1;
out.normalized_state=(out.trajectory-N.net.mu_d(:))./sd;
out.projection=pdiag;
R=scene5.state_constraint_rows_batch(out.states);
qmax=squeeze(max(R,[],[1 2]));
tol=C.run_config.projection.violation_tol;
bad=squeeze(sum(max(R,[],1)>tol,2));
out.h_max_final=qmax.';
metrics=struct('trajectory_safety',mean(qmax<=tol), ...
    'point_safety',1-sum(bad)/(size(Sproj,2)*numel(seeds)), ...
    'remaining_failed',sum(qmax>tol),'worst_max_h',max(qmax), ...
    'projected_trajectories',sum(pdiag.projected_points>0), ...
    'projected_points',sum(pdiag.projected_points), ...
    'projection_failures',sum(pdiag.failed_points), ...
    'stage1_seconds',stage1_seconds,'projection_seconds',projection_seconds, ...
    'total_seconds',toc(clock),'seconds_per_trajectory',toc(clock)/numel(seeds), ...
    'violation_tol',tol);
out.metrics=metrics;run_config=C.run_config;
save(fullfile(out_dir,'UniConFlow_Stage1_Projected_100_Full.mat'), ...
    'out','metrics','run_config','qmax','bad','pdiag','seeds','-v7.3');
summary=finalize_uniconflow_stage1_projected_archive(out_dir);
end
