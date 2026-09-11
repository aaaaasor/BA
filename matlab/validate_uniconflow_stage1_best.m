function result=validate_uniconflow_stage1_best()
%VALIDATE_UNICONFLOW_STAGE1_BEST Validate the best parameter-only setting on 100 seeds.
root='C:\Users\JieJi\BA\matlab';
formal=fullfile(root,'outputs','赛道UniConFlow_Paper_FastCertified_100');
N=load(fullfile(formal,'inputs','UniConFlow_Paper_Reconstructed_Net.mat'),'net');
D=load(fullfile(formal,'inputs','UniConFlow_Paper_Reconstructed_Dataset.mat'),'data');
L=load(fullfile(formal,'Stage1_All_Trajectories.mat'),'stage_cache');
seeds=L.stage_cache.trajectory_seeds(:);scene=uniconflow_paper_project_scene();
opts=struct('dt',D.data.dt,'n_gen',numel(seeds),'n_steps',100, ...
    't_end',0.995,'transform_switch',0.9,'transform_step',0.5, ...
    'trajectory_seeds',seeds,'state_constraint',scene.state_constraint, ...
    'P_delta',1e4,'P_u',1,'c_PT',3,'c_g',1, ...
    'record_path',false,'record_control_trace',false);
clock=tic;o=uniconflow_paper_guided_sample(N.net,D.data.states(:,1,1),opts);
seconds=toc(clock);n=numel(seeds);bad=zeros(n,1);rollout_bad=zeros(n,1);
rollout_rmse=zeros(n,1);maxh=-inf(n,1);
for q=1:n
    for k=1:101
        h=scene.state_constraint(o.states(:,k,q),k-1);hm=max(h(:));
        maxh(q)=max(maxh(q),hm);bad(q)=bad(q)+(hm>1e-8);
    end
    [~,Sr]=uniconflow_paper_reconstruct_actions(o.states(:,:,q),o.spec);
    rollout_rmse(q)=sqrt(mean((Sr-o.states(:,:,q)).^2,'all'));
    for k=1:101
        h=scene.state_constraint(Sr(:,k),k-1);
        rollout_bad(q)=rollout_bad(q)+(max(h(:))>1e-8);
    end
end
metrics=struct('n',n,'seconds',seconds,'seconds_per_trajectory',seconds/n, ...
    'trajectory_safety',mean(bad==0),'point_safety',1-sum(bad)/(101*n), ...
    'mean_bad_nodes',mean(bad),'rollout_trajectory_safety',mean(rollout_bad==0), ...
    'rollout_point_safety',1-sum(rollout_bad)/(101*n), ...
    'rollout_mean_bad_nodes',mean(rollout_bad), ...
    'rollout_rmse',mean(rollout_rmse),'median_hmax',median(maxh), ...
    'p95_hmax',prctile(maxh,95),'median_g',median(o.g_final), ...
    'p95_g',prctile(o.g_final,95));
result=struct('metrics',metrics,'bad_nodes',bad,'rollout_bad_nodes',rollout_bad, ...
    'max_h',maxh,'rollout_rmse',rollout_rmse,'seeds',seeds,'options',opts,'out',o);
out_dir=fullfile(root,'outputs','赛道UniConFlow_Stage1_ParamSweep');
save(fullfile(out_dir,'Stage1_Best_100Seeds.mat'),'result','-v7.3');
fid=fopen(fullfile(out_dir,'Stage1_Best_100Seeds.txt'),'w');
cleanup=onCleanup(@()fclose(fid));fields=fieldnames(metrics);
for k=1:numel(fields),fprintf(fid,'%s = %.12g\n',fields{k},metrics.(fields{k}));end
disp(metrics);
end
