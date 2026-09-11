function result = run_fm_small128_screen(force_retrain)
%RUN_FM_SMALL128_SCREEN Train/evaluate a 128^3 FM without overwriting baseline.
if nargin<1,force_retrain=false;end
root='C:\Users\JieJi\BA\matlab';
out_dir=fullfile(root,'outputs','NN_Size_Sweep','FM_Small128');
if ~isfolder(out_dir),mkdir(out_dir);end
net_file=fullfile(out_dir,'FM_Small128_Net.mat');
rollout_file=fullfile(out_dir,'FM_Small128_Rollout.mat');
result_file=fullfile(out_dir,'FM_Small128_Result.mat');

if isfile(net_file)&&~force_retrain
    N=load(net_file,'net');net=N.net;
    fprintf('Loaded cached Small128 network.\n');
else
    net=safeflow_nn_train(20000,false,struct('hidden_width',128, ...
        'weight_init_seed',20260101,'batch_size',256,'learning_rate',1e-3));
    save(net_file,'net','-v7.3');
end

R0=load(fullfile(root,'outputs','赛车fm','Racing_FM_NN_Rollout.mat'),'rollout');
seeds=R0.rollout.trajectory_seeds;
rollout=safeflow_nn_rollout(net,'fm',100,struct( ...
    'trajectory_seeds',seeds,'record_path',true,'n_steps',100,'t_max',0.996));
metrics=safeflow_nn_metrics(rollout);
save(rollout_file,'rollout','-v7.3');

M0=load(fullfile(root,'outputs','赛车fm','Racing_FM_NN_Metrics.mat'),'metrics');
medium=M0.metrics;
nparam=sum(cellfun(@numel,net.P));
medium_net=load(fullfile(root,'outputs','赛车fm','Racing_FM_NN_Net.mat'),'net');
nparam_medium=sum(cellfun(@numel,medium_net.net.P));
comparison=table({'Small128';'Medium256'},[nparam;nparam_medium], ...
    [100*metrics.safety;100*medium.safety],[metrics.kl;medium.kl], ...
    [metrics.cs;medium.cs],[metrics.as;medium.as], ...
    [metrics.time_seconds;medium.time_seconds], ...
    'VariableNames',{'Network','Parameters','Safety_percent','KL','CS','AS','Time_s'});
result=struct('network','261-128-128-128-260','parameters',nparam, ...
    'n_train_steps',20000,'training_seed',20260101,'trajectory_seeds',seeds, ...
    'metrics',metrics,'comparison',comparison,'net_file',net_file, ...
    'rollout_file',rollout_file,'created_at',char(datetime('now')));
save(result_file,'result','-v7.3');
writetable(comparison,fullfile(out_dir,'FM_Small128_vs_Medium256.csv'));

fid=fopen(fullfile(out_dir,'RESULTS.txt'),'w');c=onCleanup(@()fclose(fid));
fprintf(fid,'FM Small128 (261-128^3-260), 20000 training steps, 100 fixed seeds\n');
fprintf(fid,'Parameters = %d\nSafety = %.8f%%\nKL = %.8f\nCS = %.8f\nAS = %.8f\nTime = %.8f s/trajectory\n', ...
    nparam,100*metrics.safety,metrics.kl,metrics.cs,metrics.as,metrics.time_seconds);
disp(comparison);
end
