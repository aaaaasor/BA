function sweep = run_simple_1d_threshold_sweep()
% One-dimensional flow-matching example for explaining uncertainty-based
% training-point selection. Static figures are exported as EMF only.
root = 'C:\Users\JieJi\BA\matlab';
addpath(root);
out = fullfile(root, 'outputs', '1d case训练阈值实验');
if ~exist(out, 'dir'), mkdir(out); end
maxNumCompThreads(1);

thresholds = [0.30, 0.25, 0.20, 0.15];
cfg = struct('version',1,'n_train',500,'n_time_slices',100, ...
    'n_rollouts',100,'n_rollout_steps',100,'data_seed',27, ...
    'hyperparameter_seed',28,'fit_seed',29,'rollout_seed',31, ...
    'test_seed',32,'state_dim',1,'guidance_enabled',false, ...
    'seed_filter_enabled',false,'time_varying_kernel',false, ...
    'rollout_parallel',false);
gp = struct('n_pretrain',450,'max_local_data_quantity',200, ...
    'max_local_gp_quantity',ceil(2*cfg.n_train*cfg.n_time_slices/200), ...
    'aggregation_method','GPOE','training_accuracy_threshold',thresholds(1), ...
    'training_data_seed',cfg.data_seed,'o_ratio',0.1);

rng(cfg.data_seed);
source_points = randn(cfg.n_train,1);
target_points = sample_target_1d(cfg.n_train);
training_times = linspace(0,1,cfg.n_time_slices)';
velocities = target_points-source_points;
x_slices = zeros(cfg.n_time_slices,cfg.n_train,1);
y_slices = x_slices;
for ti=1:numel(training_times)
    x_slices(ti,:,1)=(1-training_times(ti))*source_points + ...
        training_times(ti)*target_points;
    y_slices(ti,:,1)=velocities;
end
X=[repmat(training_times,cfg.n_train,1),reshape(x_slices,[],1)];
Y=reshape(y_slices,[],1);
rng(cfg.rollout_seed); x_init=randn(cfg.n_rollouts,1);
save(fullfile(out,'Training_Data_and_Seeds.mat'),'cfg','source_points', ...
    'target_points','training_times','X','Y','x_init','-v7.3');

gp.hyperparameter_mat_path=fullfile(out,'Shared_Hyperparameters.mat');
gp.hyperparameter_cache_signature=struct('cfg',cfg,'X',X,'Y',Y);
rng(cfg.hyperparameter_seed);
gp=optimize_gp_hyperparameters(x_slices,y_slices,gp,training_times);

labels=[arrayfun(@(x)sprintf('threshold_%.2f',x),thresholds, ...
    'UniformOutput',false),{'all_data'}];
n_models=numel(labels); models=cell(n_models,1);runs=cell(n_models,1);
audits=cell(n_models,1);

for mode=1:n_models
    active_gp=gp;
    active_gp.training_point_selection_enabled=(mode<=numel(thresholds));
    if active_gp.training_point_selection_enabled
        active_gp.training_accuracy_threshold=thresholds(mode);
    end
    signature=struct('cfg',cfg,'gp',active_gp,'X',X,'Y',Y);
    stem=strrep(labels{mode},'.','p');
    model_path=fullfile(out,[stem,'_Model.mat']);
    use_cache=false;
    if isfile(model_path)
        z=load(model_path,'signature');
        use_cache=isfield(z,'signature')&&isequaln(z.signature,signature);
    end
    if use_cache
        z=load(model_path,'model_collection','fit_seconds');
        model_collection=z.model_collection;fit_seconds=z.fit_seconds;clear z;
        fprintf('Loaded matching 1D model: %s\n',labels{mode});
    else
        rng(cfg.fit_seed);tt=tic;
        model_collection=fit_loggp_model(training_times,x_slices,y_slices,active_gp);
        fit_seconds=toc(tt);
        save(model_path,'model_collection','signature','fit_seconds','-v7.3');
    end
    audits{mode}=audit_training_1d(model_collection,X,Y,mode==n_models);
    model_collection=strip_model_for_prediction(model_collection,['Simple1D ',labels{mode}]);
    models{mode}=model_collection;

    rollout_path=fullfile(out,[stem,'_Rollout.mat']);use_cache=false;
    if isfile(rollout_path)
        z=load(rollout_path,'signature','x_init');
        use_cache=isfield(z,'signature')&&isequaln(z.signature,signature)&&isequaln(z.x_init,x_init);
    end
    if use_cache
        z=load(rollout_path,'run');run=z.run;clear z;
    else
        tt=tic;[times,path]=plain_rollout_1d(model_collection,x_init, ...
            linspace(0,1,cfg.n_rollout_steps+1)');
        rollout_seconds=toc(tt);
        variance=evaluate_rollout_uncertainty(model_collection,times,path);
        run=struct('times',times,'path',path,'variance',variance, ...
            'fit_seconds',fit_seconds,'rollout_seconds',rollout_seconds, ...
            'source_model',model_path,'guidance_enabled',false, ...
            'seed_filter_enabled',false);
        save(rollout_path,'run','signature','x_init','-v7.3');
    end
    runs{mode}=run;
end

% Fixed held-out flow-matching queries for a common prediction comparison.
rng(cfg.test_seed);n_test=200;test_source=randn(n_test,1);
test_target=sample_target_1d(n_test);test_times=linspace(0,1,20)';
test_X=zeros(n_test*numel(test_times),2);test_Y=zeros(size(test_X,1),1);
row=1;
for sample=1:n_test
    velocity=test_target(sample)-test_source(sample);
    for ti=1:numel(test_times)
        t=test_times(ti);test_X(row,:)=[t,(1-t)*test_source(sample)+t*test_target(sample)];
        test_Y(row)=velocity;row=row+1;
    end
end

summary=nan(n_models,13);
prior=gp.signal_std_vec(1)^2;
for mode=1:n_models
    run=runs{mode};v=run.variance;nv=v/prior;
    predictions=zeros(size(test_Y));test_variance=zeros(size(test_Y));
    model=models{mode}.model.output_models{1};
    for i=1:size(test_X,1)
        q=test_X(i,:)';predictions(i)=model.predict_mean(q);
        test_variance(i)=model.predict_variance(q)/prior;
    end
    info=dir(run.source_model);
    threshold=NaN;if mode<=numel(thresholds),threshold=thresholds(mode);end
    generated=squeeze(run.path(end,:,1))';
    summary(mode,:)=[threshold,audits{mode}.count, ...
        100*audits{mode}.count/size(X,1),info.bytes/2^20,run.fit_seconds, ...
        run.rollout_seconds/cfg.n_rollouts,fixed_kde_kl_1d(target_points,generated), ...
        sqrt(mean((predictions-test_Y).^2)),mean(test_variance), ...
        mean(v,'all'),mean(v(end,:,:),'all'),mean(nv,'all'),mean(nv(end,:,:),'all')];
end
comparison=array2table(summary,'RowNames',labels,'VariableNames', ...
    {'Threshold','TrainingPoints','RetentionPercent','ModelSizeMB', ...
    'FitTimeSeconds','RolloutSecondsPerSample','KL','HeldoutVelocityRMSE', ...
    'FixedQueryNormalizedVariance','RawMeanVariance','RawTerminalVariance', ...
    'NormalizedMeanVariance','NormalizedTerminalVariance'});
sweep=struct('cfg',cfg,'gp',gp,'thresholds',thresholds,'runs',{runs}, ...
    'audits',{audits},'comparison',comparison,'X',X,'Y',Y, ...
    'source_points',source_points,'target_points',target_points,'x_init',x_init, ...
    'metadata',struct('selection_rule', ...
    'add point iff predictive standard deviation exceeds threshold', ...
    'all_data_rule','training-point selection disabled', ...
    'same_data_hyperparameters_and_seeds',true,'video_generated',false));
save(fullfile(out,'Simple1D_Threshold_Sweep.mat'),'sweep','-v7.3');
writetable(comparison,fullfile(out,'Simple1D_Threshold_Sweep.csv'),'WriteRowNames',true);
disp(comparison);draw_1d_sweep(sweep,out);
fprintf('1D threshold sweep complete: %s\n',out);
end

function target=sample_target_1d(n)
component=rand(n,1)>0.5;
target=(-2.0+0.45*randn(n,1)).*(~component) + ...
    (2.0+0.55*randn(n,1)).*component;
end

function audit=audit_training_1d(mc,X,Y,is_all)
gp=mc.model.output_models{1};slots=gp.ActivatedGPNr(1:gp.ActivatedGPQuantity);
xs=cell(numel(slots),1);ys=xs;
for j=1:numel(slots)
    lf=gp.LocalGP_set{slots(j)};xs{j}=lf.X(:,1:lf.DataQuantity)';
    ys{j}=lf.Y(1:lf.DataQuantity,:);
end
xx=vertcat(xs{:});yy=vertcat(ys{:});[found,idx]=ismember(xx,X,'rows');
assert(all(found)&&numel(unique(idx))==numel(idx)&&max(abs(yy-Y(idx)))<1e-12);
if is_all,assert(size(xx,1)==size(X,1));end
audit=struct('count',size(xx,1),'leaves',gp.ActivatedGPQuantity, ...
    'max_leaves',gp.Max_LocalGP_Quantity,'selected_indices',idx);
end

function [times,path]=plain_rollout_1d(mc,initial,times)
path=zeros(numel(times),size(initial,1),1);path(1,:,1)=initial;
for qi=1:size(initial,1)
    x=initial(qi);
    for ti=1:numel(times)-1
        t=times(ti);h=times(ti+1)-t;
        k1=flow_1d(mc,t,x);k2=flow_1d(mc,t+h/2,x+h*k1/2);
        k3=flow_1d(mc,t+h/2,x+h*k2/2);k4=flow_1d(mc,t+h,x+h*k3);
        x=x+h*(k1+2*k2+2*k3+k4)/6;path(ti+1,qi,1)=x;
    end
end
end

function v=flow_1d(mc,t,x)
v=mc.model.output_models{1}.predict_mean([t;x]);
end

function value=fixed_kde_kl_1d(reference,generated)
n=numel(reference);bw=max(1.06*std(reference)*n^(-1/5),1e-3);
grid=linspace(min(reference)-6*bw,max(reference)+6*bw,1024)';
p=kde_1d(grid,reference,bw);q=kde_1d(grid,generated,bw);
p=p/sum(p);q=q/sum(q);floor_value=1e-300;
value=sum(p.*log(max(p,floor_value)./max(q,floor_value)));
end

function density=kde_1d(grid,samples,bw)
density=zeros(size(grid));
for i=1:numel(samples),density=density+exp(-.5*((grid-samples(i))/bw).^2);end
density=density/max(numel(samples),1);
end

function draw_1d_sweep(sweep,out)
n=numel(sweep.runs);colors=lines(n);
labels=[arrayfun(@(x)sprintf('threshold %.1f',x),sweep.thresholds, ...
    'UniformOutput',false),{'all data'}];

% Selected versus skipped FM pairs in (t,x).
f=figure('Visible','off','Color','w','Position',[30 30 1200 760],'Renderer','painters');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
display_idx=round(linspace(1,size(sweep.X,1),min(10000,size(sweep.X,1))));
for i=1:numel(sweep.thresholds)
    ax=nexttile;hold(ax,'on');
    scatter(ax,sweep.X(display_idx,1),sweep.X(display_idx,2),3,[.82 .82 .82],'filled');
    idx=sweep.audits{i}.selected_indices;
    scatter(ax,sweep.X(idx,1),sweep.X(idx,2),4,colors(i,:),'filled');
    title(ax,sprintf('%s: %.1f%% retained',labels{i},sweep.comparison.RetentionPercent(i)));
    xlabel(ax,'Generation time t');ylabel(ax,'State x');grid(ax,'on');box(ax,'on');
end
sgtitle('1D uncertainty-based training-point selection');
print(f,fullfile(out,'Simple1D_Selected_Training_Points.emf'),'-dmeta','-painters');close(f);

% Endpoint distributions.
density_grid=linspace(min(sweep.target_points)-2,max(sweep.target_points)+2,600)';
bw=max(1.06*std(sweep.target_points)*numel(sweep.target_points)^(-1/5),1e-3);
target_density=kde_1d(density_grid,sweep.target_points,bw);target_density=target_density/trapz(density_grid,target_density);
f=figure('Visible','off','Color','w','Position',[30 30 1200 760],'Renderer','painters');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
for i=1:n
    ax=nexttile;hold(ax,'on');generated=squeeze(sweep.runs{i}.path(end,:,1))';
    q=kde_1d(density_grid,generated,bw);q=q/trapz(density_grid,q);
    plot(ax,density_grid,target_density,'Color',[.25 .25 .25],'LineWidth',1.8);
    plot(ax,density_grid,q,'Color',colors(i,:),'LineWidth',1.8);
    title(ax,labels{i});xlabel(ax,'x');ylabel(ax,'Density');grid(ax,'on');box(ax,'on');
end
sgtitle('1D target and generated endpoint distributions');
print(f,fullfile(out,'Simple1D_Threshold_Endpoint_Distributions.emf'),'-dmeta','-painters');close(f);

% Complete rollout trajectories.
f=figure('Visible','off','Color','w','Position',[30 30 1200 760],'Renderer','painters');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
for i=1:n
    ax=nexttile;hold(ax,'on');p=squeeze(sweep.runs{i}.path(:,:,1));
    plot(ax,sweep.runs{i}.times,p,'Color',.58+.42*colors(i,:),'LineWidth',.5);
    title(ax,labels{i});xlabel(ax,'Generation time t');ylabel(ax,'State x');grid(ax,'on');box(ax,'on');
end
sgtitle('1D LoG-GP trajectories from identical initial samples');
print(f,fullfile(out,'Simple1D_Threshold_Rollout_Trajectories.emf'),'-dmeta','-painters');close(f);

% Normalized variance curves.
f=figure('Visible','off','Color','w','Position',[80 80 820 500],'Renderer','painters');ax=axes(f);hold(ax,'on');
prior=sweep.gp.signal_std_vec(1)^2;
for i=1:n
    curve=mean(squeeze(sweep.runs{i}.variance(:,:,1))/prior,2);
    plot(ax,sweep.runs{i}.times,curve,'LineWidth',1.8,'Color',colors(i,:));
end
xlabel(ax,'Generation time t');ylabel(ax,'Mean variance / prior variance');
grid(ax,'on');box(ax,'on');legend(ax,labels,'Location','best','Box','off');
title(ax,'1D normalized posterior variance under different thresholds');
print(f,fullfile(out,'Simple1D_Threshold_Variance_Curves.emf'),'-dmeta','-painters');close(f);

% Threshold tradeoff.
C=sweep.comparison;idx=1:numel(sweep.thresholds);
f=figure('Visible','off','Color','w','Position',[80 80 1000 680],'Renderer','painters');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
fields={'RetentionPercent','ModelSizeMB','HeldoutVelocityRMSE','KL'};
ylabs={'Retained training points (%)','Model cache size (MB)','Held-out velocity RMSE','KL divergence'};
for j=1:4
    ax=nexttile;plot(ax,sweep.thresholds,C{idx,fields{j}},'-o','LineWidth',1.8);
    xlabel(ax,'Training uncertainty threshold');ylabel(ax,ylabs{j});grid(ax,'on');box(ax,'on');
end
sgtitle('1D threshold--storage--quality tradeoff');
print(f,fullfile(out,'Simple1D_Threshold_Tradeoff.emf'),'-dmeta','-painters');close(f);
end
