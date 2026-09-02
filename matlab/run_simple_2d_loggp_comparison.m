function results = run_simple_2d_loggp_comparison(partial_threshold, output_subdir)
% Original two-state point-flow case, partial vs all data, both using LoG-GP.
% Independent experiment: no edits to get_config or any three-level cache.
root=fileparts(mfilename('fullpath')); addpath(root);
if nargin<1,partial_threshold=0.6;end
validateattributes(partial_threshold,{'numeric'},{'scalar','positive','finite'});
if nargin<2
 if partial_threshold==0.6,output_subdir='Simple2D_LoGGP_Comparison';
 else,output_subdir=['Simple2D_LoGGP_Threshold_',strrep(sprintf('%.8g',partial_threshold),'.','p')];end
end
reference_out=fullfile(root,'outputs','2d case 全部数据');
out=fullfile(root,'outputs',output_subdir);
if ~exist(out,'dir'),mkdir(out);end
maxNumCompThreads(1);
cfg=struct('version',1,'n_train',500,'n_time_slices',100, ...
 'n_rollouts',100,'n_rollout_steps',100,'data_seed',7, ...
 'hyperparameter_seed',8,'fit_seed',9,'rollout_seed',11, ...
 'state_dim',2,'guidance_enabled',false,'seed_filter_enabled',false, ...
 'time_varying_kernel',false,'rollout_parallel',false);
gp=struct('n_pretrain',450,'max_local_data_quantity',200, ...
 'max_local_gp_quantity',ceil(2*cfg.n_train*cfg.n_time_slices/200), ...
 'aggregation_method','GPOE','training_accuracy_threshold',0.6, ...
 'training_data_seed',cfg.data_seed,'o_ratio',0.1);
rng(cfg.data_seed);
source_points=randn(cfg.n_train,2);
target_points=sample_simple_2d_target(cfg.n_train);
training_times=linspace(0,1,cfg.n_time_slices)';
velocities=target_points-source_points;
x_slices=zeros(cfg.n_time_slices,cfg.n_train,2);y_slices=x_slices;
for ti=1:numel(training_times)
 x_slices(ti,:,:)=(1-training_times(ti))*source_points+training_times(ti)*target_points;
 y_slices(ti,:,:)=velocities;
end
X=[repmat(training_times,cfg.n_train,1),reshape(x_slices,[],2)];
Y=reshape(y_slices,[],2);
rng(cfg.rollout_seed);x_init=randn(cfg.n_rollouts,2);
save(fullfile(out,'Training_Data_and_Seeds.mat'),'cfg','source_points','target_points', ...
 'training_times','X','Y','x_init','-v7.3');
gp.hyperparameter_mat_path=fullfile(reference_out,'Shared_Hyperparameters.mat');
gp.hyperparameter_cache_signature=struct('cfg',cfg,'X',X,'Y',Y);
rng(cfg.hyperparameter_seed);
gp=optimize_gp_hyperparameters(x_slices,y_slices,gp,training_times);
assert(all(isfinite(gp.signal_std_vec))&&all(gp.signal_std_vec>0));
fprintf('Shared ell: %s; signal std %s; noise std %s\n', ...
 mat2str(gp.length_scale_mat,6),mat2str(gp.signal_std_vec,6),mat2str(gp.noise_std_vec,6));
labels={'Partial','AllData'};models=cell(1,2);runs=cell(1,2);audits=cell(1,2);
for mode=1:2
 active_gp=gp;active_gp.training_point_selection_enabled=(mode==1);
 if mode==1,active_gp.training_accuracy_threshold=partial_threshold;end
 signature=struct('cfg',cfg,'gp',active_gp,'X',X,'Y',Y);
 % The unchanged all-data model and rollout are shared with the baseline.
 cache_dir=out;if mode==2,cache_dir=reference_out;end
 model_path=fullfile(cache_dir,[labels{mode},'_Model.mat']);
 use_cache=false;
 if isfile(model_path)
  z=load(model_path,'signature');use_cache=isfield(z,'signature')&&isequaln(z.signature,signature);
 end
 if use_cache
  z=load(model_path,'model_collection','fit_seconds');model_collection=z.model_collection;fit_seconds=z.fit_seconds;clear z;
  fprintf('Loaded matching %s model\n',labels{mode});
 else
  rng(cfg.fit_seed);tt=tic;
  model_collection=fit_loggp_model(training_times,x_slices,y_slices,active_gp);
  fit_seconds=toc(tt);
  save(model_path,'model_collection','signature','fit_seconds','-v7.3');
 end
 % Check actual stored X/Y, not only acceptance counters.
 audits{mode}=audit_training(model_collection,X,Y,mode==2);
 query=[0;x_init(1,:)'];
 before=arrayfun(@(o) model_collection.model.output_models{o}.predict_variance(query),1:2);
 model_collection=strip_model_for_prediction(model_collection,['Simple2D ',labels{mode}]);
 after=arrayfun(@(o) model_collection.model.output_models{o}.predict_variance(query),1:2);
 assert(max(abs(before-after))<1e-12);
 models{mode}=model_collection;
 rollout_path=fullfile(cache_dir,[labels{mode},'_Rollout.mat']);use_cache=false;
 if isfile(rollout_path)
  z=load(rollout_path,'signature','x_init');
  use_cache=isfield(z,'signature')&&isequaln(z.signature,signature)&&isequaln(z.x_init,x_init);
 end
 if use_cache
  z=load(rollout_path,'run');run=z.run;clear z;
  fprintf('Loaded matching %s rollout\n',labels{mode});
 else
  tt=tic;[times,path]=plain_rollout(model_collection,x_init,linspace(0,1,cfg.n_rollout_steps+1)');
  rollout_seconds=toc(tt);
  fprintf('%s rollout completed in %.3f seconds. Evaluating variance...\n',labels{mode},rollout_seconds);
  variance=evaluate_rollout_uncertainty(model_collection,times,path);
  assert(all(isfinite(path),'all')&&all(isfinite(variance),'all')&&all(variance>=0,'all'));
  run=struct('times',times,'path',path,'variance',variance, ...
   'u',zeros(size(path)),'fit_seconds',fit_seconds,'rollout_seconds',rollout_seconds, ...
   'source_model',model_path,'guidance_enabled',false,'seed_filter_enabled',false);
  save(rollout_path,'run','signature','x_init','-v7.3');
 end
 runs{mode}=run;
 fprintf('MODEL %s: retained=%s of 50000/output; leaf count=%s; max leaves=%s\n', ...
  labels{mode},mat2str(audits{mode}.counts'),mat2str(audits{mode}.leaves'),mat2str(audits{mode}.max_leaves'));
end
assert(isequaln(runs{1}.times,runs{2}.times));
assert(isequaln(runs{1}.path(1,:,:),runs{2}.path(1,:,:)));
% Cross-query evaluation distinguishes model variance from trajectory drift.
cross_variance=cell(2,2);
for model_idx=1:2
 for path_idx=1:2
  if model_idx==path_idx,cross_variance{model_idx,path_idx}=runs{path_idx}.variance;
  else
   fprintf('Cross-evaluation: %s model on %s rollout\n',labels{model_idx},labels{path_idx});
   cross_variance{model_idx,path_idx}=evaluate_rollout_uncertainty( ...
    models{model_idx},runs{path_idx}.times,runs{path_idx}.path);
  end
 end
end
prior=reshape(gp.signal_std_vec.^2,1,1,2);
summary=zeros(2,8);curves=cell(2,1);
for mi=1:2
 v=runs{mi}.variance;nv=v./prior;
 own=mean(nv,3);
 common=mean(cat(2,cross_variance{mi,1},cross_variance{mi,2})./prior,3);
 curves{mi}=struct('own_mean',mean(own,2),'own_sd',std(own,0,2), ...
  'common_mean',mean(common,2),'common_sd',std(common,0,2));
 summary(mi,:)=[mean(audits{mi}.counts),100*mean(audits{mi}.counts)/size(X,1), ...
  mean(v,'all'),mean(v(end,:,:),'all'),mean(nv,'all'),mean(nv(end,:,:),'all'), ...
  mean(nv(1,:,:),'all'),runs{mi}.rollout_seconds/cfg.n_rollouts];
 % Direct spot re-predictions verify time/trajectory/output array mapping.
 for ti=[1 51 101]
  for qi=[1 50 100]
   q=[runs{mi}.times(ti);squeeze(runs{mi}.path(ti,qi,:))];
   for oi=1:2,assert(abs(models{mi}.model.output_models{oi}.predict_variance(q)-v(ti,qi,oi))<1e-12);end
  end
 end
end
comparison=array2table(summary,'RowNames',labels,'VariableNames', ...
 {'MeanTrainingPointsPerOutput','RetentionPercent','RawMeanVariance','RawTerminalVariance', ...
 'NormalizedMeanVariance','NormalizedTerminalVariance','NormalizedInitialVariance','RolloutSecondsPerSample'});
metadata=struct('state','single 2D point [x,y]','gp_input','[t,x,y]', ...
 'gp_output','[vx,vy]','kernel','stationary ARD squared exponential, shared hyperparameters', ...
 'normalization','per-output predictive variance divided by signal_std^2', ...
 'own_curve','each model queried on its own rollout', ...
 'common_curve','both models queried on identical union of the two rollout paths (200 locations per time)', ...
 'band','mean +/- one sample SD over 100 own-rollout samples', ...
 'time','serial RK4 wall time / 100, excludes fitting and variance diagnostics', ...
 'geometry_metrics','Not equated to CS/AS of a generated 65/80-point spatial trajectory');
initial_difference=(cross_variance{2,1}(1,:,:)-cross_variance{1,1}(1,:,:))./prior;
fprintf('Initial same-query AllData > Partial: %d / %d output queries\n',nnz(initial_difference>1e-10),numel(initial_difference));
disp(comparison);
results=struct('cfg',cfg,'gp',gp,'partial_training_accuracy_threshold',partial_threshold, ...
 'runs',{runs},'cross_variance',{cross_variance}, ...
 'curves',{curves},'comparison',comparison,'audits',{audits},'metadata',metadata,'x_init',x_init);
save(fullfile(out,'Comparison_Diagnostics.mat'),'results','-v7.3');
writetable(comparison,fullfile(out,'Comparison.csv'),'WriteRowNames',true);
draw_results(results,target_points,out);
check=load(fullfile(out,'Comparison_Diagnostics.mat'),'results');
assert(isequaln(check.results.comparison,comparison));
fprintf('COMPLETE: %s\n',out);
end

function audit=audit_training(mc,X,Y,is_all)
counts=zeros(2,1);leaves=counts;max_leaves=counts;indices=cell(2,1);
for oi=1:2
 gp=mc.model.output_models{oi};slots=gp.ActivatedGPNr(1:gp.ActivatedGPQuantity);
 xs=cell(numel(slots),1);ys=xs;
 for j=1:numel(slots)
  lf=gp.LocalGP_set{slots(j)};xs{j}=lf.X(:,1:lf.DataQuantity)';ys{j}=lf.Y(1:lf.DataQuantity,:);
 end
 xx=vertcat(xs{:});yy=vertcat(ys{:});[found,idx]=ismember(xx,X,'rows');
 assert(all(found)&&numel(unique(idx))==numel(idx)&&max(abs(yy-Y(idx,oi)))<1e-12);
 counts(oi)=size(xx,1);leaves(oi)=gp.ActivatedGPQuantity;max_leaves(oi)=gp.Max_LocalGP_Quantity;indices{oi}=idx;
 assert(counts(oi)==mc.n_added_per_output(oi)&&counts(oi)==gp.DataQuantity);
 if is_all,assert(counts(oi)==size(X,1)&&mc.n_skipped_per_output(oi)==0);end
end
audit=struct('counts',counts,'leaves',leaves,'max_leaves',max_leaves,'selected_indices',{indices});
end

function [times,path]=plain_rollout(mc,initial,times)
path=zeros(numel(times),size(initial,1),2);path(1,:,:)=initial;
for qi=1:size(initial,1)
 x=initial(qi,:)';
 for ti=1:numel(times)-1
  t=times(ti);h=times(ti+1)-t;
  k1=flow(mc,t,x);k2=flow(mc,t+h/2,x+h*k1/2);
  k3=flow(mc,t+h/2,x+h*k2/2);k4=flow(mc,t+h,x+h*k3);
  x=x+h*(k1+2*k2+2*k3+k4)/6;path(ti+1,qi,:)=x;
 end
 if mod(qi,20)==0,fprintf('Rollout %d/%d\n',qi,size(initial,1));end
end
end
function v=flow(mc,t,x)
v=zeros(2,1);for oi=1:2,v(oi)=mc.model.output_models{oi}.predict_mean([t;x]);end
end

function draw_results(r,target,out)
colors=[0.10 .36 .68;.83 .26 .12];labels={'Partial data','All data'};t=r.runs{1}.times;
f=figure('Visible','off','Color','w','Position',[50 50 1100 480],'Renderer','painters');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
for panel=1:2
 ax=nexttile;hold(ax,'on');h=gobjects(2,1);
 if panel==1
  for mi=1:2
   c=r.curves{mi};fill(ax,[t;flipud(t)],[max(c.own_mean-c.own_sd,0);flipud(c.own_mean+c.own_sd)], ...
    .82+.18*colors(mi,:),'EdgeColor','none','HandleVisibility','off');
  end
 end
 for mi=1:2
  if panel==1,v=r.curves{mi}.own_mean;else,v=r.curves{mi}.common_mean;end
  h(mi)=plot(ax,t,v,'Color',colors(mi,:),'LineWidth',2);
 end
 if panel==1,title(ax,'Each model on its own rollout (mean +/- SD)');
 else,title(ax,'Both models on identical query locations');end
 xlabel(ax,'Generation time t');ylabel(ax,'Mean variance / prior variance');xlim(ax,[0 1]);
 grid(ax,'on');box(ax,'on');set(ax,'FontSize',11,'FontName','Times New Roman');
 legend(ax,h,labels,'Location','best','Box','off');
end
sgtitle('Original 2D point flow | LoG-GP | no guidance | stationary kernel');
savefigs(f,fullfile(out,'Variance_Comparison_Normalized'));close(f);
f=figure('Visible','off','Color','w','Position',[50 50 1200 410],'Renderer','painters');
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
allxy=target;for mi=1:2,allxy=[allxy;squeeze(r.runs{mi}.path(end,:,:))];end %#ok<AGROW>
bounds=[min(allxy,[],1);max(allxy,[],1)];pad=.08*(bounds(2,:)-bounds(1,:));
for panel=1:3
 ax=nexttile;
 if panel==1,xy=target;col=[.25 .25 .25];titletext='Training target (500 points)';
 else,xy=squeeze(r.runs{panel-1}.path(end,:,:));col=colors(panel-1,:);titletext=[labels{panel-1},' (100 generated points)'];end
 scatter(ax,xy(:,1),xy(:,2),18,col,'filled');title(ax,titletext);
 axis(ax,'equal');xlim(ax,[bounds(1,1)-pad(1),bounds(2,1)+pad(1)]);ylim(ax,[bounds(1,2)-pad(2),bounds(2,2)+pad(2)]);
 grid(ax,'on');box(ax,'on');xlabel(ax,'x');ylabel(ax,'y');set(ax,'FontSize',11,'FontName','Times New Roman');
end
sgtitle('Original 2D point-flow case: target and generated endpoint distributions');
savefigs(f,fullfile(out,'Generated_Distributions'));close(f);
end
function savefigs(f,stem)
print(f,[stem,'.emf'],'-dmeta','-painters');
end


