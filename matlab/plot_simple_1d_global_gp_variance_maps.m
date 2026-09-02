function result = plot_simple_1d_global_gp_variance_maps()
% Compare posterior-variance maps for the 1000-candidate 1D experiment.
% The partial and all-data models use the same query grid and color limits.

root = fileparts(mfilename('fullpath'));
out = fullfile(root,'outputs','1d case全局GP训练阈值实验_500点');
partial_path = fullfile(out,'threshold_0p70_GlobalGP_Model.mat');
all_path = fullfile(out,'all_data_GlobalGP_Model.mat');
summary_path = fullfile(out,'Simple1D_GlobalGP_Threshold_Sweep.mat');
assert(isfile(partial_path) && isfile(all_path) && isfile(summary_path), ...
    'The matching 1000-candidate global-GP caches were not found.');

P = load(partial_path,'global_gp_model','selected_idx','signature');
A = load(all_path,'global_gp_model','selected_idx','signature');
S = load(summary_path,'sweep'); S = S.sweep;
candidate_count=S.cfg.n_train*S.cfg.n_time_slices;
assert(candidate_count == 500);
assert(P.global_gp_model.DataQuantity == 154);
assert(A.global_gp_model.DataQuantity == candidate_count);
assert(~S.metadata.loggp_used && ~S.metadata.local_experts_used && ...
    ~S.metadata.aggregation_used);

all_x = S.X(:,2);
pad = 0.08*range(all_x);
t_grid = linspace(0,1,151);
x_grid = linspace(min(all_x)-pad,max(all_x)+pad,181);
[T,X] = meshgrid(t_grid,x_grid);
Q = [T(:)';X(:)'];
prior = A.global_gp_model.SigmaF^2;
partial_var = reshape(predict_variance_batch(P.global_gp_model,Q),size(T))/prior;
all_var = reshape(predict_variance_batch(A.global_gp_model,Q),size(T))/prior;

f=figure('Visible','off','Color','w','Position',[40 80 1450 590]);
tl=tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
ax1=nexttile(tl); draw_map(ax1,T,X,partial_var, ...
    S.X(P.selected_idx,:),sprintf('Partial data: threshold 0.7, %d/%d points',numel(P.selected_idx),candidate_count));
ax2=nexttile(tl); draw_map(ax2,T,X,all_var, ...
    S.X(A.selected_idx,:),sprintf('All data: %d/%d points',numel(A.selected_idx),candidate_count));
clim(ax1,[0 1]);clim(ax2,[0 1]);
colormap(turbo(256));
cb=colorbar(ax2);cb.Label.String='Posterior variance / prior variance';
sgtitle(tl,'1D global exact-GP posterior variance maps (shared grid and color scale)');
output_path=fullfile(out,'Simple1D_GlobalGP_Variance_Map_Partial_vs_All.emf');
print(f,output_path,'-dmeta','-painters');close(f);

result=struct('output_path',output_path,'partial_threshold',0.7, ...
    'partial_training_points',numel(P.selected_idx), ...
    'all_training_points',numel(A.selected_idx), ...
    'grid_size',size(T),'prior_variance',prior, ...
    'partial_map_minmax',[min(partial_var,[],'all'),max(partial_var,[],'all')], ...
    'all_map_minmax',[min(all_var,[],'all'),max(all_var,[],'all')], ...
    'shared_color_limits',[0 1], ...
    'variance_definition','latent posterior variance divided by SigmaF^2');
save(fullfile(out,'Simple1D_GlobalGP_Variance_Map_Partial_vs_All.mat'), ...
    'result','t_grid','x_grid','partial_var','all_var','-v7.3');
fprintf('Saved %s\n',output_path);
disp(result);
end

function draw_map(ax,T,X,V,training,title_text)
imagesc(ax,T(1,:),X(:,1),V);set(ax,'YDir','normal');hold(ax,'on');
contour(ax,T,X,V,[.05 .1 .2 .4 .6 .8],'k:','LineWidth',.45);
scatter(ax,training(:,1),training(:,2),5,'k','filled');
xlabel(ax,'Generation time t');ylabel(ax,'State x');title(ax,title_text);
box(ax,'on');set(ax,'FontName','Times New Roman','FontSize',11);
end

function variance = predict_variance_batch(model,Q)
nq=size(Q,2);variance=zeros(nq,1);chunk=400;
for first=1:chunk:nq
    last=min(first+chunk-1,nq);
    k=kernel(model,model.X,Q(:,first:last));
    z=model.L\k;
    variance(first:last)=max(model.SigmaF^2-sum(z.^2,1)',0);
end
end

function k=kernel(model,X,Q)
dx=reshape(X,2,[],1)-reshape(Q,2,1,[]);
scaled=dx./reshape(model.SigmaL(:),2,1,1);
k=model.SigmaF^2*exp(-0.5*squeeze(sum(scaled.^2,1)));
if size(Q,2)==1,k=reshape(k,[],1);end
end
