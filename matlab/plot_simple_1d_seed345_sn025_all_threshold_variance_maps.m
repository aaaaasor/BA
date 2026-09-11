function result = plot_simple_1d_seed345_sn025_all_threshold_variance_maps()
% Plot normalized posterior-variance maps for the four configured manual-HP
% thresholds and the all-data model.

root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs',['1d case' char([20840 23616 71 80])]);
D=load(fullfile(out,'Training_Data_and_Seeds.mat'),'X','Y','cfg');
H=load(fullfile(out,'Manual_Hyperparameters.mat'),'SigmaL','SigmaF','SigmaN');
X=D.X;Y=D.Y;n=size(X,1);assert(n==2000);
S=load(fullfile(out,'Simple1D_GlobalGP_Threshold_Sweep.mat'),'sweep');
thresholds=S.sweep.thresholds;nm=numel(thresholds)+1;   % read from the sweep, never hardcode
labels=cell(nm,1);models=cell(nm,1);indices=cell(nm,1);
for i=1:numel(thresholds)
    model_path=fullfile(out,sprintf('threshold_0p%02d_GlobalGP_Model.mat',round(100*thresholds(i))));
    if isfile(model_path)
        A=load(model_path,'global_gp_model','selected_idx');
        models{i}=A.global_gp_model;indices{i}=A.selected_idx;
    else
        [models{i},indices{i}]=fit_selected(X,Y,thresholds(i),H.SigmaN,H.SigmaF,H.SigmaL(:,1));
    end
    labels{i}=sprintf('threshold %.2f: %d/%d points (%.1f%%)', ...
        thresholds(i),numel(indices{i}),n,100*numel(indices{i})/n);
end
A=load(fullfile(out,'all_data_GlobalGP_Model.mat'),'global_gp_model','selected_idx');
models{nm}=A.global_gp_model;indices{nm}=A.selected_idx;
labels{nm}=sprintf('all data: %d/%d points (100%%)',numel(indices{nm}),n);

pad=.08*range(X(:,2));t_grid=linspace(0,1,151);x_grid=linspace(min(X(:,2))-pad,max(X(:,2))+pad,181);
[T,Xgrid]=meshgrid(t_grid,x_grid);Q=[T(:)';Xgrid(:)'];prior=H.SigmaF^2;
maps=cell(nm,1);map_stats=zeros(nm,3);tbl_stats=zeros(nm,4);
for i=1:nm
    maps{i}=reshape(predict_var(models{i},Q),size(T))/prior;
    map_stats(i,:)=[mean(maps{i},'all'),min(maps{i},[],'all'),max(maps{i},[],'all')];
    % Table 8 of the thesis: raw and normalised variance, whole map and the
    % terminal (t = 1) column, on the SAME shared grid for every model.
    tbl_stats(i,:)=[mean(maps{i},'all')*prior, mean(maps{i}(:,end))*prior, ...
        mean(maps{i},'all'), mean(maps{i}(:,end))];
end

f=figure('Visible','off','Color','w','Position',[20 30 1800 1050]);
tl=tiledlayout(f,2,3,'Padding','compact','TileSpacing','compact');
axes_list=gobjects(nm,1);
for i=1:nm
    ax=nexttile(tl);axes_list(i)=ax;
    imagesc(ax,t_grid,x_grid,maps{i});set(ax,'YDir','normal');hold(ax,'on');
    contour(ax,T,Xgrid,maps{i},[.05 .1 .2 .4 .6 .8],'k:','LineWidth',.4);
    scatter(ax,X(indices{i},1),X(indices{i},2),4,'k','filled');
    xlabel(ax,'Generation time t');ylabel(ax,'State x');title(ax,labels{i});
    clim(ax,[0 1]);box(ax,'on');set(ax,'FontName','Times New Roman','FontSize',10);
end
colormap(turbo(256));cb=colorbar(axes_list(nm));
cb.Label.String='Posterior variance / prior variance';
sgtitle(tl,'1D global exact-GP posterior variance maps for all thresholds');
emf=fullfile(out,'Simple1D_GlobalGP_Variance_Maps_All_Thresholds.emf');
print(f,emf,'-dmeta','-painters');close(f);

row_names=[arrayfun(@(z)sprintf('threshold_%.2f',z),thresholds,'UniformOutput',false),{'all_data'}];
stats=array2table([[thresholds(:);NaN],cellfun(@numel,indices), ...
    100*cellfun(@numel,indices)/n,map_stats], ...
    'RowNames',row_names, ...
    'VariableNames',{'Threshold','TrainingPoints','RetentionPercent', ...
    'MapMeanNormalizedVariance','MapMinNormalizedVariance','MapMaxNormalizedVariance'});
result=struct('thresholds',thresholds,'labels',{labels},'selected_indices',{indices}, ...
    'maps',{maps},'statistics',stats,'t_grid',t_grid,'x_grid',x_grid, ...
    'shared_color_limits',[0 1],'prior_variance',prior,'emf_path',emf, ...
    'variance_definition','latent posterior variance / SigmaF^2');
save(fullfile(out,'Simple1D_GlobalGP_Variance_Maps_All_Thresholds.mat'),'result','-v7.3');
writetable(stats,fullfile(out,'Simple1D_GlobalGP_Variance_Maps_All_Thresholds.csv'), ...
    'WriteRowNames',true);
disp(stats);fprintf('Saved %s\n',emf);

% Thesis table 8: shared-map variance statistics.  Regenerated here so it can
% never go stale relative to the thresholds actually used by the sweep.
tbl9=array2table([[thresholds(:);NaN],cellfun(@numel,indices), ...
    100*cellfun(@numel,indices)/n,tbl_stats], ...
    'VariableNames',{'Threshold','TrainingPoints','RetentionPercent', ...
    'MapMeanRawVariance','MapTerminalRawVariance', ...
    'MapMeanNormalizedVariance','MapTerminalNormalizedVariance'});
writetable(tbl9,fullfile(out,'Simple1D_GlobalGP_Table9_Map_Variance_Statistics.csv'));
save(fullfile(out,'Simple1D_GlobalGP_Table9_Map_Variance_Statistics.mat'),'tbl9','-v7.3');
disp(tbl9);
end

function [m,idx]=fit_selected(X,Y,thr,sn,sf,sl)
n=size(X,1);g=LocalGP_MultiOutput(2,1,n,sn,sf,sl);idx=zeros(n,1);c=0;
for i=1:n
    if g.DataQuantity==0||sqrt(g.predict_variance(X(i,:)'))>thr
        assert(g.addPoint(X(i,:)',Y(i))==1);c=c+1;idx(c)=i;
    end
end
idx=idx(1:c);m=compact(g);clear g;
end

function m=compact(g)
n=g.DataQuantity;m=struct('X',g.X(:,1:n),'L',g.L(1:n,1:n), ...
    'SigmaF',g.SigmaF,'SigmaL',g.SigmaL,'DataQuantity',n);
end

function v=predict_var(m,Q)
nq=size(Q,2);v=zeros(nq,1);chunk=400;
for first=1:chunk:nq
    last=min(first+chunk-1,nq);k=kern(m,m.X,Q(:,first:last));z=m.L\k;
    v(first:last)=max(m.SigmaF^2-sum(z.^2,1)',0);
end
end

function k=kern(m,X,Q)
d=reshape(X,2,[],1)-reshape(Q,2,1,[]);d=d./reshape(m.SigmaL(:),2,1,1);
k=m.SigmaF^2*exp(-.5*squeeze(sum(d.^2,1)));if size(Q,2)==1,k=reshape(k,[],1);end
end

