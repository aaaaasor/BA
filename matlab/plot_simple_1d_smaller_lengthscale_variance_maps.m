function result = plot_simple_1d_smaller_lengthscale_variance_maps()
% Repeat all threshold variance maps with both ARD length scales at 0.90x.

root=fileparts(mfilename('fullpath'));
base=fullfile(root,'outputs','1d case全局GP训练阈值实验_500点');
out=fullfile(base,'length_scale_0p90');if ~exist(out,'dir'),mkdir(out);end
D=load(fullfile(base,'Training_Data_and_Seeds.mat'),'X','Y','cfg');
H=load(fullfile(base,'Shared_Hyperparameters.mat'),'SigmaL','SigmaF','SigmaN');
X=D.X;Y=D.Y;n=size(X,1);assert(n==500);
length_scale_factor=.90;manual_ell=length_scale_factor*H.SigmaL(:,1);
thresholds=[.7 .6 .5 .4 .3];labels=cell(6,1);models=cell(6,1);indices=cell(6,1);
for i=1:numel(thresholds)
    [models{i},indices{i}]=fit_selected(X,Y,thresholds(i),H.SigmaN,H.SigmaF,manual_ell);
    labels{i}=sprintf('threshold %.1f: %d/%d points (%.1f%%)', ...
        thresholds(i),numel(indices{i}),n,100*numel(indices{i})/n);
end
[models{6},indices{6}]=fit_all(X,Y,H.SigmaN,H.SigmaF,manual_ell);
labels{6}=sprintf('all data: %d/%d points (100%%)',numel(indices{6}),n);

pad=.08*range(X(:,2));t_grid=linspace(0,1,151);x_grid=linspace(min(X(:,2))-pad,max(X(:,2))+pad,181);
[T,Xgrid]=meshgrid(t_grid,x_grid);Q=[T(:)';Xgrid(:)'];prior=H.SigmaF^2;
maps=cell(6,1);map_stats=zeros(6,3);
for i=1:6
    maps{i}=reshape(predict_var(models{i},Q),size(T))/prior;
    map_stats(i,:)=[mean(maps{i},'all'),min(maps{i},[],'all'),max(maps{i},[],'all')];
end

f=figure('Visible','off','Color','w','Position',[20 30 1800 1050]);
tl=tiledlayout(f,2,3,'Padding','compact','TileSpacing','compact');
axes_list=gobjects(6,1);
for i=1:6
    ax=nexttile(tl);axes_list(i)=ax;
    imagesc(ax,t_grid,x_grid,maps{i});set(ax,'YDir','normal');hold(ax,'on');
    contour(ax,T,Xgrid,maps{i},[.05 .1 .2 .4 .6 .8],'k:','LineWidth',.4);
    scatter(ax,X(indices{i},1),X(indices{i},2),4,'k','filled');
    xlabel(ax,'Generation time t');ylabel(ax,'State x');title(ax,labels{i});
    clim(ax,[0 1]);box(ax,'on');set(ax,'FontName','Times New Roman','FontSize',10);
end
colormap(turbo(256));cb=colorbar(axes_list(6));
cb.Label.String='Posterior variance / prior variance';
sgtitle(tl,'1D global exact GP: all thresholds with length scale 0.90x');
emf=fullfile(out,'Simple1D_GlobalGP_Variance_Maps_LengthScale_0p90.emf');
print(f,emf,'-dmeta','-painters');close(f);

stats=array2table([[thresholds(:);NaN],cellfun(@numel,indices), ...
    100*cellfun(@numel,indices)/n,map_stats], ...
    'RowNames',{'threshold_0.7','threshold_0.6','threshold_0.5', ...
    'threshold_0.4','threshold_0.3','all_data'}, ...
    'VariableNames',{'Threshold','TrainingPoints','RetentionPercent', ...
    'MapMeanNormalizedVariance','MapMinNormalizedVariance','MapMaxNormalizedVariance'});
result=struct('thresholds',thresholds,'labels',{labels},'selected_indices',{indices}, ...
    'maps',{maps},'statistics',stats,'t_grid',t_grid,'x_grid',x_grid, ...
    'shared_color_limits',[0 1],'prior_variance',prior,'emf_path',emf, ...
    'variance_definition','latent posterior variance / SigmaF^2', ...
    'length_scale_factor',length_scale_factor,'base_length_scale',H.SigmaL(:,1), ...
    'manual_length_scale',manual_ell);
save(fullfile(out,'Simple1D_GlobalGP_Variance_Maps_LengthScale_0p90.mat'),'result','-v7.3');
writetable(stats,fullfile(out,'Simple1D_GlobalGP_Variance_Maps_LengthScale_0p90.csv'), ...
    'WriteRowNames',true);
disp(stats);fprintf('Saved %s\n',emf);
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

function [m,idx]=fit_all(X,Y,sn,sf,sl)
n=size(X,1);g=LocalGP_MultiOutput(2,1,n,sn,sf,sl);g.add_Alldata(X,Y);
m=compact(g);idx=(1:n)';clear g;
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
