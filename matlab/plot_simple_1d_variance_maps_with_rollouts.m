function result = plot_simple_1d_variance_maps_with_rollouts()
% Overlay each model's own generated trajectories on its variance map.
% Compare threshold 0.3 against the all-data exact GP using identical x_init.

root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
D=load(fullfile(out,'Training_Data_and_Seeds.mat'),'X','Y','x_init','cfg');
H=load(fullfile(out,'Shared_Hyperparameters.mat'),'SigmaL','SigmaF','SigmaN');
V=load(fullfile(out,'Simple1D_GlobalGP_Variance_Maps_All_Thresholds.mat'),'result');
V=V.result;X=D.X;Y=D.Y;n=size(X,1);assert(n==1000);

% Rebuild threshold 0.3 only for this diagnostic; do not add it to Table 6.
[partial_model,partial_idx]=fit_selected(X,Y,.3,H.SigmaN,H.SigmaF,H.SigmaL(:,1));
times=linspace(0,1,D.cfg.n_rollout_steps+1)';
partial_path=plain_rollout(partial_model,D.x_init,times);

A=load(fullfile(out,'all_data_GlobalGP_Rollout.mat'),'run');
all_path=A.run.path;all_times=A.run.times;
assert(isequal(times,all_times));
assert(isequal(squeeze(partial_path(1,:,1))',D.x_init));
assert(isequal(squeeze(all_path(1,:,1))',D.x_init));

maps={V.maps{5},V.maps{6}};indices={partial_idx,(1:n)'};
paths={partial_path,all_path};titles={ ...
    sprintf('threshold 0.3: %d/%d points',numel(partial_idx),n), ...
    sprintf('all data: %d/%d points',n,n)};

f=figure('Visible','off','Color','w','Position',[30 80 1500 650]);
tl=tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
axes_list=gobjects(2,1);
for i=1:2
    ax=nexttile(tl);axes_list(i)=ax;
    imagesc(ax,V.t_grid,V.x_grid,maps{i});set(ax,'YDir','normal');hold(ax,'on');
    contour(ax,V.t_grid,V.x_grid,maps{i},[.05 .1 .2 .4 .6 .8], ...
        'k:','LineWidth',.35);
    scatter(ax,X(indices{i},1),X(indices{i},2),4,'k','filled');
    P=squeeze(paths{i}(:,:,1));
    plot(ax,times,P,'Color',[.92 .92 .92],'LineWidth',.45);
    scatter(ax,times(1)*ones(size(P,2),1),P(1,:)',10,[0 .65 1],'filled');
    scatter(ax,times(end)*ones(size(P,2),1),P(end,:)',12,[1 0 1],'filled');
    xlabel(ax,'Generation time t');ylabel(ax,'State x');title(ax,titles{i});
    clim(ax,[0 1]);box(ax,'on');set(ax,'FontName','Times New Roman','FontSize',11);
end
colormap(turbo(256));cb=colorbar(axes_list(2));
cb.Label.String='Posterior variance / prior variance';
sgtitle(tl,['1D global exact GP: own rollout trajectories overlaid on ', ...
    'posterior-variance maps']);
emf=fullfile(out,'Simple1D_GlobalGP_Variance_Maps_with_Own_Rollouts_0p3_vs_All.emf');
print(f,emf,'-dmeta','-painters');close(f);

result=struct('partial_threshold',.3,'partial_training_points',numel(partial_idx), ...
    'all_training_points',n,'times',times,'partial_path',partial_path, ...
    'all_path',all_path,'partial_selected_indices',partial_idx, ...
    'same_initial_samples',true,'shared_color_limits',[0 1], ...
    'line_legend','white=rollout, cyan=start, magenta=terminal, black=training point', ...
    'emf_path',emf);
save(fullfile(out,'Simple1D_GlobalGP_Variance_Maps_with_Own_Rollouts_0p3_vs_All.mat'), ...
    'result','-v7.3');
fprintf('Saved %s\n',emf);disp(result);
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
    'alpha',g.alpha(1:n,:),'SigmaF',g.SigmaF,'SigmaL',g.SigmaL);
end

function path=plain_rollout(m,initial,times)
path=zeros(numel(times),numel(initial),1);path(1,:,1)=initial;
for q=1:numel(initial)
    x=initial(q);
    for i=1:numel(times)-1
        t=times(i);h=times(i+1)-t;
        k1=pred(m,[t;x]);k2=pred(m,[t+h/2;x+h*k1/2]);
        k3=pred(m,[t+h/2;x+h*k2/2]);k4=pred(m,[t+h;x+h*k3]);
        x=x+h*(k1+2*k2+2*k3+k4)/6;path(i+1,q,1)=x;
    end
end
end

function mu=pred(m,q)
k=kern(m,m.X,q);mu=m.alpha'*k;
end

function k=kern(m,X,Q)
d=reshape(X,2,[],1)-reshape(Q,2,1,[]);d=d./reshape(m.SigmaL(:),2,1,1);
k=m.SigmaF^2*exp(-.5*squeeze(sum(d.^2,1)));if size(Q,2)==1,k=reshape(k,[],1);end
end
