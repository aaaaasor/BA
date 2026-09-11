function result = plot_simple_1d_seed345_sn025_maps_with_rollouts(line_alpha, n_show)
% line_alpha : opacity of the rollout lines (default 0.15).  NOTE: EMF has no
%              alpha channel, so this only affects screen/PNG rendering; the
%              EMF is checked separately and n_show is the EMF-safe control.
% n_show     : how many of the 1000 trajectories to draw (default 200).  With
%              all 1000 opaque lines the variance map underneath is invisible.
if nargin < 1 || isempty(line_alpha), line_alpha = 0.15; end
if nargin < 2 || isempty(n_show),     n_show = 200; end
% Overlay each model's own 100 rollout trajectories on the four configured
% threshold variance maps and the all-data variance map.

root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs',['1d case' char([20840 23616 71 80])]);
D=load(fullfile(out,'Training_Data_and_Seeds.mat'),'X','x_init');
V=load(fullfile(out,'Simple1D_GlobalGP_Variance_Maps_All_Thresholds.mat'),'result');
V=V.result;thresholds=V.thresholds;assert(numel(thresholds)==4);   % thresholds now come from the sweep

nm=numel(thresholds)+1;paths=cell(nm,1);times=[];
for i=1:numel(thresholds)
    p=fullfile(out,sprintf('threshold_0p%02d_GlobalGP_Rollout.mat',round(100*thresholds(i))));
    A=load(p,'run');paths{i}=A.run.path;
    if isempty(times),times=A.run.times;else,assert(isequal(times,A.run.times));end
end
A=load(fullfile(out,'all_data_GlobalGP_Rollout.mat'),'run');
paths{nm}=A.run.path;assert(isequal(times,A.run.times));
for i=1:nm
    assert(isequal(squeeze(paths{i}(1,:,1))',D.x_init));
end

candidate_count=size(D.X,1);labels=cell(nm,1);
for i=1:numel(thresholds)
    labels{i}=sprintf('threshold %.2f: %d/%d points',thresholds(i),numel(V.selected_indices{i}),candidate_count);
end
labels{nm}=sprintf('all data: %d/%d points',candidate_count,candidate_count);

f=figure('Visible','off','Color','w','Position',[20 30 1800 1050]);
tl=tiledlayout(f,2,3,'Padding','compact','TileSpacing','compact');
axes_list=gobjects(nm,1);
for i=1:nm
    ax=nexttile(tl);axes_list(i)=ax;
    imagesc(ax,V.t_grid,V.x_grid,V.maps{i});set(ax,'YDir','normal');hold(ax,'on');
    contour(ax,V.t_grid,V.x_grid,V.maps{i},[.05 .1 .2 .4 .6 .8], ...
        'k:','LineWidth',.3);
    scatter(ax,D.X(V.selected_indices{i},1),D.X(V.selected_indices{i},2),3,'k','filled');
    P=squeeze(paths{i}(:,:,1));
    keep=round(linspace(1,size(P,2),min(n_show,size(P,2))));
    P=P(:,keep);
    plot(ax,times,P,'Color',[1 1 1 line_alpha],'LineWidth',.35);
    scatter(ax,times(1)*ones(size(P,2),1),P(1,:)',7,[0 .65 1],'filled');
    scatter(ax,times(end)*ones(size(P,2),1),P(end,:)',9,[1 0 1],'filled');
    xlabel(ax,'Generation time t');ylabel(ax,'State x');title(ax,labels{i});
    clim(ax,[0 1]);box(ax,'on');set(ax,'FontName','Times New Roman','FontSize',10);
end
colormap(turbo(256));cb=colorbar(axes_list(nm));
cb.Label.String='Posterior variance / prior variance';
sgtitle(tl,'1D global exact GP: own rollout trajectories on all variance maps');
emf=fullfile(out,'Simple1D_GlobalGP_All_Threshold_Variance_Maps_with_Own_Rollouts.emf');
print(f,emf,'-dmeta','-painters');
png=fullfile(out,'Simple1D_GlobalGP_All_Threshold_Variance_Maps_with_Own_Rollouts_MobilePreview.png');
exportgraphics(f,png,'Resolution',180);close(f);

result=struct('thresholds',thresholds,'labels',{labels},'times',times, ...
    'paths',{paths},'selected_indices',{V.selected_indices}, ...
    'same_initial_samples',true,'shared_color_limits',[0 1], ...
    'line_legend','white=rollout, cyan=start, magenta=terminal, black=training point', ...
    'variance_map_source',fullfile(out,'Simple1D_GlobalGP_Variance_Maps_All_Thresholds.mat'), ...
    'emf_path',emf,'png_path',png);
save(fullfile(out,'Simple1D_GlobalGP_All_Threshold_Variance_Maps_with_Own_Rollouts.mat'), ...
    'result','-v7.3');
fprintf('Saved %s\n',emf);disp(result);
end

