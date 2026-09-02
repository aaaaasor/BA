function paths = plot_simple_1d_seed345_sn025_rollout_trajectories()
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_50x40_seed345_sn025');
A=load(fullfile(out,'Simple1D_GlobalGP_Threshold_Sweep.mat'),'sweep');
S=A.sweep;n=numel(S.runs);
labels=[arrayfun(@(z)sprintf('threshold %.2f',z),S.thresholds, ...
    'UniformOutput',false),{'all data'}];
% Keep trajectory panels categorically coloured.  The single-hue gradient
% is used only for the endpoint-distribution comparison figure.
colors=lines(n);
% Plot the same evenly spaced 100 trajectories in every panel.
n_show=min(100,size(S.runs{1}.path,2));
show_idx=unique(round(linspace(1,size(S.runs{1}.path,2),n_show)));
f=figure('Visible','off','Color','w','Position',[25 30 1450 820],'Renderer','painters');
tl=tiledlayout(f,2,3,'Padding','compact','TileSpacing','compact');
for i=1:n
    ax=nexttile(tl);hold(ax,'on');
    p=squeeze(S.runs{i}.path(:,show_idx,1));
    curve_color=0.18*[1 1 1]+0.82*colors(i,:);
    plot(ax,S.runs{i}.times,p,'Color',curve_color,'LineWidth',0.55);
    yline(ax,-2,'--','Color',[.35 .35 .35],'LineWidth',1.0);
    yline(ax, 2,'--','Color',[.35 .35 .35],'LineWidth',1.0);
    scatter(ax,repmat(S.runs{i}.times(end),1,numel(show_idx)),p(end,:), ...
        8,colors(i,:),'filled');
    if i<n
        ttl=sprintf('%s: %.2f%% retained',labels{i},S.comparison.RetentionPercent(i));
    else
        ttl='all data: 100% retained';
    end
    title(ax,ttl);xlabel(ax,'Generation time t');ylabel(ax,'State x');
    xlim(ax,[0 1]);ylim(ax,[-4.5 4.5]);grid(ax,'on');box(ax,'on');
end
nexttile(tl,6);axis off;
text(.05,.72,sprintf('Same %d of %d initial samples in every panel', ...
    numel(show_idx),size(S.runs{1}.path,2)),'FontSize',11);
text(.05,.55,'Dashed lines: true target-mode centers x = -2 and x = 2','FontSize',11);
text(.05,.38,'Dots: generated terminal samples at t = 1','FontSize',11);
sgtitle(tl,'1D exact-GP generation trajectories (seed 345, \sigma_n = 0.25)');
set(findall(f,'Type','axes'),'FontName','Times New Roman','FontSize',10.5);
emf=fullfile(out,'Simple1D_GlobalGP_Rollout_Trajectories.emf');
png=fullfile(out,'Simple1D_GlobalGP_Rollout_Trajectories_MobilePreview.png');
print(f,emf,'-dmeta','-painters');exportgraphics(f,png,'Resolution',180);close(f);
paths=struct('emf',emf,'png',png,'shown_indices',show_idx,'n_total',size(S.runs{1}.path,2));
save(fullfile(out,'Simple1D_GlobalGP_Rollout_Trajectories_PlotInfo.mat'),'paths');
fprintf('Saved %s\nSaved %s\n',emf,png);
end
