function plot_threshold_table_curves()
root='C:\Users\JieJi\BA\matlab\outputs';
plot_case(fullfile(root,'2d case训练阈值实验','Simple2D_Threshold_Sweep.mat'), ...
    fullfile(root,'2d case训练阈值实验','Simple2D_Table4_Threshold_Curves.emf'), ...
    '2D LoG-GP threshold comparison');
plot_case(fullfile(root,'1d case全局GP训练阈值实验','Simple1D_GlobalGP_Threshold_Sweep.mat'), ...
    fullfile(root,'1d case全局GP训练阈值实验','Simple1D_GlobalGP_Table5_Threshold_Curves.emf'), ...
    '1D global exact-GP threshold comparison');
end

function plot_case(mat_path,output_path,figure_title)
A=load(mat_path,'sweep');S=A.sweep;C=S.comparison;
n=numel(S.thresholds);[x,order]=sort(S.thresholds(:));
rows=order;full_row=n+1;
blue=[0.10 0.36 0.68];red=[0.82 0.25 0.15];black=[0.15 0.15 0.15];
x_curve=[0;x];
f=figure('Visible','off','Color','w','Position',[60 60 1050 700]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

ax=nexttile;hold(ax,'on');
h1=plot(ax,x_curve,[C.RetentionPercent(full_row);C.RetentionPercent(rows)], ...
    '-o','Color',blue,'LineWidth',1.8, ...
    'MarkerFaceColor',blue,'MarkerSize',6);
scatter(ax,0,C.RetentionPercent(full_row),52,black,'d','filled');
xlabel(ax,'Training uncertainty threshold');ylabel(ax,'Training-point retention (%)');
title(ax,'Data retention (threshold 0 = all data)');
format_axes(ax,x_curve);

ax=nexttile;hold(ax,'on');
h1=plot(ax,x_curve,[C.RolloutSecondsPerSample(full_row);C.RolloutSecondsPerSample(rows)], ...
    '-o','Color',blue,'LineWidth',1.8, ...
    'MarkerFaceColor',blue,'MarkerSize',6);
scatter(ax,0,C.RolloutSecondsPerSample(full_row),52,black,'d','filled');
xlabel(ax,'Training uncertainty threshold');ylabel(ax,'Time (s/trajectory)');
title(ax,'Rollout time (threshold 0 = all data)');
format_axes(ax,x_curve);

ax=nexttile;hold(ax,'on');
h1=plot(ax,x_curve,[C.RawMeanVariance(full_row);C.RawMeanVariance(rows)], ...
    '-o','Color',blue,'LineWidth',1.8, ...
    'MarkerFaceColor',blue,'MarkerSize',6);
h2=plot(ax,x_curve,[C.RawTerminalVariance(full_row);C.RawTerminalVariance(rows)], ...
    '-s','Color',red,'LineWidth',1.8, ...
    'MarkerFaceColor',red,'MarkerSize',6);
scatter(ax,0,C.RawMeanVariance(full_row),52,black,'d','filled');
scatter(ax,0,C.RawTerminalVariance(full_row),52,black,'d','filled');
xlabel(ax,'Training uncertainty threshold');ylabel(ax,'Predictive variance');
title(ax,'Raw posterior variance');
legend(ax,[h1 h2],{'Mean','Terminal'},'Location','best','Box','off');format_axes(ax,x_curve);

ax=nexttile;hold(ax,'on');
h1=plot(ax,x_curve,[C.NormalizedMeanVariance(full_row);C.NormalizedMeanVariance(rows)], ...
    '-o','Color',blue,'LineWidth',1.8, ...
    'MarkerFaceColor',blue,'MarkerSize',6);
h2=plot(ax,x_curve,[C.NormalizedTerminalVariance(full_row);C.NormalizedTerminalVariance(rows)], ...
    '-s','Color',red,'LineWidth',1.8, ...
    'MarkerFaceColor',red,'MarkerSize',6);
scatter(ax,0,C.NormalizedMeanVariance(full_row),52,black,'d','filled');
scatter(ax,0,C.NormalizedTerminalVariance(full_row),52,black,'d','filled');
xlabel(ax,'Training uncertainty threshold');ylabel(ax,'Variance / prior variance');
title(ax,'Normalized posterior variance');
legend(ax,[h1 h2],{'Mean','Terminal'},'Location','best','Box','off');format_axes(ax,x_curve);

sgtitle(figure_title);
print(f,output_path,'-dmeta','-painters');close(f);
fprintf('Saved %s\n',output_path);
end

function format_axes(ax,x)
grid(ax,'on');box(ax,'on');xticks(ax,x);xlim(ax,[min(x)-.04*range(x),max(x)+.04*range(x)]);
set(ax,'FontName','Times New Roman','FontSize',11);
end
