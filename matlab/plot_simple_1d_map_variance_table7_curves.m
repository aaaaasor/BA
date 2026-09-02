function result=plot_simple_1d_map_variance_table7_curves()
% Plot Table-7 common-grid variance-map statistics against data retention.
root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
V=load(fullfile(out,'Simple1D_GlobalGP_Variance_Maps_All_Thresholds.mat'),'result');
S=load(fullfile(out,'Simple1D_GlobalGP_Threshold_Sweep.mat'),'sweep');
H=load(fullfile(out,'Manual_Hyperparameters.mat'),'SigmaF');
V=V.result;S=S.sweep;prior=H.SigmaF^2;

n=numel(V.maps);
retention=S.comparison.RetentionPercent;
mean_norm=zeros(n,1);term_norm=zeros(n,1);
for i=1:n
    M=V.maps{i};
    mean_norm(i)=mean(M,'all');
    term_norm(i)=mean(M(:,end));
end
mean_raw=prior*mean_norm;term_raw=prior*term_norm;
threshold_labels=[arrayfun(@(z)sprintf('thr. %.2f',z),V.thresholds, ...
    'UniformOutput',false),{'all'}];

f=figure('Visible','off','Color','w','Position',[40 70 1500 590], ...
    'Renderer','painters');
tl=tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');

ax=nexttile(tl);hold(ax,'on');
plot(ax,retention,mean_raw,'-o','LineWidth',2,'MarkerSize',7);
plot(ax,retention,term_raw,'-s','LineWidth',2,'MarkerSize',7);
for i=1:n
    text(ax,retention(i),mean_raw(i)+0.018,threshold_labels{i}, ...
        'HorizontalAlignment','center','FontSize',9);
end
xlabel(ax,'Training-point retention (%)');ylabel(ax,'Posterior variance');
title(ax,'Raw common-grid variance-map statistics');
legend(ax,{'Grid mean','Terminal slice mean'},'Location','northeast','Box','off');
grid(ax,'on');box(ax,'on');xlim(ax,[15 105]);

ax=nexttile(tl);hold(ax,'on');
plot(ax,retention,mean_norm,'-o','LineWidth',2,'MarkerSize',7);
plot(ax,retention,term_norm,'-s','LineWidth',2,'MarkerSize',7);
for i=1:n
    text(ax,retention(i),mean_norm(i)+0.0045,threshold_labels{i}, ...
        'HorizontalAlignment','center','FontSize',9);
end
xlabel(ax,'Training-point retention (%)');
ylabel(ax,'Posterior variance / prior variance');
title(ax,'Normalized common-grid variance-map statistics');
legend(ax,{'Grid mean','Terminal slice mean'},'Location','northeast','Box','off');
grid(ax,'on');box(ax,'on');xlim(ax,[15 105]);

sgtitle(tl,'1D exact GP: Table 7 variance-map statistics versus retained training data');
set(findall(f,'Type','axes'),'FontName','Times New Roman','FontSize',11);
emf=fullfile(out,'Simple1D_GlobalGP_Table7_Map_Variance_Curves.emf');
png=fullfile(out,'Simple1D_GlobalGP_Table7_Map_Variance_Curves_MobilePreview.png');
print(f,emf,'-dmeta','-painters');exportgraphics(f,png,'Resolution',180);close(f);

result=struct('retention_percent',retention,'thresholds',[V.thresholds(:);NaN], ...
    'mean_raw_variance',mean_raw,'terminal_raw_variance',term_raw, ...
    'mean_normalized_variance',mean_norm, ...
    'terminal_normalized_variance',term_norm, ...
    'strictly_decreasing_mean',all(diff(mean_norm)<0), ...
    'strictly_decreasing_terminal',all(diff(term_norm)<0), ...
    'emf_path',emf,'png_path',png);
save(fullfile(out,'Simple1D_GlobalGP_Table7_Map_Variance_Curves.mat'), ...
    'result','-v7.3');
fprintf('Saved %s\nSaved %s\n',emf,png);disp(result);
end
