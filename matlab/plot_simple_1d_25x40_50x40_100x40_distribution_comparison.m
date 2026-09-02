function result=plot_simple_1d_25x40_50x40_100x40_distribution_comparison()
root=fileparts(mfilename('fullpath'));
base=fullfile(root,'outputs');
p25=fullfile(base,'1d case全局GP训练阈值实验_25x40', ...
    'Simple1D_GlobalGP_t0_t1_True_Distributions.mat');
p50=fullfile(base,'1d case全局GP训练阈值实验_50x40', ...
    'Simple1D_GlobalGP_50x40_t0_t1_True_Distributions.mat');
p100=fullfile(base,'1d case全局GP训练阈值实验_100x40', ...
    'Simple1D_GlobalGP_100x40_t0_t1_True_Distributions.mat');
A=load(p25,'result');B=load(p50,'result');C=load(p100,'result');
A=A.result;B=B.result;C=C.result;
assert(isequal(A.x_grid,B.x_grid,C.x_grid));
assert(isequal(A.initial_samples,B.initial_samples,C.initial_samples));
x=A.x_grid;P=A.target_true_density;P=P/trapz(x,P);
Q25=A.terminal_kde(:,end);Q25=Q25/trapz(x,Q25);
Q50=B.terminal_kde(:,end);Q50=Q50/trapz(x,Q50);
Q100=C.terminal_kde(:,end);Q100=Q100/trapz(x,Q100);
kl25=trapz(x,P.*log(max(P,1e-300)./max(Q25,1e-300)));
kl50=trapz(x,P.*log(max(P,1e-300)./max(Q50,1e-300)));
kl100=trapz(x,P.*log(max(P,1e-300)./max(Q100,1e-300)));

f=figure('Visible','off','Color','w','Position',[30 60 1520 610],'Renderer','painters');
tl=tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
ax=nexttile(tl);hold(ax,'on');
plot(ax,x,A.source_true_density,'k-','LineWidth',2.4);
plot(ax,x,A.initial_kde,'--','Color',[.1 .45 .85],'LineWidth',2);
xlabel(ax,'State x');ylabel(ax,'Probability density');
title(ax,'t = 0: shared 1000 initial samples');
legend(ax,{'True N(0,1)','Empirical source'},'Location','northwest','Box','off');
grid(ax,'on');box(ax,'on');xlim(ax,[x(1) x(end)]);

ax=nexttile(tl);hold(ax,'on');
plot(ax,x,P,'k-','LineWidth',2.6);
plot(ax,x,Q25,'--','Color',[.15 .45 .85],'LineWidth',2.1);
plot(ax,x,Q50,'-','Color',[.15 .65 .30],'LineWidth',2.1);
plot(ax,x,Q100,'-','Color',[.65 .20 .75],'LineWidth',2.1);
xlabel(ax,'State x');ylabel(ax,'Probability density');
title(ax,'t = 1: all-data models with more independent training pairs');
legend(ax,{'True target mixture',sprintf('25x40, KL = %.4f',kl25), ...
    sprintf('50x40, KL = %.4f',kl50),sprintf('100x40, KL = %.4f',kl100)}, ...
    'Location','northeastoutside','Box','off');
grid(ax,'on');box(ax,'on');xlim(ax,[x(1) x(end)]);
sgtitle(tl,'Effect of increasing independent FM training pairs (shared hyperparameters and seeds)');
set(findall(f,'Type','axes'),'FontName','Times New Roman','FontSize',11);

out=fullfile(base,'1d case全局GP训练阈值实验_100x40');
emf=fullfile(out,'Simple1D_GlobalGP_25x40_50x40_100x40_Distributions.emf');
png=fullfile(out,'Simple1D_GlobalGP_25x40_50x40_100x40_Distributions_MobilePreview.png');
print(f,emf,'-dmeta','-painters');exportgraphics(f,png,'Resolution',180);close(f);
result=struct('x_grid',x,'true_target_density',P,'all_25x40_density',Q25, ...
    'all_50x40_density',Q50,'all_100x40_density',Q100, ...
    'analytic_kl_25x40',kl25,'analytic_kl_50x40',kl50, ...
    'analytic_kl_100x40',kl100, ...
    'relative_kl_reduction_25_to_100_percent',100*(kl25-kl100)/kl25, ...
    'same_hyperparameters',true,'same_initial_samples',true, ...
    'emf_path',emf,'png_path',png);
save(fullfile(out,'Simple1D_GlobalGP_25x40_50x40_100x40_Distributions.mat'), ...
    'result','-v7.3');
fprintf('KL: 25x40 %.6f; 50x40 %.6f; 100x40 %.6f; 25->100 reduction %.2f%%\n', ...
    kl25,kl50,kl100,result.relative_kl_reduction_25_to_100_percent);
fprintf('Saved %s\nSaved %s\n',emf,png);
end
