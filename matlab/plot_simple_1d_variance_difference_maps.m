function result = plot_simple_1d_variance_difference_maps()
% Difference maps: threshold-model normalized posterior variance minus the
% all-data normalized posterior variance, on the identical fixed grid.

root=fileparts(mfilename('fullpath'));
out=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
source=fullfile(out,'Simple1D_GlobalGP_Variance_Maps_All_Thresholds.mat');
assert(isfile(source),'Run plot_simple_1d_all_threshold_variance_maps first.');
A=load(source,'result');S=A.result;
assert(isequal(S.thresholds,[.7 .6 .5 .4 .3]));
all_map=S.maps{6};n=numel(S.maps);difference_maps=cell(n,1);
stats=zeros(n,4);
for i=1:n
    difference_maps{i}=S.maps{i}-all_map;
    % Only round-off at machine precision is permitted below zero.
    assert(min(difference_maps{i},[],'all')>-1e-10);
    difference_maps{i}=max(difference_maps{i},0);
    stats(i,:)=[mean(difference_maps{i},'all'), ...
        mean(difference_maps{i}(:,end)),max(difference_maps{i},[],'all'), ...
        sqrt(mean(difference_maps{i}.^2,'all'))];
end
cmax=max(cellfun(@(z)max(z,[],'all'),difference_maps));
if cmax<=0,cmax=1;end

f=figure('Visible','off','Color','w','Position',[20 30 1800 1050]);
tl=tiledlayout(f,2,3,'Padding','compact','TileSpacing','compact');
axes_list=gobjects(n,1);
for i=1:n
    ax=nexttile(tl);axes_list(i)=ax;
    imagesc(ax,S.t_grid,S.x_grid,difference_maps{i});set(ax,'YDir','normal');
    xlabel(ax,'Generation time t');ylabel(ax,'State x');
    if i<=5
        title(ax,sprintf('threshold %.1f - all data; mean \\Delta = %.5f', ...
            S.thresholds(i),stats(i,1)));
    else
        title(ax,'all data - all data = 0');
    end
    clim(ax,[0 cmax]);box(ax,'on');set(ax,'FontName','Times New Roman','FontSize',10);
end
colormap(turbo(256));cb=colorbar(axes_list(end));
cb.Label.String='Difference in normalized posterior variance';
sgtitle(tl,'1D global exact GP: variance difference relative to the all-data model');
emf=fullfile(out,'Simple1D_GlobalGP_Variance_Difference_Maps_vs_All.emf');
print(f,emf,'-dmeta','-painters');close(f);

row_names={'threshold_0.7','threshold_0.6','threshold_0.5', ...
    'threshold_0.4','threshold_0.3','all_data'};
comparison=array2table([[S.thresholds(:);NaN],stats], ...
    'RowNames',row_names,'VariableNames',{'Threshold','MeanDifference', ...
    'TerminalGridMeanDifference','MaxDifference','RMSDifference'});
result=struct('definition','threshold variance minus all-data variance', ...
    'difference_maps',{difference_maps},'statistics',comparison, ...
    't_grid',S.t_grid,'x_grid',S.x_grid,'shared_color_limits',[0 cmax], ...
    'source_mat',source,'emf_path',emf);
save(fullfile(out,'Simple1D_GlobalGP_Variance_Difference_Maps_vs_All.mat'), ...
    'result','-v7.3');
writetable(comparison,fullfile(out, ...
    'Simple1D_GlobalGP_Variance_Difference_Maps_vs_All.csv'),'WriteRowNames',true);
disp(comparison);fprintf('Saved %s\n',emf);
end
