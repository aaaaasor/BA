function result=plot_simple_1d_target_training_data()
root=fileparts(mfilename('fullpath'));
base=fullfile(root,'outputs');
d25=fullfile(base,'1d case全局GP训练阈值实验_25x40');
d50=fullfile(base,'1d case全局GP训练阈值实验_50x40');
A=load(fullfile(d25,'Training_Data_and_Seeds.mat'),'target_points');
B=load(fullfile(d50,'Training_Data_and_Seeds.mat'),'target_points');
targets={A.target_points(:),B.target_points(:)};
names={'25 training pairs','50 training pairs'};
x=linspace(-4.5,4.5,1600)';
truth=.5*normal_pdf(x,-2,.45)+.5*normal_pdf(x,2,.55);
bw=.25; colors=[.15 .45 .85;.15 .65 .30];
counts=zeros(2,2);kdes=zeros(numel(x),2);

f=figure('Visible','off','Color','w','Position',[50 70 1500 600],'Renderer','painters');
tl=tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
for i=1:2
    y=targets{i};counts(i,:)=[sum(y<0),sum(y>0)];
    kdes(:,i)=kde_density(x,y,bw);
    ax=nexttile(tl);hold(ax,'on');
    plot(ax,x,truth,'k-','LineWidth',2.5);
    plot(ax,x,kdes(:,i),'Color',colors(i,:),'LineWidth',2.1);
    ymark=-.025*ones(size(y));
    scatter(ax,y,ymark,30,colors(i,:),'filled','MarkerFaceAlpha',.85);
    xline(ax,0,':','Color',[.35 .35 .35],'LineWidth',1);
    xlabel(ax,'Target state x_1');ylabel(ax,'Probability density');
    title(ax,sprintf('%s: left %d, right %d',names{i},counts(i,1),counts(i,2)));
    if i==2
        legend(ax,{'True target mixture','Training-target KDE','Training targets'}, ...
            'Location','northeastoutside','Box','off');
    end
    grid(ax,'on');box(ax,'on');xlim(ax,[x(1),x(end)]);ylim(ax,[-.05,.85]);
end
sgtitle(tl,'Independent target samples used to construct the FM training data');
set(findall(f,'Type','axes'),'FontName','Times New Roman','FontSize',11);

out=d50;
emf=fullfile(out,'Simple1D_Target_Training_Data_25_vs_50.emf');
png=fullfile(out,'Simple1D_Target_Training_Data_25_vs_50_MobilePreview.png');
print(f,emf,'-dmeta','-painters');exportgraphics(f,png,'Resolution',180);close(f);
result=struct('x_grid',x,'true_density',truth,'targets_25',targets{1}, ...
    'targets_50',targets{2},'kde_25',kdes(:,1),'kde_50',kdes(:,2), ...
    'kde_bandwidth',bw,'left_right_counts',counts, ...
    'left_right_percent',100*counts./sum(counts,2), ...
    'time_slices_per_pair',40, ...
    'note','Each independent target is counted once; time slices are not duplicated.', ...
    'emf_path',emf,'png_path',png);
save(fullfile(out,'Simple1D_Target_Training_Data_25_vs_50.mat'),'result','-v7.3');
fprintf('25 pairs left/right %d/%d; 50 pairs left/right %d/%d\n', ...
    counts(1,1),counts(1,2),counts(2,1),counts(2,2));
fprintf('Saved %s\nSaved %s\n',emf,png);
end

function p=normal_pdf(x,mu,sigma)
p=exp(-.5*((x-mu)/sigma).^2)/(sqrt(2*pi)*sigma);
end
function p=kde_density(x,s,bw)
p=zeros(size(x));for i=1:numel(s),p=p+normal_pdf(x,s(i),bw);end
p=p/max(numel(s),1);
end
