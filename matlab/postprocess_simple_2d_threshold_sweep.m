function postprocess_simple_2d_threshold_sweep()
root='C:\Users\JieJi\BA\matlab';
out=fullfile(root,'outputs','2d case训练阈值实验');
A=load(fullfile(out,'Simple2D_Threshold_Sweep.mat'),'sweep');sweep=A.sweep;
T=load(fullfile(root,'outputs','2d case部分数据','Training_Data_and_Seeds.mat'),'X');X=T.X;
colors=lines(numel(sweep.runs));
labels=[arrayfun(@(x)sprintf('threshold %.1f',x),sweep.thresholds, ...
    'UniformOutput',false),{'all data'}];

% Selection projections for the four thresholds. The two output GPs select
% independently, so the union of their retained indices is displayed.
f=figure('Visible','off','Color','w','Position',[30 30 1350 650]);
tiledlayout(2,4,'Padding','compact','TileSpacing','compact');
display_idx=round(linspace(1,size(X,1),min(8000,size(X,1))));
for threshold_idx=1:numel(sweep.thresholds)
    selected=unique(vertcat( ...
        sweep.partial_results{threshold_idx}.audits{1}.selected_indices{:}));
    selected_display=selected(round(linspace(1,numel(selected), ...
        min(8000,numel(selected)))));
    for coordinate=1:2
        ax=nexttile((coordinate-1)*4+threshold_idx);hold(ax,'on');
        scatter(ax,X(display_idx,1),X(display_idx,coordinate+1),2,[.85 .85 .85],'filled');
        scatter(ax,X(selected_display,1),X(selected_display,coordinate+1),3,colors(threshold_idx,:),'filled');
        title(ax,sprintf('%s, x_%d',labels{threshold_idx},coordinate));
        xlabel(ax,'Generation time t');ylabel(ax,sprintf('State x_%d',coordinate));
        grid(ax,'on');box(ax,'on');
    end
end
sgtitle('2D uncertainty-based training-point selection (union over output GPs)');
print(f,fullfile(out,'Simple2D_Selected_Training_Points.emf'),'-dmeta','-painters');close(f);

% Five-model threshold animation. No raster still image is exported.
n_models=numel(sweep.runs);n_steps=size(sweep.runs{1}.path,1);
target=sweep.target_points;allxy=target;
for i=1:n_models,allxy=[allxy;reshape(sweep.runs{i}.path,[],2)];end %#ok<AGROW>
pad=.06*max(range(allxy,1));xl=[min(allxy(:,1))-pad,max(allxy(:,1))+pad];
yl=[min(allxy(:,2))-pad,max(allxy(:,2))+pad];
fig=figure('Visible','off','Color','w','Position',[30 30 1300 760]);
layout=tiledlayout(fig,2,3,'Padding','compact','TileSpacing','compact');
trail=cell(n_models,1);current=gobjects(n_models,1);
sample_colors=turbo(size(sweep.runs{1}.path,2));
for i=1:n_models
    ax=nexttile(layout);hold(ax,'on');p=sweep.runs{i}.path;
    scatter(ax,target(:,1),target(:,2),6,[.82 .82 .82],'filled');
    trail{i}=gobjects(size(p,2),1);
    for k=1:size(p,2)
        trail{i}(k)=plot(ax,p(1,k,1),p(1,k,2),'-','Color',sample_colors(k,:),'LineWidth',.45);
    end
    current(i)=scatter(ax,p(1,:,1),p(1,:,2),11,sample_colors,'filled');
    title(ax,labels{i});axis(ax,'equal');xlim(ax,xl);ylim(ax,yl);grid(ax,'on');box(ax,'on');
    xlabel(ax,'x_1');ylabel(ax,'x_2');
end
main_title=title(layout,'2D threshold comparison: t = 0.00');
mp4=fullfile(out,'Simple2D_Threshold_Animation.mp4');
gif=fullfile(out,'Simple2D_Threshold_Animation.gif');
writer=VideoWriter(mp4,'MPEG-4');writer.FrameRate=12;writer.Quality=95;open(writer);
frame_indices=unique([1:2:n_steps,n_steps]);
for frame_no=1:numel(frame_indices)
    step=frame_indices(frame_no);
    for i=1:n_models
        p=sweep.runs{i}.path;
        for k=1:size(p,2)
            set(trail{i}(k),'XData',p(1:step,k,1),'YData',p(1:step,k,2));
        end
        set(current(i),'XData',p(step,:,1),'YData',p(step,:,2));
    end
    main_title.String=sprintf('2D threshold comparison: t = %.2f',(step-1)/(n_steps-1));
    drawnow;frame=getframe(fig);writeVideo(writer,frame);
    [rgb,~]=frame2im(frame);[indexed,map]=rgb2ind(rgb,256);
    if frame_no==1
        imwrite(indexed,map,gif,'gif','LoopCount',inf,'DelayTime',.1);
    else
        imwrite(indexed,map,gif,'gif','WriteMode','append','DelayTime',.1);
    end
end
for i=1:12,writeVideo(writer,getframe(fig));end
close(writer);close(fig);
fprintf('Saved 2D selection EMF and threshold animation in %s\n',out);
end
