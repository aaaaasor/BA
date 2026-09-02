function sweep = run_simple_2d_threshold_sweep()
% Threshold ablation for the fixed original 2D LoG-GP point-flow case.
root = 'C:\Users\JieJi\BA\matlab';
addpath(root);
thresholds = [0.20, 0.40, 0.60, 0.80];
subdirs = {'Simple2D_Threshold_0p20', 'Simple2D_Threshold_0p40', ...
    'Simple2D_Threshold_0p60', 'Simple2D_Threshold_0p80'};
summary_dir = fullfile(root, 'outputs', '2d case训练阈值实验');
if ~exist(summary_dir, 'dir'), mkdir(summary_dir); end

partial_results = cell(numel(thresholds), 1);
for i = 1:numel(thresholds)
    fprintf('\n===== 2D threshold %.7g (%d/%d) =====\n', ...
        thresholds(i), i, numel(thresholds));
    partial_results{i} = run_simple_2d_loggp_comparison( ...
        thresholds(i), subdirs{i});
end

target_file = fullfile(root, 'outputs', '2d case部分数据', ...
    'Training_Data_and_Seeds.mat');
T = load(target_file, 'target_points');
target = T.target_points;
all_run = partial_results{1}.runs{2};
n_rows = numel(thresholds) + 1;
row_names = cell(n_rows, 1);
summary = nan(n_rows, 11);
runs = cell(n_rows, 1);

for i = 1:numel(thresholds)
    r = partial_results{i};
    c = r.comparison('Partial', :);
    runs{i} = r.runs{1};
    row_names{i} = sprintf('threshold_%.7g', thresholds(i));
    model_file = fullfile(root, 'outputs', subdirs{i}, 'Partial_Model.mat');
    info = dir(model_file);
    summary(i,:) = [thresholds(i), c.MeanTrainingPointsPerOutput, ...
        c.RetentionPercent, info.bytes / 2^20, runs{i}.fit_seconds, ...
        runs{i}.rollout_seconds / r.cfg.n_rollouts, ...
        fixed_kde_kl_2d(target, squeeze(runs{i}.path(end,:,:))), ...
        c.RawMeanVariance, c.RawTerminalVariance, ...
        c.NormalizedMeanVariance, c.NormalizedTerminalVariance];
end

r = partial_results{1};
c = r.comparison('AllData', :);
runs{end} = all_run;
row_names{end} = 'all_data';
all_model_file = fullfile(root, 'outputs', '2d case 全部数据', ...
    'AllData_Model.mat');
info = dir(all_model_file);
summary(end,:) = [NaN, c.MeanTrainingPointsPerOutput, c.RetentionPercent, ...
    info.bytes / 2^20, all_run.fit_seconds, ...
    all_run.rollout_seconds / r.cfg.n_rollouts, ...
    fixed_kde_kl_2d(target, squeeze(all_run.path(end,:,:))), ...
    c.RawMeanVariance, c.RawTerminalVariance, ...
    c.NormalizedMeanVariance, c.NormalizedTerminalVariance];

comparison = array2table(summary, 'RowNames', row_names, 'VariableNames', ...
    {'Threshold','TrainingPointsPerOutput','RetentionPercent','ModelSizeMB', ...
    'FitTimeSeconds','RolloutSecondsPerSample','KL','RawMeanVariance', ...
    'RawTerminalVariance','NormalizedMeanVariance', ...
    'NormalizedTerminalVariance'});
sweep = struct('thresholds', thresholds, 'runs', {runs}, ...
    'comparison', comparison, 'target_points', target, ...
    'partial_results', {partial_results}, ...
    'metadata', struct('selection_rule', ...
    'add point iff predictive standard deviation exceeds threshold/sqrt(output dimension)', ...
    'all_data_rule', 'training-point selection disabled', ...
    'same_data_hyperparameters_and_seeds', true));
save(fullfile(summary_dir, 'Simple2D_Threshold_Sweep.mat'), 'sweep', '-v7.3');
writetable(comparison, fullfile(summary_dir, 'Simple2D_Threshold_Sweep.csv'), ...
    'WriteRowNames', true);
disp(comparison);
draw_2d_sweep(sweep, summary_dir);
fprintf('2D threshold sweep complete: %s\n', summary_dir);
end

function value = fixed_kde_kl_2d(reference, generated)
n = size(reference, 1);
bw = std(reference, 0, 1) .* n^(-1/6);
bw = max(bw, 1e-3);
lo = min(reference, [], 1) - 6*bw;
hi = max(reference, [], 1) + 6*bw;
xg = linspace(lo(1), hi(1), 96);
yg = linspace(lo(2), hi(2), 96);
[Xg,Yg] = meshgrid(xg,yg);
grid = [Xg(:),Yg(:)];
p = kde(grid, reference, bw);
q = kde(grid, generated, bw);
p = p / sum(p); q = q / sum(q);
floor_value = 1e-300;
value = sum(p .* log(max(p,floor_value) ./ max(q,floor_value)));
end

function density = kde(grid, samples, bandwidth)
density = zeros(size(grid,1),1);
for i = 1:size(samples,1)
    z = (grid-samples(i,:))./bandwidth;
    density = density + exp(-0.5*sum(z.^2,2));
end
density = density / max(size(samples,1),1);
end

function draw_2d_sweep(sweep, out)
colors = lines(numel(sweep.runs));
labels = [arrayfun(@(x)sprintf('threshold %.4g',x), ...
    sweep.thresholds,'UniformOutput',false), {'all data'}];

% Endpoint distributions.
f = figure('Visible','off','Color','w','Position',[40 40 1300 760], ...
    'Renderer','painters');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
allxy = sweep.target_points;
for i=1:numel(sweep.runs)
    allxy=[allxy;squeeze(sweep.runs{i}.path(end,:,:))]; %#ok<AGROW>
end
bounds=[min(allxy,[],1);max(allxy,[],1)]; pad=.06*range(bounds,1);
for i=1:numel(sweep.runs)
    ax=nexttile;hold(ax,'on');
    scatter(ax,sweep.target_points(:,1),sweep.target_points(:,2),10,[.8 .8 .8],'filled');
    xy=squeeze(sweep.runs{i}.path(end,:,:));
    scatter(ax,xy(:,1),xy(:,2),18,colors(i,:),'filled');
    title(ax,labels{i});axis(ax,'equal');grid(ax,'on');box(ax,'on');
    xlim(ax,[bounds(1,1)-pad(1),bounds(2,1)+pad(1)]);
    ylim(ax,[bounds(1,2)-pad(2),bounds(2,2)+pad(2)]);
    xlabel(ax,'x_1');ylabel(ax,'x_2');
end
sgtitle('2D LoG-GP threshold comparison: target and generated endpoints');
print(f,fullfile(out,'Simple2D_Threshold_Endpoint_Distributions.emf'),'-dmeta','-painters');close(f);

% Full generative trajectories.
f = figure('Visible','off','Color','w','Position',[40 40 1300 760], ...
    'Renderer','painters');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
for i=1:numel(sweep.runs)
    ax=nexttile;hold(ax,'on');p=sweep.runs{i}.path;
    scatter(ax,sweep.target_points(:,1),sweep.target_points(:,2),7,[.85 .85 .85],'filled');
    for k=1:size(p,2),plot(ax,p(:,k,1),p(:,k,2),'Color',colors(i,:),'LineWidth',.45);end
    scatter(ax,p(1,:,1),p(1,:,2),10,'o','MarkerEdgeColor',[.2 .2 .2]);
    scatter(ax,p(end,:,1),p(end,:,2),12,colors(i,:),'filled');
    title(ax,labels{i});axis(ax,'equal');grid(ax,'on');box(ax,'on');
    xlim(ax,[bounds(1,1)-pad(1),bounds(2,1)+pad(1)]);
    ylim(ax,[bounds(1,2)-pad(2),bounds(2,2)+pad(2)]);
    xlabel(ax,'x_1');ylabel(ax,'x_2');
end
sgtitle('2D LoG-GP rollout trajectories from identical initial samples');
print(f,fullfile(out,'Simple2D_Threshold_Rollout_Trajectories.emf'),'-dmeta','-painters');close(f);

% Normalized variance curves.
f=figure('Visible','off','Color','w','Position',[80 80 820 500],'Renderer','painters');
ax=axes(f);hold(ax,'on');
for i=1:numel(sweep.runs)
    v=sweep.runs{i}.variance;
    if i<=numel(sweep.thresholds),prior=reshape(sweep.partial_results{i}.gp.signal_std_vec.^2,1,1,2);
    else,prior=reshape(sweep.partial_results{1}.gp.signal_std_vec.^2,1,1,2);end
    curve=mean(mean(v./prior,3),2);
    plot(ax,sweep.runs{i}.times,curve,'LineWidth',1.8,'Color',colors(i,:));
end
xlabel(ax,'Generation time t');ylabel(ax,'Mean variance / prior variance');
grid(ax,'on');box(ax,'on');legend(ax,labels,'Location','best','Box','off');
title(ax,'2D normalized posterior variance under different thresholds');
print(f,fullfile(out,'Simple2D_Threshold_Variance_Curves.emf'),'-dmeta','-painters');close(f);

% Threshold tradeoff.
C=sweep.comparison;idx=1:numel(sweep.thresholds);
f=figure('Visible','off','Color','w','Position',[80 80 1000 680],'Renderer','painters');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
fields={'RetentionPercent','ModelSizeMB','KL','NormalizedMeanVariance'};
ylabs={'Retained training points (%)','Model cache size (MB)','KL divergence','Normalized mean variance'};
for j=1:4
    ax=nexttile;plot(ax,sweep.thresholds,C{idx,fields{j}},'-o','LineWidth',1.8);
    xlabel(ax,'Training uncertainty threshold');ylabel(ax,ylabs{j});grid(ax,'on');box(ax,'on');
end
sgtitle('2D threshold--storage--quality tradeoff');
print(f,fullfile(out,'Simple2D_Threshold_Tradeoff.emf'),'-dmeta','-painters');close(f);
end
