function sweep = run_simple_1d_global_gp_50x40_seed345_sn025()
% One-dimensional flow-matching experiment with ONE global exact GP.
% No LoG-GP tree, local experts, or GPoE aggregation are used.
% Static figures are exported as EMF only; no video is generated.

root = fileparts(mfilename('fullpath'));
out = fullfile(root, 'outputs', ['1d case' char([20840 23616 71 80])]);
if ~exist(out, 'dir'), mkdir(out); end
maxNumCompThreads(1);

% Clean, human-readable thresholds.  The full-data model is reported as
% threshold 0 in the comparison curve, but it is fitted in one batch.
thresholds = [0.14, 0.11, 0.08, 0.05];
cfg = struct('version', 1, 'n_train', 50, 'n_time_slices', 40, ...
    'n_rollouts', 1000, 'n_rollout_steps', 100, 'data_seed', 27, ...
    'append_seed', 345, ...
    'hyperparameter_seed', 28, 'rollout_seed', 31, 'test_seed', 32, ...
    'state_dim', 1, 'guidance_enabled', false, ...
    'seed_filter_enabled', false, 'global_exact_gp', true, ...
    'manual_hyperparameters_enabled', true, ...
    'manual_noise_std', 0.25, 'manual_state_length_scale', 0.45, ...
    'loggp_enabled', false, 'video_generated', false);

% Nested design: preserve every one of the original 25 independent pairs,
% then append 25 new pairs.  This isolates the effect of additional
% independent training trajectories from a wholesale dataset replacement.
base_out=fullfile(root,'outputs','1d case全局GP训练阈值实验_25x40');
B=load(fullfile(base_out,'Training_Data_and_Seeds.mat'), ...
    'source_points','target_points');
assert(numel(B.source_points)==25 && numel(B.target_points)==25);
rng(cfg.append_seed);
source_points=[B.source_points(:);randn(25,1)];
target_points=[B.target_points(:);sample_target_1d(25)];
training_times = linspace(0, 1, cfg.n_time_slices)';
velocities = target_points - source_points;
x_slices = zeros(cfg.n_time_slices, cfg.n_train, 1);
y_slices = x_slices;
for ti = 1:numel(training_times)
    x_slices(ti,:,1) = (1-training_times(ti))*source_points + ...
        training_times(ti)*target_points;
    y_slices(ti,:,1) = velocities;
end
X = [repmat(training_times, cfg.n_train, 1), reshape(x_slices, [], 1)];
Y = reshape(y_slices, [], 1);
rng(cfg.rollout_seed);
x_init = randn(cfg.n_rollouts, 1);
save(fullfile(out, 'Training_Data_and_Seeds.mat'), 'cfg', 'source_points', ...
    'target_points', 'training_times', 'X', 'Y', 'x_init', '-v7.3');

% Reuse exactly the same manually tuned hyperparameters as the 25x40 run.
H=load(fullfile(base_out,'Manual_Hyperparameters.mat'), ...
    'SigmaL','SigmaF','SigmaN','hp');
sigma_l=H.SigmaL; sigma_f=H.SigmaF; sigma_n=cfg.manual_noise_std; hp=H.hp;
assert(abs(sigma_l(2)-cfg.manual_state_length_scale)<1e-12);
hp.reused_from=fullfile(base_out,'Manual_Hyperparameters.mat');
hp.manual_noise_override=cfg.manual_noise_std;
hp.training_pair_comparison='nested 25 pairs plus 25 appended pairs';
SigmaL = sigma_l; SigmaF = sigma_f; SigmaN = sigma_n; %#ok<NASGU>
save(fullfile(out,'Manual_Hyperparameters.mat'),'SigmaL','SigmaF', ...
    'SigmaN','hp','cfg','-v7.3');
prior_variance = sigma_f^2;

labels = [arrayfun(@(z) sprintf('threshold_%.2f', z), thresholds, ...
    'UniformOutput', false), {'all_data'}];
n_models = numel(labels);
runs = cell(n_models,1);
models = cell(n_models,1);
selected_indices = cell(n_models,1);

for mode = 1:n_models
    stem = strrep(labels{mode}, '.', 'p');
    if mode <= numel(thresholds)
        threshold = thresholds(mode);
    else
        threshold = NaN;
    end
    signature = struct('cfg', cfg, 'threshold', threshold, ...
        'sigma_l', sigma_l, 'sigma_f', sigma_f, 'sigma_n', sigma_n, ...
        'X', X, 'Y', Y, 'implementation', 'single_global_exact_gp_v1');
    model_path = fullfile(out, [stem, '_GlobalGP_Model.mat']);
    use_cache = false;
    if isfile(model_path)
        z = load(model_path, 'signature');
        use_cache = isfield(z, 'signature') && isequaln(z.signature, signature);
    end
    if use_cache
        z = load(model_path, 'global_gp_model', 'fit_seconds', 'selected_idx');
        global_gp_model = z.global_gp_model;
        fit_seconds = z.fit_seconds;
        selected_idx = z.selected_idx;
        fprintf('Loaded matching global-GP model: %s\n', labels{mode});
    else
        fprintf('Fitting ONE global exact GP: %s\n', labels{mode});
        fit_tic = tic;
        if mode <= numel(thresholds)
            [global_gp_model, selected_idx] = fit_selected_global_gp( ...
                X, Y, threshold, sigma_n, sigma_f, sigma_l);
        else
            [global_gp_model, selected_idx] = fit_all_global_gp( ...
                X, Y, sigma_n, sigma_f, sigma_l);
        end
        fit_seconds = toc(fit_tic);
        save(model_path, 'global_gp_model', 'signature', 'fit_seconds', ...
            'selected_idx', '-v7.3');
    end
    assert(numel(unique(selected_idx)) == numel(selected_idx));
    assert(all(selected_idx >= 1 & selected_idx <= size(X,1)));
    if mode == n_models, assert(numel(selected_idx) == size(X,1)); end
    models{mode} = global_gp_model;
    selected_indices{mode} = selected_idx;

    rollout_path = fullfile(out, [stem, '_GlobalGP_Rollout.mat']);
    use_cache = false;
    if isfile(rollout_path)
        z = load(rollout_path, 'signature', 'x_init');
        use_cache = isfield(z, 'signature') && isequaln(z.signature, signature) ...
            && isequaln(z.x_init, x_init);
    end
    if use_cache
        z = load(rollout_path, 'run');
        run = z.run;
    else
        rollout_tic = tic;
        [times, path] = plain_rollout_1d(global_gp_model, x_init, ...
            linspace(0,1,cfg.n_rollout_steps+1)');
        rollout_seconds = toc(rollout_tic);
        variance = evaluate_global_gp_variance(global_gp_model, times, path);
        run = struct('times', times, 'path', path, 'variance', variance, ...
            'fit_seconds', fit_seconds, 'rollout_seconds', rollout_seconds, ...
            'source_model', model_path, 'guidance_enabled', false, ...
            'seed_filter_enabled', false, 'loggp_enabled', false, ...
            'global_exact_gp', true);
        save(rollout_path, 'run', 'signature', 'x_init', '-v7.3');
    end
    runs{mode} = run;
end

% Common held-out FM queries: every model is evaluated at identical points.
rng(cfg.test_seed);
n_test = 200;
test_source = randn(n_test,1);
test_target = sample_target_1d(n_test);
test_times = linspace(0,1,20)';
test_X = zeros(n_test*numel(test_times),2);
test_Y = zeros(size(test_X,1),1);
row = 1;
for sample = 1:n_test
    velocity = test_target(sample)-test_source(sample);
    for ti = 1:numel(test_times)
        t = test_times(ti);
        test_X(row,:) = [t, (1-t)*test_source(sample)+t*test_target(sample)];
        test_Y(row) = velocity;
        row = row+1;
    end
end

summary = nan(n_models,13);
for mode = 1:n_models
    model = models{mode};
    run = runs{mode};
    v = run.variance;
    nv = v/prior_variance;
    predictions = predict_global_gp_mean_batch(model, test_X');
    test_variance = predict_global_gp_variance_batch(model, test_X')/prior_variance;
    info = dir(run.source_model);
    threshold = NaN;
    if mode <= numel(thresholds), threshold = thresholds(mode); end
    generated = squeeze(run.path(end,:,1))';
    summary(mode,:) = [threshold, numel(selected_indices{mode}), ...
        100*numel(selected_indices{mode})/size(X,1), info.bytes/2^20, ...
        run.fit_seconds, run.rollout_seconds/cfg.n_rollouts, ...
        fixed_kde_kl_1d(target_points, generated), ...
        sqrt(mean((predictions-test_Y).^2)), mean(test_variance), ...
        mean(v,'all'), mean(v(end,:,:),'all'), ...
        mean(nv,'all'), mean(nv(end,:,:),'all')];
end
comparison = array2table(summary, 'RowNames', labels, 'VariableNames', ...
    {'Threshold','TrainingPoints','RetentionPercent','ModelSizeMB', ...
    'FitTimeSeconds','RolloutSecondsPerSample','KL','HeldoutVelocityRMSE', ...
    'FixedQueryNormalizedVariance','RawMeanVariance','RawTerminalVariance', ...
    'NormalizedMeanVariance','NormalizedTerminalVariance'});

sweep = struct('cfg', cfg, 'hyperparameters', hp, 'thresholds', thresholds, ...
    'runs', {runs}, 'selected_indices', {selected_indices}, ...
    'comparison', comparison, 'X', X, 'Y', Y, ...
    'source_points', source_points, 'target_points', target_points, ...
    'x_init', x_init, 'metadata', struct( ...
    'model_type', 'single global exact GP', ...
    'loggp_used', false, 'local_experts_used', false, ...
    'aggregation_used', false, ...
    'selection_rule', 'add point iff global-GP posterior std exceeds threshold', ...
    'all_data_rule', 'one batch exact-GP fit with every FM pair', ...
    'same_hyperparameters_and_rollout_seed', true, ...
    'nested_training_pairs', true, 'video_generated', false));
save(fullfile(out, 'Simple1D_GlobalGP_Threshold_Sweep.mat'), 'sweep', '-v7.3');
writetable(comparison, fullfile(out, 'Simple1D_GlobalGP_Threshold_Sweep.csv'), ...
    'WriteRowNames', true);
disp(comparison);
draw_1d_sweep(sweep, out, prior_variance);
fprintf('1D global exact-GP threshold sweep complete: %s\n', out);
end

function [model, selected_idx] = fit_selected_global_gp(X,Y,threshold,sn,sf,sl)
n = size(X,1);
gp = LocalGP_MultiOutput(size(X,2),1,n,sn,sf,sl);
selected_idx = zeros(n,1);
count = 0;
for i = 1:n
    if gp.DataQuantity == 0 || sqrt(gp.predict_variance(X(i,:)')) > threshold
        flag = gp.addPoint(X(i,:)',Y(i));
        assert(flag == 1);
        count = count+1;
        selected_idx(count) = i;
    end
end
selected_idx = selected_idx(1:count);
model = compact_global_gp(gp);
clear gp;
end

function [model, selected_idx] = fit_all_global_gp(X,Y,sn,sf,sl)
n = size(X,1);
gp = LocalGP_MultiOutput(size(X,2),1,n,sn,sf,sl);
gp.add_Alldata(X,Y);
selected_idx = (1:n)';
model = compact_global_gp(gp);
clear gp;
end

function model = compact_global_gp(gp)
n = gp.DataQuantity;
model = struct('X',gp.X(:,1:n),'Y',gp.Y(1:n,:), ...
    'L',gp.L(1:n,1:n),'alpha',gp.alpha(1:n,:), ...
    'SigmaN',gp.SigmaN,'SigmaF',gp.SigmaF,'SigmaL',gp.SigmaL, ...
    'DataQuantity',n,'model_type','single global exact GP');
end

function [times,path] = plain_rollout_1d(model,initial,times)
path = zeros(numel(times),size(initial,1),1);
path(1,:,1) = initial;
for qi = 1:size(initial,1)
    x = initial(qi);
    for ti = 1:numel(times)-1
        t = times(ti); h = times(ti+1)-t;
        k1 = predict_global_gp_mean(model,[t;x]);
        k2 = predict_global_gp_mean(model,[t+h/2;x+h*k1/2]);
        k3 = predict_global_gp_mean(model,[t+h/2;x+h*k2/2]);
        k4 = predict_global_gp_mean(model,[t+h;x+h*k3]);
        x = x+h*(k1+2*k2+2*k3+k4)/6;
        path(ti+1,qi,1) = x;
    end
end
end

function mu = predict_global_gp_mean(model,q)
k = global_gp_kernel(model,model.X,q);
mu = model.alpha'*k;
end

function mu = predict_global_gp_mean_batch(model,Q)
k = global_gp_kernel(model,model.X,Q);
mu = (model.alpha'*k)';
end

function variance = evaluate_global_gp_variance(model,times,path)
n_t = numel(times); n_q = size(path,2);
Q = [repelem(times,n_q)'; reshape(path(:,:,1)',1,[])];
v = predict_global_gp_variance_batch(model,Q);
variance = reshape(v,n_q,n_t)';
variance = reshape(variance,n_t,n_q,1);
end

function variance = predict_global_gp_variance_batch(model,Q)
nq = size(Q,2); variance = zeros(nq,1); chunk = 500;
for first = 1:chunk:nq
    last = min(first+chunk-1,nq);
    k = global_gp_kernel(model,model.X,Q(:,first:last));
    z = model.L\k;
    variance(first:last) = max(model.SigmaF^2-sum(z.^2,1)',0);
end
end

function k = global_gp_kernel(model,X,Q)
dx = reshape(X,2,[],1)-reshape(Q,2,1,[]);
scaled = dx./reshape(model.SigmaL(:),2,1,1);
k = model.SigmaF^2*exp(-0.5*squeeze(sum(scaled.^2,1)));
if size(Q,2)==1, k=reshape(k,[],1); end
end

function target = sample_target_1d(n)
component = rand(n,1)>0.5;
target = (-2.0+0.45*randn(n,1)).*(~component) + ...
    (2.0+0.55*randn(n,1)).*component;
end

function value = fixed_kde_kl_1d(reference,generated)
n = numel(reference); bw = max(1.06*std(reference)*n^(-1/5),1e-3);
grid = linspace(min(reference)-6*bw,max(reference)+6*bw,1024)';
p = kde_1d(grid,reference,bw); q = kde_1d(grid,generated,bw);
p = p/sum(p); q = q/sum(q); floor_value = 1e-300;
value = sum(p.*log(max(p,floor_value)./max(q,floor_value)));
end

function density = kde_1d(grid,samples,bw)
density = zeros(size(grid));
for i=1:numel(samples)
    density = density+exp(-.5*((grid-samples(i))/bw).^2);
end
density = density/max(numel(samples),1);
end

function draw_1d_sweep(sweep,out,prior)
n = numel(sweep.runs); colors = lines(n);
labels = [arrayfun(@(z)sprintf('threshold %.2f',z),sweep.thresholds, ...
    'UniformOutput',false),{'all data'}];

f=figure('Visible','off','Color','w','Position',[30 30 1200 760],'Renderer','painters');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
for i=1:numel(sweep.thresholds)
    ax=nexttile;hold(ax,'on');
    scatter(ax,sweep.X(:,1),sweep.X(:,2),5,[.82 .82 .82],'filled');
    idx=sweep.selected_indices{i};
    scatter(ax,sweep.X(idx,1),sweep.X(idx,2),8,colors(i,:),'filled');
    title(ax,sprintf('%s: %.1f%% retained',labels{i},sweep.comparison.RetentionPercent(i)));
    xlabel(ax,'Generation time t');ylabel(ax,'State x');grid(ax,'on');box(ax,'on');
end
sgtitle('1D global-GP uncertainty-based training-point selection');
print(f,fullfile(out,'Simple1D_GlobalGP_Selected_Training_Points.emf'),'-dmeta','-painters');close(f);

density_grid=linspace(min(sweep.target_points)-2,max(sweep.target_points)+2,600)';
bw=max(1.06*std(sweep.target_points)*numel(sweep.target_points)^(-1/5),1e-3);
target_density=kde_1d(density_grid,sweep.target_points,bw);
target_density=target_density/trapz(density_grid,target_density);
f=figure('Visible','off','Color','w','Position',[30 30 1200 760],'Renderer','painters');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
for i=1:n
    ax=nexttile;hold(ax,'on');generated=squeeze(sweep.runs{i}.path(end,:,1))';
    q=kde_1d(density_grid,generated,bw);q=q/trapz(density_grid,q);
    plot(ax,density_grid,target_density,'Color',[.25 .25 .25],'LineWidth',1.8);
    plot(ax,density_grid,q,'Color',colors(i,:),'LineWidth',1.8);
    title(ax,labels{i});xlabel(ax,'x');ylabel(ax,'Density');grid(ax,'on');box(ax,'on');
end
sgtitle('1D target and generated endpoint distributions: one global GP');
print(f,fullfile(out,'Simple1D_GlobalGP_Endpoint_Distributions.emf'),'-dmeta','-painters');close(f);

f=figure('Visible','off','Color','w','Position',[30 30 1200 760],'Renderer','painters');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
for i=1:n
    ax=nexttile;hold(ax,'on');p=squeeze(sweep.runs{i}.path(:,:,1));
    plot(ax,sweep.runs{i}.times,p,'Color',.58+.42*colors(i,:),'LineWidth',.5);
    title(ax,labels{i});xlabel(ax,'Generation time t');ylabel(ax,'State x');grid(ax,'on');box(ax,'on');
end
sgtitle('1D global-GP trajectories from identical initial samples');
print(f,fullfile(out,'Simple1D_GlobalGP_Rollout_Trajectories.emf'),'-dmeta','-painters');close(f);

f=figure('Visible','off','Color','w','Position',[80 80 820 500],'Renderer','painters');
ax=axes(f);hold(ax,'on');
for i=1:n
    curve=mean(squeeze(sweep.runs{i}.variance(:,:,1))/prior,2);
    plot(ax,sweep.runs{i}.times,curve,'LineWidth',1.8,'Color',colors(i,:));
end
xlabel(ax,'Generation time t');ylabel(ax,'Mean variance / prior variance');
grid(ax,'on');box(ax,'on');legend(ax,labels,'Location','best','Box','off');
title(ax,'1D global-GP normalized posterior variance');
print(f,fullfile(out,'Simple1D_GlobalGP_Variance_Curves.emf'),'-dmeta','-painters');close(f);

% Table-5 style curve: display all data at threshold zero.
C=sweep.comparison; x=[0;sweep.thresholds(:)]; order=[n;(1:numel(sweep.thresholds))'];
[x,sort_idx]=sort(x); order=order(sort_idx);
f=figure('Visible','off','Color','w','Position',[20 20 1600 1040],'Renderer','painters');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
ax=nexttile;plot(ax,x,C.RetentionPercent(order),'-o','LineWidth',1.8,'MarkerSize',7);
xlabel(ax,'Training uncertainty threshold');ylabel(ax,'Training-point retention (%)');
title(ax,'Data retention (threshold 0 = all data)');grid(ax,'on');box(ax,'on');
ax=nexttile;plot(ax,x,C.RolloutSecondsPerSample(order),'-o','LineWidth',1.8,'MarkerSize',7);
xlabel(ax,'Training uncertainty threshold');ylabel(ax,'Time (s/trajectory)');
title(ax,'Rollout time (threshold 0 = all data)');grid(ax,'on');box(ax,'on');
ax=nexttile;hold(ax,'on');
plot(ax,x,C.RawMeanVariance(order),'-o','LineWidth',1.8,'MarkerSize',7);
plot(ax,x,C.RawTerminalVariance(order),'-s','LineWidth',1.8,'MarkerSize',7);
xlabel(ax,'Training uncertainty threshold');ylabel(ax,'Predictive variance');
title(ax,'Raw posterior variance');legend(ax,{'Mean','Terminal'},'Location','northwest','Box','off');grid(ax,'on');box(ax,'on');
ax=nexttile;hold(ax,'on');
plot(ax,x,C.NormalizedMeanVariance(order),'-o','LineWidth',1.8,'MarkerSize',7);
plot(ax,x,C.NormalizedTerminalVariance(order),'-s','LineWidth',1.8,'MarkerSize',7);
xlabel(ax,'Training uncertainty threshold');ylabel(ax,'Variance / prior variance');
title(ax,'Normalized posterior variance');legend(ax,{'Mean','Terminal'},'Location','northwest','Box','off');grid(ax,'on');box(ax,'on');
sgtitle('1D global exact-GP threshold comparison');
print(f,fullfile(out,'Simple1D_GlobalGP_Table5_Threshold_Curves.emf'),'-dmeta','-painters');close(f);
end



