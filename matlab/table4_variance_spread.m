function T = table4_variance_spread()
%TABLE4_VARIANCE_SPREAD  Point estimate + spread for the 2D threshold table.
%
% Reads the archived rollouts only; nothing is refitted or re-rolled.  The
% point estimates are recomputed the same way the archived CSV computed them
% (mean over time x sample x output) and printed next to the archived values
% as a check before the spread is reported.
%
% Spread is reported as the STANDARD ERROR of the mean over the 100 generated
% trajectories (SD/sqrt(100)).  The per-trajectory distribution is strongly
% right-skewed (skewness 3.8-6.0) with SD larger than the mean, so mean +/- SD
% would print negative lower bounds for a non-negative quantity.  SEM states
% the precision of the reported mean, which is what the table's point estimate
% actually is.  SD and the median/IQR are written to the CSV as well so a
% different convention can be adopted without rerunning anything.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
arch = fullfile('outputs', ['2d case loggp ' char([20840 28857 118 115 35757 32451 28857])]);
cache = fullfile(arch, 'experiment_cache');

runs = { ...
    fullfile(cache, 'Simple2D_Threshold_0p80', 'Partial_Rollout.mat'), 0.80, '部分点'; ...
    fullfile(cache, 'Simple2D_Threshold_0p60', 'Partial_Rollout.mat'), 0.60, '部分点'; ...
    fullfile(cache, 'Simple2D_Threshold_0p40', 'Partial_Rollout.mat'), 0.40, '部分点'; ...
    fullfile(cache, 'Simple2D_Threshold_0p20', 'Partial_Rollout.mat'), 0.20, '部分点'; ...
    fullfile(cache, ['2d case ' char([20840 37096 25968 25454])], 'AllData_Rollout.mat'), NaN, '全部点'};

ref = readtable(fullfile(arch, 'Table4_Latest.csv'));

prior = prior_variance(fullfile(cache, 'Simple2D_Threshold_0p20', 'Partial_Model.mat'));
fprintf('per-output prior variance sigma_F^2 = %s  (mean %.6f)\n', ...
    mat2str(prior, 6), mean(prior));

n = size(runs, 1);
[mr, tr, mn, tn] = deal(zeros(n,1));
[mr_sd, tr_sd, mn_sd, tn_sd] = deal(zeros(n,1));
[mr_sd_all, tr_sd_all] = deal(zeros(n,1));
[med_mn, med_tn] = deal(zeros(n,1));
iqr_mn = zeros(n,2); iqr_tn = zeros(n,2);
n_traj = 0;
for i = 1:n
    S = load(runs{i,1}, 'run');
    V = S.run.variance;                       % (time x sample x output)
    P = reshape(prior, 1, 1, []);
    N = V ./ P;

    mr(i) = mean(V(:));  tr(i) = mean(reshape(V(end,:,:), [], 1));
    mn(i) = mean(N(:));  tn(i) = mean(reshape(N(end,:,:), [], 1));

    % per-trajectory summaries -> SD across the 100 trajectories
    per_traj_mean_raw = squeeze(mean(mean(V, 1), 3));
    per_traj_term_raw = squeeze(mean(V(end,:,:), 3));
    per_traj_mean_nrm = squeeze(mean(mean(N, 1), 3));
    per_traj_term_nrm = squeeze(mean(N(end,:,:), 3));
    mr_sd(i) = std(per_traj_mean_raw);  tr_sd(i) = std(per_traj_term_raw);
    mn_sd(i) = std(per_traj_mean_nrm);  tn_sd(i) = std(per_traj_term_nrm);

    mr_sd_all(i) = std(V(:));
    tr_sd_all(i) = std(reshape(V(end,:,:), [], 1));
    med_mn(i) = median(per_traj_mean_nrm);
    iqr_mn(i,:) = prctile(per_traj_mean_nrm, [25 75]);
    med_tn(i) = median(per_traj_term_nrm);
    iqr_tn(i,:) = prctile(per_traj_term_nrm, [25 75]);
    n_traj = numel(per_traj_mean_nrm);
end

fprintf('\n--- reproduction check against Table4_Latest.csv ---\n');
fprintf('%-8s %-11s %-11s %-11s %-11s %-9s\n', 'model', 'mean s2', '(archived)', ...
    'term s2', '(archived)', 'max rel');
for i = 1:n
    r = max(abs(mr(i)-ref.RawMeanVariance(i))/ref.RawMeanVariance(i), ...
            abs(tr(i)-ref.RawTerminalVariance(i))/ref.RawTerminalVariance(i));
    lbl = ternary(isnan(runs{i,2}), 'all', sprintf('%.2f', runs{i,2}));
    fprintf('%-8s %-11.4f %-11.4f %-11.4f %-11.4f %-9.1e\n', lbl, ...
        mr(i), ref.RawMeanVariance(i), tr(i), ref.RawTerminalVariance(i), r);
end

fprintf('\n--- mean +/- SD across the 100 generated trajectories ---\n');
fprintf('%-8s %-20s %-20s %-20s %-20s\n', 'model', 'mean s2', 'term s2', ...
    'mean s~2', 'term s~2');
for i = 1:n
    lbl = ternary(isnan(runs{i,2}), 'all', sprintf('%.2f', runs{i,2}));
    fprintf('%-8s %8.4f +/- %-8.4f %8.4f +/- %-8.4f %8.4f +/- %-8.4f %8.4f +/- %-8.4f\n', ...
        lbl, mr(i), mr_sd(i), tr(i), tr_sd(i), mn(i), mn_sd(i), tn(i), tn_sd(i));
end

fprintf('\n--- for comparison: SD across ALL elements (time x sample x output) ---\n');
for i = 1:n
    lbl = ternary(isnan(runs{i,2}), 'all', sprintf('%.2f', runs{i,2}));
    fprintf('%-8s mean s2 SD %-10.4f   term s2 SD %-10.4f\n', lbl, ...
        mr_sd_all(i), tr_sd_all(i));
end

T = table(string(runs(:,3)), cell2mat(runs(:,2)), ref.RetentionPercent, ...
    ref.TimeSeconds, mr, mr_sd, mr_sd/sqrt(n_traj), tr, tr_sd, tr_sd/sqrt(n_traj), ...
    mn, mn_sd, mn_sd/sqrt(n_traj), tn, tn_sd, tn_sd/sqrt(n_traj), ...
    med_mn, iqr_mn(:,1), iqr_mn(:,2), med_tn, iqr_tn(:,1), iqr_tn(:,2), ...
    'VariableNames', {'Model','Threshold','RetentionPercent','TimeSeconds', ...
    'RawMeanVariance','RawMeanVarianceSD','RawMeanVarianceSEM', ...
    'RawTerminalVariance','RawTerminalVarianceSD','RawTerminalVarianceSEM', ...
    'NormMeanVariance','NormMeanVarianceSD','NormMeanVarianceSEM', ...
    'NormTerminalVariance','NormTerminalVarianceSD','NormTerminalVarianceSEM', ...
    'NormMeanVarianceMedian','NormMeanVarianceP25','NormMeanVarianceP75', ...
    'NormTerminalVarianceMedian','NormTerminalVarianceP25','NormTerminalVarianceP75'});
out = fullfile(arch, 'Table4_Latest_WithSpread.csv');
writetable(T, out);
fprintf('\nwrote %s\n', out);
end

function p = prior_variance(model_path)
M = load(model_path);
f = fieldnames(M); mc = [];
for i = 1:numel(f)
    v = M.(f{i});
    if isstruct(v) && isfield(v, 'model'), mc = v; break; end
end
assert(~isempty(mc), 'model collection not found in %s', model_path);
om = mc.model.output_models;
p = zeros(1, numel(om));
for j = 1:numel(om), p(j) = om{j}.SigmaF^2; end
end

function v = ternary(c, a, b), if c, v = a; else, v = b; end, end

function s = cell_pm(value, spread, digits, bold)
% "$0.0515 \pm 0.0088$", bolded as a whole when it is the best in its column.
core = sprintf('%.*f \pm %.*f', digits, value, digits, spread);
if bold
    s = sprintf('$\mathbf{%s}$', core);
else
    s = sprintf('$%s$', core);
end
end
