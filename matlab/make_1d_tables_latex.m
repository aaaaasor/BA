function make_1d_tables_latex()
%MAKE_1D_TABLES_LATEX  LaTeX for thesis tables 7 and 8 from the 1D sweep.
%
% Table 7 reports statistics of the posterior variance along the 1000 generated
% trajectories, so it carries an error bar: the standard error over those 1000
% trajectories (SD/sqrt(1000)).  SEM rather than SD, because the per-trajectory
% distribution is right-skewed and bounded below by zero -- mean +/- SD would
% print negative lower bounds for a variance.
%
% Table 8 evaluates the posterior variance on a FIXED shared grid, identical
% for every model.  Nothing is sampled there, so it has no error bar; adding
% one would be inventing a spread that does not exist.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
d = fullfile('outputs', ['1d case' char([20840 23616 71 80])]);
S = load(fullfile(d, 'Simple1D_GlobalGP_Threshold_Sweep.mat'), 'sweep');
s = S.sweep;
H = load(fullfile(d, 'Manual_Hyperparameters.mat'), 'SigmaF');
prior = H.SigmaF^2;
C = s.comparison;
n = height(C);

fprintf('prior variance sigma_F^2 = %.6f\n', prior);
fprintf('thresholds = %s\n\n', mat2str(s.thresholds));

% Per-trajectory summaries -> SEM over the 1000 rollouts.
mv = zeros(n,1); tv = zeros(n,1); smv = zeros(n,1); stv = zeros(n,1);
for i = 1:n
    V = s.runs{i}.variance;                 % (time x trajectory)
    per_mean = mean(V, 1);                  % one number per trajectory
    per_term = V(end, :);
    mv(i) = mean(per_mean);  smv(i) = std(per_mean)/sqrt(numel(per_mean));
    tv(i) = mean(per_term);  stv(i) = std(per_term)/sqrt(numel(per_term));
end
% Consistency with what the sweep itself recorded.
fprintf('%-14s %-14s %-14s %-14s %-14s\n', 'row', 'mean mine', 'mean sweep', ...
    'term mine', 'term sweep');
for i = 1:n
    fprintf('%-14s %-14.6g %-14.6g %-14.6g %-14.6g\n', C.Row{i}, mv(i), ...
        C.RawMeanVariance(i), tv(i), C.RawTerminalVariance(i));
end

thr = C.Threshold; ret = C.RetentionPercent; tsec = C.RolloutSecondsPerSample;
mvn = mv/prior; tvn = tv/prior; smvn = smv/prior; stvn = stv/prior;

best = [find(tsec == min(tsec)), find(mv == min(mv)), find(tv == min(tv)), ...
        find(mvn == min(mvn)), find(tvn == min(tvn))];

fprintf('\n%% ===================== table 7 =====================\n');
for i = 1:n
    if isnan(thr(i))
        head = sprintf('%s\n& --\n& %.2f\\%%', dq_all(), ret(i));
    else
        head = sprintf('%s\n& %.2f\n& %.2f\\%%', dq_part(), thr(i), ret(i));
    end
    fprintf('%s\n& %s\n& %s\n& %s\n& %s\n& %s \\\\\n\n', head, ...
        num_cell(tsec(i), 4, best(1) == i), ...
        pm_cell(mv(i),  smv(i),  4, best(2) == i), ...
        pm_cell(tv(i),  stv(i),  4, best(3) == i), ...
        pm_cell(mvn(i), smvn(i), 5, best(4) == i), ...
        pm_cell(tvn(i), stvn(i), 5, best(5) == i));
end

M = readtable(fullfile(d, 'Simple1D_GlobalGP_Table9_Map_Variance_Statistics.csv'));
bm = [find(M.MapMeanRawVariance == min(M.MapMeanRawVariance)), ...
      find(M.MapTerminalRawVariance == min(M.MapTerminalRawVariance)), ...
      find(M.MapMeanNormalizedVariance == min(M.MapMeanNormalizedVariance)), ...
      find(M.MapTerminalNormalizedVariance == min(M.MapTerminalNormalizedVariance))];
fprintf('\n%% ===================== table 8 =====================\n');
for i = 1:height(M)
    if isnan(M.Threshold(i))
        head = sprintf('%s\n& --\n& %.2f\\%%', dq_all(), M.RetentionPercent(i));
    else
        head = sprintf('%s\n& %.2f\n& %.2f\\%%', dq_part(), M.Threshold(i), ...
            M.RetentionPercent(i));
    end
    fprintf('%s\n& %s\n& %s\n& %s\n& %s\n& %s \\\\\n\n', head, ...
        num_cell(tsec(i), 4, best(1) == i), ...
        num_cell(M.MapMeanRawVariance(i),        5, bm(1) == i), ...
        num_cell(M.MapTerminalRawVariance(i),    5, bm(2) == i), ...
        num_cell(M.MapMeanNormalizedVariance(i), 5, bm(3) == i), ...
        num_cell(M.MapTerminalNormalizedVariance(i), 5, bm(4) == i));
end

fprintf('\n%% monotonicity check\n');
check('Time (should increase)', tsec, true);
check('KL', C.KL, false);
check('mean sigma2', mv, false);
check('term sigma2', tv, false);
check('map mean sigma2', M.MapMeanRawVariance, false);
check('map term sigma2', M.MapTerminalRawVariance, false);
end

% =====================================================================
function check(name, v, increasing)
d = diff(v);
if increasing, ok = all(d > 0); else, ok = all(d < 0); end
fprintf('%%   %-26s %s\n', name, tern(ok, 'monotone', 'NOT monotone'));
end

function s = num_cell(v, dig, bold)
s = sprintf('%.*f', dig, v);
if bold, s = sprintf('\\textbf{%s}', s); end
end

function s = pm_cell(v, e, dig, bold)
core = sprintf('%.*f \\pm %.*f', dig, v, dig, e);
if bold, s = sprintf('$\\mathbf{%s}$', core); else, s = sprintf('$%s$', core); end
end

function s = dq_part(), s = char([37096 20998 28857]); end   % partial points
function s = dq_all(),  s = char([20840 37096 28857]); end   % all points
function v = tern(c,a,b), if c, v=a; else, v=b; end, end
