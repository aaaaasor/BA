function plot_1d_threshold_curves_with_errorbars(out_emf, preview_png)
%PLOT_1D_THRESHOLD_CURVES_WITH_ERRORBARS  Table-7 curves, with error bars.
%
% Same four-panel layout as plot_threshold_table_curves produced, but the two
% variance panels now carry error bars, matching the 2D figure.
%
% The bar is the standard error over the 1000 generated trajectories
% (SD/sqrt(1000)), not the standard deviation.  The per-trajectory variance
% distribution is right-skewed and bounded below by zero, so mean +/- SD would
% extend below zero for a non-negative quantity; SEM states the precision of
% the plotted mean, which is what the point estimate is.
%
% Retention and time are single measurements and get no bars.
%
% Bands are deliberately not used: the x axis is five discrete thresholds, not
% a continuous variable, so a shaded region would imply values between them.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
d = fullfile('outputs', ['1d case' char([20840 23616 71 80])]);
if nargin < 1 || isempty(out_emf)
    out_emf = fullfile(d, 'Simple1D_GlobalGP_Table5_Threshold_Curves.emf');
end
if nargin < 2, preview_png = ''; end

S = load(fullfile(d, 'Simple1D_GlobalGP_Threshold_Sweep.mat'), 'sweep');
s = S.sweep;
H = load(fullfile(d, 'Manual_Hyperparameters.mat'), 'SigmaF');
prior = H.SigmaF^2;
C = s.comparison;
n = height(C);

% Per-trajectory summaries -> SEM over the 1000 rollouts.
mv = zeros(n,1); tv = zeros(n,1); smv = zeros(n,1); stv = zeros(n,1);
for i = 1:n
    V = s.runs{i}.variance;
    pm = mean(V, 1); pt = V(end, :);
    mv(i) = mean(pm); smv(i) = std(pm)/sqrt(numel(pm));
    tv(i) = mean(pt); stv(i) = std(pt)/sqrt(numel(pt));
end

% Row order: all-data first (plotted at x = 0), then thresholds ascending.
thr = C.Threshold;
full_row = find(isnan(thr));
[xs, ord] = sort(thr(~isnan(thr)));
part_rows = find(~isnan(thr));
part_rows = part_rows(ord);
rows = [full_row; part_rows];
x = [0; xs];

ret  = C.RetentionPercent(rows);
tsec = C.RolloutSecondsPerSample(rows);
mvv  = mv(rows);  tvv  = tv(rows);
smvv = smv(rows); stvv = stv(rows);

blue  = [0.10 0.36 0.68];
red   = [0.82 0.25 0.15];
black = [0.15 0.15 0.15];

f = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 1050 700], ...
    'Renderer', 'painters');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

one_curve(nexttile, x, ret, blue, black, ...
    'Training-point retention (%)', 'Data retention (threshold 0 = all data)');

one_curve(nexttile, x, tsec, blue, black, ...
    'Time (s/trajectory)', 'Rollout time (threshold 0 = all data)');

two_curves(nexttile, x, mvv, smvv, tvv, stvv, blue, red, black, ...
    'Predictive variance', 'Raw posterior variance');

two_curves(nexttile, x, mvv/prior, smvv/prior, tvv/prior, stvv/prior, ...
    blue, red, black, 'Variance / prior variance', 'Normalized posterior variance');

sgtitle('1D global exact-GP threshold comparison');
print(f, out_emf, '-dmeta', '-painters');
fprintf('Saved %s\n', out_emf);
if ~isempty(preview_png)
    exportgraphics(f, preview_png, 'Resolution', 150);
    fprintf('Saved preview %s\n', preview_png);
end
close(f);

fprintf('\n%-9s %-11s %-11s %-22s %-22s\n', 'threshold', 'retention%', ...
    'Time(s)', 'mean s2 +/- SEM', 'term s2 +/- SEM');
for i = 1:n
    if x(i) == 0, lbl = 'all'; else, lbl = sprintf('%.2f', x(i)); end
    fprintf('%-9s %-11.2f %-11.4f %9.4f +/- %-9.4f %9.4f +/- %-9.4f\n', ...
        lbl, ret(i), tsec(i), mvv(i), smvv(i), tvv(i), stvv(i));
end
end

% =====================================================================
function one_curve(ax, x, y, c, black, ylab, ttl)
hold(ax, 'on');
plot(ax, x, y, '-o', 'Color', c, 'LineWidth', 1.8, ...
    'MarkerFaceColor', c, 'MarkerSize', 6);
scatter(ax, 0, y(1), 52, black, 'd', 'filled');
xlabel(ax, 'Training uncertainty threshold'); ylabel(ax, ylab); title(ax, ttl);
format_axes(ax, x);
end

function two_curves(ax, x, y1, e1, y2, e2, c1, c2, black, ylab, ttl)
hold(ax, 'on');
h1 = errorbar(ax, x, y1, e1, '-o', 'Color', c1, 'LineWidth', 1.8, ...
    'MarkerFaceColor', c1, 'MarkerSize', 6, 'CapSize', 8);
h2 = errorbar(ax, x, y2, e2, '-s', 'Color', c2, 'LineWidth', 1.8, ...
    'MarkerFaceColor', c2, 'MarkerSize', 6, 'CapSize', 8);
scatter(ax, 0, y1(1), 52, black, 'd', 'filled');
scatter(ax, 0, y2(1), 52, black, 'd', 'filled');
xlabel(ax, 'Training uncertainty threshold'); ylabel(ax, ylab); title(ax, ttl);
legend(ax, [h1 h2], {'Mean \pm SEM', 'Terminal \pm SEM'}, ...
    'Location', 'best', 'Box', 'off');
format_axes(ax, x);
ylim(ax, [0, max([y1 + e1; y2 + e2]) * 1.08]);
end

function format_axes(ax, x)
grid(ax, 'on'); box(ax, 'on'); xticks(ax, x);
xlim(ax, [min(x) - .04*range(x), max(x) + .04*range(x)]);
set(ax, 'FontName', 'Times New Roman', 'FontSize', 11);
end
