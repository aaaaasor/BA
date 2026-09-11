function plot_table4_threshold_curves(csv_path, out_emf, figure_title, preview_png)
%PLOT_TABLE4_THRESHOLD_CURVES  Four-panel threshold sweep, same layout as the
% 1D figure produced by plot_threshold_table_curves.m.
%
% Reads Table4_Latest_WithSpread.csv, whose rows are the thresholded models
% plus one all-data row (NaN threshold).  The all-data row is drawn at x = 0
% and additionally marked with a black diamond, exactly as in the 1D figure.
%
% The two variance panels carry error bars: mean +/- SEM over the 100 generated
% trajectories (SD/sqrt(100)).  SEM, not SD -- the per-trajectory distribution
% is strongly right-skewed (skewness 3.8-6.0) with SD larger than the mean, so
% SD bars would extend below zero for a non-negative quantity.  The bar states
% the precision of the plotted mean, which is what the point estimate is.
%
% Retention and time are single measurements, so those panels have no bars.
%
% Use this instead of plot_threshold_table_curves for the 2D case: that
% function reads Simple2D_Threshold_Sweep.mat, whose Time column is the
% 2026-09-01 single-shot timing.  Table4_Latest carries the 2026-09-03
% warmed-up re-timing that the thesis table reports.

root = 'C:\Users\JieJi\BA\matlab';
d = fullfile(root, 'outputs', ['2d case loggp ' char([20840 28857 118 115 35757 32451 28857])]);
if nargin < 1 || isempty(csv_path),  csv_path = fullfile(d, 'Table4_Latest_WithSpread.csv'); end
if nargin < 2 || isempty(out_emf),   out_emf  = fullfile(d, 'Simple2D_Table4_Threshold_Curves_Latest.emf'); end
if nargin < 3 || isempty(figure_title), figure_title = '2D LoG-GP threshold comparison'; end
if nargin < 4, preview_png = ''; end

T = readtable(csv_path);
has_sem = ismember('NormTerminalVarianceSEM', T.Properties.VariableNames);
if ~has_sem
    warning('%s has no SEM columns; drawing without error bars.', csv_path);
end
is_all = isnan(T.Threshold);
assert(nnz(is_all) == 1, 'expected exactly one all-data row');
part = T(~is_all, :);
full = T(is_all, :);
[thr, order] = sort(part.Threshold);
part = part(order, :);

x = [0; thr];
grab = @(name) [full.(name); part.(name)];
err  = @(name) ternary(has_sem, @() grab(name), @() zeros(size(x)));

blue  = [0.10 0.36 0.68];
red   = [0.82 0.25 0.15];
black = [0.15 0.15 0.15];

f = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 1050 700], ...
    'Renderer', 'painters');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

one_curve(nexttile, x, grab('RetentionPercent'), blue, black, ...
    'Training-point retention (%)', 'Data retention (threshold 0 = all data)');

one_curve(nexttile, x, grab('TimeSeconds'), blue, black, ...
    'Time (s/trajectory)', 'Rollout time (threshold 0 = all data)');

two_curves(nexttile, x, grab('RawMeanVariance'), err('RawMeanVarianceSEM'), ...
    grab('RawTerminalVariance'), err('RawTerminalVarianceSEM'), ...
    blue, red, black, 'Predictive variance', 'Raw posterior variance');

two_curves(nexttile, x, grab('NormMeanVariance'), err('NormMeanVarianceSEM'), ...
    grab('NormTerminalVariance'), err('NormTerminalVarianceSEM'), ...
    blue, red, black, 'Variance / prior variance', 'Normalized posterior variance');

sgtitle(figure_title);
print(f, out_emf, '-dmeta', '-painters');
fprintf('Saved %s\n', out_emf);
if ~isempty(preview_png)
    exportgraphics(f, preview_png, 'Resolution', 150);
    fprintf('Saved preview %s\n', preview_png);
end
close(f);
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

function v = ternary(c, fa, fb), if c, v = fa(); else, v = fb(); end, end
