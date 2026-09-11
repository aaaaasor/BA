function result = plot_1d_density_evolution(n_grid, bandwidth)
%PLOT_1D_DENSITY_EVOLUTION  How the generated distribution evolves in time.
%
% The existing two-panel figure shows only t = 0 and t = 1.  This fills in the
% middle: for every model, the kernel density of the 1000 generated samples is
% evaluated on a fixed x grid at every rollout step, giving p(x, t) as a heat
% map.  All panels share one colour scale so they can be compared directly.
%
% The true target mixture is drawn on the right edge of each panel, so a model
% whose terminal density lands in the wrong place is visible immediately.
%
% This is the 1D companion to the variance-map figure: that one shows where the
% GP is uncertain, this one shows where the samples actually go.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_grid),   n_grid = 241; end
if nargin < 2 || isempty(bandwidth), bandwidth = 0.18; end

out = fullfile('outputs', ['1d case' char([20840 23616 71 80])]);
S = load(fullfile(out, 'Simple1D_GlobalGP_Threshold_Sweep.mat'), 'sweep');
s = S.sweep;
thresholds = s.thresholds;
nm = numel(thresholds) + 1;
times = s.runs{1}.times;
n_t = numel(times);

x_all = [];
for i = 1:nm, x_all = [x_all; s.runs{i}.path(:)]; end %#ok<AGROW>
lo = prctile(x_all, 0.2); hi = prctile(x_all, 99.8);
pad = 0.06 * (hi - lo);
xg = linspace(lo - pad, hi + pad, n_grid)';

% True target mixture, for the reference curve on the right edge.
truth = 0.5 * normpdf(xg, -2.0, 0.45) + 0.5 * normpdf(xg, 2.0, 0.55);
truth = truth / max(truth);

labels = cell(nm, 1); D = cell(nm, 1);
for i = 1:nm
    P = squeeze(s.runs{i}.path(:, :, 1));      % (time x sample)
    M = zeros(n_grid, n_t);
    for k = 1:n_t
        M(:, k) = kde1(xg, P(k, :), bandwidth);
    end
    D{i} = M;
    if i <= numel(thresholds)
        labels{i} = sprintf('threshold %.2f: %.2f%% of points', thresholds(i), ...
            s.comparison.RetentionPercent(i));
    else
        labels{i} = sprintf('all data: %.2f%% of points', ...
            s.comparison.RetentionPercent(i));
    end
end
% One shared colour scale, so the panels are comparable.
cmax = max(cellfun(@(M) max(M(:)), D));

f = figure('Visible', 'off', 'Color', 'w', 'Position', [20 30 1800 1050], ...
    'Renderer', 'painters');
tl = tiledlayout(f, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
ax_list = gobjects(nm, 1);
for i = 1:nm
    ax = nexttile(tl); ax_list(i) = ax;
    imagesc(ax, times, xg, D{i}); set(ax, 'YDir', 'normal'); hold(ax, 'on');
    contour(ax, times, xg, D{i}, cmax*[.15 .35 .6 .85], 'w:', 'LineWidth', .4);
    % True target density, drawn as a curve hugging the right edge.
    w = 0.16 * (times(end) - times(1));
    plot(ax, times(end) - w*truth, xg, 'w-', 'LineWidth', 1.6);
    plot(ax, times(end) - w*truth, xg, '-', 'Color', [0 0 0 0.55], 'LineWidth', .8);
    yline(ax, -2.0, 'w--', 'LineWidth', .6); yline(ax, 2.0, 'w--', 'LineWidth', .6);
    clim(ax, [0 cmax]);
    xlabel(ax, 'Generation time t'); ylabel(ax, 'State x'); title(ax, labels{i});
    xlim(ax, [times(1) times(end)]); ylim(ax, [xg(1) xg(end)]);
    box(ax, 'on'); set(ax, 'FontName', 'Times New Roman', 'FontSize', 10);
end
colormap(ax_list(1), turbo(256));
for i = 2:nm, colormap(ax_list(i), turbo(256)); end

% The 2x3 layout leaves one tile free; use it for the legend so the data
% panels stay uncluttered.
axl = nexttile(tl); hold(axl, 'on');
lh = gobjects(3, 1);
lh(1) = plot(axl, nan, nan, 'k-',  'LineWidth', 1.6);
lh(2) = plot(axl, nan, nan, 'k--', 'LineWidth', 1.0);
lh(3) = plot(axl, nan, nan, 'k:',  'LineWidth', 1.2);
lvl = sprintf('%.2f, ', cmax*[.15 .35 .6 .85]);
legend(axl, lh, { ...
    'True target mixture density (right edge)', ...
    'Target mode centres, x = \pm2', ...
    ['Density contours at ' lvl(1:end-2)]}, ...
    'Location', 'northwest', 'Box', 'off', 'FontSize', 11);
axis(axl, 'off');
text(axl, 0.02, 0.42, sprintf(['Colour: kernel density of the 1000 generated\n' ...
    'samples at each rollout step, Gaussian kernel,\n' ...
    'bandwidth %.2f. All panels share one colour scale.'], bandwidth), ...
    'Units', 'normalized', 'FontName', 'Times New Roman', 'FontSize', 10, ...
    'VerticalAlignment', 'top');

cb = colorbar(ax_list(nm), 'eastoutside');
cb.Label.String = 'Generated sample density (shared scale)';
sgtitle(tl, '1D global exact GP: generated distribution over generation time');

emf = fullfile(out, 'Simple1D_GlobalGP_Density_Evolution.emf');
print(f, emf, '-dmeta', '-painters');
close(f);
fprintf('Saved %s\n', emf);

% How far each model's density is from the target, as a function of time.
kl_t = zeros(n_t, nm);
tgt = 0.5*normpdf(xg, -2.0, 0.45) + 0.5*normpdf(xg, 2.0, 0.55);
tgt = tgt / sum(tgt);
for i = 1:nm
    for k = 1:n_t
        q = D{i}(:, k); q = q / sum(q);
        kl_t(k, i) = sum(tgt .* log(max(tgt, 1e-300) ./ max(q, 1e-300)));
    end
end
f2 = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 900 460], ...
    'Renderer', 'painters');
ax = axes(f2); hold(ax, 'on');
cols = lines(nm);
h = gobjects(nm, 1);
for i = 1:nm
    h(i) = plot(ax, times, kl_t(:, i), '-', 'Color', cols(i, :), 'LineWidth', 1.6);
end
grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'Generation time t');
ylabel(ax, 'KL divergence: target vs generated');
title(ax, 'Distance to the target distribution over generation time');
legend(ax, h, labels, 'Location', 'northeast', 'Box', 'off', 'FontSize', 9);
set(ax, 'FontName', 'Times New Roman', 'FontSize', 11, 'YScale', 'log');
emf2 = fullfile(out, 'Simple1D_GlobalGP_Density_KL_Over_Time.emf');
print(f2, emf2, '-dmeta', '-painters');
close(f2);
fprintf('Saved %s\n', emf2);

result = struct('times', times, 'x_grid', xg, 'densities', {D}, ...
    'labels', {labels}, 'kl_over_time', kl_t, 'bandwidth', bandwidth, ...
    'shared_color_limit', cmax, 'emf_density', emf, 'emf_kl', emf2);
save(fullfile(out, 'Simple1D_GlobalGP_Density_Evolution.mat'), 'result', '-v7.3');

fprintf('\nD_KL(target || generated) at selected times\n');
fprintf('%-34s', 'model');
ks = [1 round(n_t*0.25) round(n_t*0.5) round(n_t*0.75) n_t];
for k = ks, fprintf('t=%-8.2f', times(k)); end
fprintf('\n');
for i = 1:nm
    fprintf('%-34s', labels{i});
    for k = ks, fprintf('%-10.4f', kl_t(k, i)); end
    fprintf('\n');
end
end

% =====================================================================
function d = kde1(grid, samples, bw)
samples = samples(:)';
d = sum(exp(-0.5*((grid - samples)/bw).^2), 2);
d = d / (numel(samples) * bw * sqrt(2*pi));
end
