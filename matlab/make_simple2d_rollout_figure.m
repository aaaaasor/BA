function make_simple2d_rollout_figure(out_stem, preview_png, n_bands)
%MAKE_SIMPLE2D_ROLLOUT_FIGURE  Simple2D rollout trajectories, revised styling.
%
% Replaces Simple2D_Rollout_Trajectories.emf.  Two changes requested by the
% supervisor:
%   * every generated endpoint uses ONE colour (previously one colour per
%     trajectory, which carried no information);
%   * trajectory lines are a single-hue ramp along generation time t.
%
% The ramp is drawn as N_BANDS constant-colour line groups, NOT as a patch
% with 'EdgeColor','interp'.  EMF has no per-vertex colour interpolation along
% a stroke, so an interpolated patch looks correct on screen and in PNG but
% degrades when printed to EMF.  Banding keeps every primitive a plain solid
% line, which EMF reproduces exactly.  Segments in adjacent bands share a
% vertex, so the curves stay gap-free.
%
% The original generating script no longer exists in the repository; this one
% rebuilds the figure from the archived rollout caches.

root = 'C:\Users\JieJi\BA\matlab';
[d_partial, d_all] = simple2d_dirs(root);
if nargin < 1 || isempty(out_stem)
    out_stem = fullfile(d_partial, 'Simple2D_Rollout_Trajectories');
end
if nargin < 2, preview_png = ''; end
if nargin < 3 || isempty(n_bands), n_bands = 32; end

P = load(fullfile(d_partial, 'Partial_Rollout.mat'), 'run');
A = load(fullfile(d_all,     'AllData_Rollout.mat'), 'run');
D = load(fullfile(d_partial, 'Training_Data_and_Seeds.mat'), 'target_points');
C = readtable(fullfile(d_partial, 'Comparison.csv'));
retention = C.RetentionPercent;

runs   = {P.run, A.run};
labels = {sprintf('Partial data (%.2f%%)', retention(1)), ...
          sprintf('All data (%.0f%%)',     retention(2))};

cmap = single_hue_ramp(n_bands);
endpoint_color = [0.85 0.33 0.10];
target_color   = [0.78 0.78 0.78];

f = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1100 470], ...
    'Renderer', 'painters');
tl = tiledlayout(f, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

xy_all = D.target_points;
for k = 1:2, xy_all = [xy_all; reshape(runs{k}.path, [], 2)]; end %#ok<AGROW>
lo = min(xy_all, [], 1); hi = max(xy_all, [], 1);
pad = 0.06 * (hi - lo);

ax = gobjects(1, 2);
for k = 1:2
    r = runs{k};
    t = r.times(:);
    ax(k) = nexttile(tl); hold(ax(k), 'on');
    colormap(ax(k), cmap);

    scatter(ax(k), D.target_points(:,1), D.target_points(:,2), 6, target_color, ...
        'filled', 'MarkerFaceAlpha', 0.55, 'DisplayName', 'Target samples');

    draw_banded_gradient(ax(k), r.path, t, cmap, 0.9);

    clim(ax(k), [t(1) t(end)]);
    scatter(ax(k), r.path(1,:,1), r.path(1,:,2), 26, 'k', 'o', ...
        'LineWidth', 0.7, 'DisplayName', 'Initial states');
    scatter(ax(k), r.path(end,:,1), r.path(end,:,2), 26, endpoint_color, ...
        'filled', 'DisplayName', 'Generated endpoints');

    axis(ax(k), 'equal');
    xlim(ax(k), [lo(1)-pad(1), hi(1)+pad(1)]);
    ylim(ax(k), [lo(2)-pad(2), hi(2)+pad(2)]);
    grid(ax(k), 'on'); box(ax(k), 'on');
    xlabel(ax(k), 'x_1'); ylabel(ax(k), 'x_2');
    title(ax(k), labels{k});
    set(ax(k), 'FontSize', 11, 'FontName', 'Times New Roman');
    if k == 1
        legend(ax(k), 'Location', 'southeast', 'Box', 'off', 'FontSize', 9);
    end
end

cb = colorbar(ax(2), 'eastoutside');
cb.Label.String = 'Generation time t';
cb.Label.FontSize = 10;
sgtitle(tl, '2D LoG-GP rollout trajectories (same 100 initial samples, no guidance)', ...
    'FontSize', 12, 'FontName', 'Times New Roman');

print(f, [out_stem '.emf'], '-dmeta', '-painters');
fprintf('wrote %s.emf  (%d colour bands)\n', out_stem, n_bands);
% PNG is only ever a scratch preview; the deliverable is vector EMF.
if ~isempty(preview_png)
    exportgraphics(f, preview_png, 'Resolution', 160);
    fprintf('wrote preview %s\n', preview_png);
end
close(f);
end

% =====================================================================
function draw_banded_gradient(ax, path, t, cmap, lw)
% One solid-colour line object per colour band, all trajectories pooled and
% separated by NaN.  n_bands objects instead of n_traj*n_steps.
n_bands = size(cmap, 1);
n_steps = size(path, 1) - 1;
n_traj  = size(path, 2);
tmid = 0.5 * (t(1:end-1) + t(2:end));
edges = linspace(t(1), t(end), n_bands + 1);
band = discretize(tmid, edges);
band(isnan(band)) = n_bands;

for b = 1:n_bands
    seg = find(band == b);
    if isempty(seg), continue; end
    % Each segment contributes [p_k; p_{k+1}; nan] for every trajectory.
    X = nan(3 * numel(seg) * n_traj, 1);
    Y = X;
    w = 0;
    for i = 1:n_traj
        for s = seg(:)'
            X(w+1:w+3) = [path(s, i, 1); path(s+1, i, 1); nan];
            Y(w+1:w+3) = [path(s, i, 2); path(s+1, i, 2); nan];
            w = w + 3;
        end
    end
    line(ax, X(1:w), Y(1:w), 'Color', cmap(b, :), 'LineWidth', lw, ...
        'HandleVisibility', 'off');
end
end

function cmap = single_hue_ramp(n)
% Light -> dark ramp of one hue.  Light end = t 0, dark end = t 1.
light = [0.84 0.89 0.96];
dark  = [0.05 0.19 0.42];
w = linspace(0, 1, n)';
cmap = light + (dark - light) .* w;
end

function [d_partial, d_all] = simple2d_dirs(root)
d_partial = fullfile(root, 'outputs', ['2d case' char([37096 20998 25968 25454])]);
d_all     = fullfile(root, 'outputs', ['2d case ' char([20840 37096 25968 25454])]);
assert(isfolder(d_partial), 'missing %s', d_partial);
assert(isfolder(d_all),     'missing %s', d_all);
end
