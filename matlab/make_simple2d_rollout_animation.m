function make_simple2d_rollout_animation(out_stem, preview_frame)
%MAKE_SIMPLE2D_ROLLOUT_ANIMATION  Animated version of the Simple2D rollout
% figure, using the same styling as make_simple2d_rollout_figure:
%   * one colour for every generated point;
%   * trajectory lines as a single-hue ramp along generation time.
% The trail is drawn progressively; the moving points share the endpoint colour.

root = 'C:\Users\JieJi\BA\matlab';
[d_partial, d_all] = simple2d_dirs(root);
if nargin < 1 || isempty(out_stem)
    out_stem = fullfile(d_partial, 'Simple2D_Rollout_Animation');
end
if nargin < 2, preview_frame = {}; end   % {step, png_path}, scratch preview only

P = load(fullfile(d_partial, 'Partial_Rollout.mat'), 'run');
A = load(fullfile(d_all,     'AllData_Rollout.mat'), 'run');
D = load(fullfile(d_partial, 'Training_Data_and_Seeds.mat'), 'target_points');
C = readtable(fullfile(d_partial, 'Comparison.csv'));
ret = C.RetentionPercent;

runs   = {P.run, A.run};
labels = {sprintf('Partial data (%.2f%%)', ret(1)), ...
          sprintf('All data (%.0f%%)',     ret(2))};
cmap = single_hue_ramp(256);
point_color  = [0.85 0.33 0.10];
target_color = [0.78 0.78 0.78];

t = runs{1}.times(:);
n_t = numel(t);
assert(numel(runs{2}.times) == n_t, 'the two rollouts must share the time grid');

xy_all = D.target_points;
for k = 1:2, xy_all = [xy_all; reshape(runs{k}.path, [], 2)]; end %#ok<AGROW>
lo = min(xy_all, [], 1); hi = max(xy_all, [], 1);
pad = 0.06 * (hi - lo);

f = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1100 470]);
tl = tiledlayout(f, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
ax = gobjects(1, 2); trail = cell(1, 2); head = gobjects(1, 2);
for k = 1:2
    r = runs{k};
    ax(k) = nexttile(tl); hold(ax(k), 'on'); colormap(ax(k), cmap);
    scatter(ax(k), D.target_points(:,1), D.target_points(:,2), 6, target_color, ...
        'filled', 'MarkerFaceAlpha', 0.55, 'DisplayName', 'Target samples');
    n = size(r.path, 2);
    trail{k} = gobjects(n, 1);
    for i = 1:n
        trail{k}(i) = patch(ax(k), 'XData', nan, 'YData', nan, 'CData', nan, ...
            'FaceColor', 'none', 'EdgeColor', 'interp', 'LineWidth', 0.9, ...
            'HandleVisibility', 'off');
    end
    scatter(ax(k), r.path(1,:,1), r.path(1,:,2), 26, 'k', 'o', ...
        'LineWidth', 0.7, 'DisplayName', 'Initial states');
    head(k) = scatter(ax(k), r.path(1,:,1), r.path(1,:,2), 26, point_color, ...
        'filled', 'DisplayName', 'Generated points');
    caxis(ax(k), [t(1) t(end)]);
    axis(ax(k), 'equal');
    xlim(ax(k), [lo(1)-pad(1), hi(1)+pad(1)]);
    ylim(ax(k), [lo(2)-pad(2), hi(2)+pad(2)]);
    grid(ax(k), 'on'); box(ax(k), 'on');
    xlabel(ax(k), 'x_1'); ylabel(ax(k), 'x_2'); title(ax(k), labels{k});
    set(ax(k), 'FontSize', 11, 'FontName', 'Times New Roman');
    if k == 1
        legend(ax(k), 'Location', 'southeast', 'Box', 'off', 'FontSize', 9);
    end
end
cb = colorbar(ax(2), 'eastoutside');
cb.Label.String = 'Generation time t';
ttl = sgtitle(tl, '', 'FontSize', 12, 'FontName', 'Times New Roman');

vw = VideoWriter([out_stem '.mp4'], 'MPEG-4');
vw.FrameRate = 20; vw.Quality = 95; open(vw);
gif_path = [out_stem '.gif'];

for step = 1:n_t
    for k = 1:2
        r = runs{k};
        for i = 1:size(r.path, 2)
            set(trail{k}(i), 'XData', [r.path(1:step,i,1); nan], ...
                'YData', [r.path(1:step,i,2); nan], 'CData', [t(1:step); nan]);
        end
        set(head(k), 'XData', r.path(step,:,1), 'YData', r.path(step,:,2));
    end
    ttl.String = sprintf(['2D LoG-GP rollout trajectories ' ...
        '(same 100 initial samples, no guidance)   t = %.2f'], t(step));
    drawnow limitrate;
    frame = getframe(f);
    writeVideo(vw, frame);
    [gif_idx, gif_map] = rgb2ind(frame2im(frame), 256);
    if step == 1
        imwrite(gif_idx, gif_map, gif_path, 'gif', 'LoopCount', inf, 'DelayTime', 0.05);
    else
        imwrite(gif_idx, gif_map, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
    end
    if ~isempty(preview_frame) && step == preview_frame(1)
        % preview_frame = {step, png_path}; scratch only, never beside the EMF.
        if numel(preview_frame) > 1 && ischar(preview_frame{2}) %#ok<ISCL>
            exportgraphics(f, preview_frame{2}, 'Resolution', 140);
        end
    end
end
close(vw); close(f);
fprintf('wrote %s.mp4  (%d frames @ %d fps)\n', out_stem, n_t, vw.FrameRate);
fprintf('wrote %s.gif\n', out_stem);
end

% =====================================================================
function cmap = single_hue_ramp(n)
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
