function plot_pchip_vs_real_track(cfg)
%PLOT_PCHIP_VS_REAL_TRACK Compare the pchip boundary splines against the raw track.
% The boundary CBF only ever sees geometry.curves(1|2), i.e. a pchip through
% n_spline_points arclength-resampled knots. This figure quantifies how far
% that representation sits from the raw Nuerburgring CSV polyline it came from.
%
% Called from main_demo behind cfg.output.plot_pchip_vs_raw_track. Reuses the
% segment and geometry the pipeline already built so the figure shows what the
% CBF actually sees; rebuilds them from the CSV when called standalone.
if nargin < 1
    cfg = struct();
end
this_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(this_dir, 'trajectory_data');

%% Pipeline geometry (exactly what the CBF uses)
segment = struct_field_default(cfg, 'track_segment', []);
n_control_points = 400;
if isfield(cfg, 'track_boundary')
    n_control_points = struct_field_default(cfg.track_boundary, ...
        'n_spline_points', 400);
end
if isempty(segment)
    segment = extract_track_segment(fullfile(data_dir, 'Nuerburgring.csv'), ...
        fullfile(data_dir, 'Nuerburgring_raceline.csv'), ...
        struct('s_range_m', [250, 1150]));
end
s_range_m = segment.meta.s_range_m;
if isfield(cfg, 'track_boundary') && isfield(cfg.track_boundary, 'geometry')
    G = cfg.track_boundary.geometry;
    n_control_points = G.n_control_points;
else
    G = build_track_boundary_geometry(segment, n_control_points);
end
to_metric = segment.transform.to_metric;

s_dense = linspace(0, 1, 20000)';
pch_left = to_metric([ppval(G.curves(1).pp_x, s_dense), ...
    ppval(G.curves(1).pp_y, s_dense)]);
pch_right = to_metric([ppval(G.curves(2).pp_x, s_dense), ...
    ppval(G.curves(2).pp_y, s_dense)]);
control_left = to_metric(G.curves(1).control_points);
control_right = to_metric(G.curves(2).control_points);

%% Raw CSV track at native resolution
raw = readmatrix(fullfile(data_dir, 'Nuerburgring.csv'), 'NumHeaderLines', 1);
c_all = raw(:, 1:2);
wr_all = raw(:, 3);
wl_all = raw(:, 4);
s_all = [0; cumsum(vecnorm(diff(c_all), 2, 2))];
keep = s_all >= max(s_range_m(1), s_all(1)) & s_all <= min(s_range_m(2), s_all(end));
c_raw = c_all(keep, :);
wl = wl_all(keep);
wr = wr_all(keep);
s_raw = s_all(keep) - s_all(find(keep, 1));
tan_raw = gradient_rows(c_raw);
tan_raw = tan_raw ./ max(vecnorm(tan_raw, 2, 2), eps);
nl_raw = [-tan_raw(:, 2), tan_raw(:, 1)];
raw_left = c_raw + nl_raw .* wl;
raw_right = c_raw - nl_raw .* wr;

%% Deviation: raw boundary point -> pchip curve
dev_left = point_to_polyline(raw_left, pch_left);
dev_right = point_to_polyline(raw_right, pch_right);
[max_dev, max_idx] = max([dev_left; dev_right]);
if max_idx <= numel(dev_left)
    max_side = 'left';
else
    max_side = 'right';
end

fprintf('raw CSV points in [%g, %g] m : %d  (mean spacing %.2f m)\n', ...
    s_range_m(1), s_range_m(2), size(c_raw, 1), mean(vecnorm(diff(c_raw), 2, 2)));
fprintf('pchip knots                  : %d  (spacing %.2f m)\n', ...
    n_control_points, (s_raw(end) - s_raw(1)) / (n_control_points - 1));
fprintf('deviation left  (m): median %.4f  p95 %.4f  max %.4f\n', ...
    median(dev_left), prctile(dev_left, 95), max(dev_left));
fprintf('deviation right (m): median %.4f  p95 %.4f  max %.4f\n', ...
    median(dev_right), prctile(dev_right, 95), max(dev_right));
fprintf('worst deviation %.4f m on the %s rail\n', max_dev, max_side);

%% Figure
c_left = [0.231, 0.435, 0.831];
c_right = [0.851, 0.467, 0.024];
c_raw_left = [0.05, 0.05, 0.05];
c_raw_right = [0.30, 0.30, 0.30];
fig = figure('Color', 'w', 'Units', 'normalized', ...
    'Position', [0.14, 0.10, 0.60, 0.80]);
ax1 = axes(fig);
hold(ax1, 'on');
grid(ax1, 'on');
box(ax1, 'on');
plot(ax1, raw_left(:, 1), raw_left(:, 2), '-', 'Color', c_raw_left, ...
    'LineWidth', 2.6, 'DisplayName', 'raw left boundary');
plot(ax1, raw_right(:, 1), raw_right(:, 2), '-', 'Color', c_raw_right, ...
    'LineWidth', 2.6, 'DisplayName', 'raw right boundary');
plot(ax1, pch_left(:, 1), pch_left(:, 2), '--', 'Color', c_left, ...
    'LineWidth', 1.5, 'DisplayName', 'pchip left');
plot(ax1, pch_right(:, 1), pch_right(:, 2), '--', 'Color', c_right, ...
    'LineWidth', 1.5, 'DisplayName', 'pchip right');
scatter(ax1, control_left(:, 1), control_left(:, 2), 14, ...
    'o', 'MarkerEdgeColor', c_left, 'MarkerFaceColor', 'w', ...
    'LineWidth', 0.8, 'DisplayName', 'pchip left control points');
scatter(ax1, control_right(:, 1), control_right(:, 2), 14, ...
    'o', 'MarkerEdgeColor', c_right, 'MarkerFaceColor', 'w', ...
    'LineWidth', 0.8, 'DisplayName', 'pchip right control points');
axis(ax1, 'equal');
xlabel(ax1, 'x [m]');
ylabel(ax1, 'y [m]');
title(ax1, sprintf(['pchip boundary splines vs raw Nuerburgring CSV ', ...
    '(s = %g-%g m, %d knots)'], s_range_m(1), s_range_m(2), n_control_points), ...
    'FontWeight', 'bold');
legend(ax1, 'Location', 'best');
set(ax1, 'GridAlpha', 0.12);

export_enabled = true;
if isfield(cfg, 'output')
    export_enabled = struct_field_default(cfg.output, 'enabled', true);
end
if export_enabled
    out_dir = fullfile(this_dir, 'outputs');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    emf_path = fullfile(out_dir, 'Track_Pchip_vs_Raw.emf');
    export_graphics_compat(fig, emf_path);
    fprintf('wrote %s\n', emf_path);
end
end

function d = point_to_polyline(pts, poly)
d = zeros(size(pts, 1), 1);
a = poly(1:end - 1, :);
b = poly(2:end, :);
ab = b - a;
den = max(sum(ab .^ 2, 2), eps);
for k = 1:size(pts, 1)
    ap = pts(k, :) - a;
    u = min(max(sum(ap .* ab, 2) ./ den, 0), 1);
    d(k) = min(vecnorm(ap - u .* ab, 2, 2));
end
end
