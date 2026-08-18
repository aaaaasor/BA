function output_path = plot_second_level_sample(sample_idx)
%PLOT_SECOND_LEVEL_SAMPLE Plot one 17-point second-level parent trajectory.
% The four five-point segments share endpoints, so their union contains
% 4*(5-1)+1 = 17 distinct generated points.

if nargin < 1
    sample_idx = 2;
end
cache_path = fullfile(fileparts(mfilename('fullpath')), 'outputs', ...
    'Racing_SecondLevel_Rollout_SerialTest.mat');
S = load(cache_path, 'segment_traj_path_10d', 'segment_data_transform', ...
    'saved_segment_rollout_constraint');
T = S.segment_data_transform;
standardized = squeeze(S.segment_traj_path_10d(end, :, :));
absolute = standardized .* T.std' + T.mean';
n_segments = 4;
n_points = 5;
n_samples = size(absolute, 1) / n_segments;
if sample_idx < 1 || sample_idx > n_samples || sample_idx ~= round(sample_idx)
    error('sample_idx must be an integer from 1 to %d.', n_samples);
end

segments = zeros(n_points, 2, n_segments);
polyline = zeros(n_segments * (n_points - 1) + 1, 2);
for segment_idx = 1:n_segments
    row_idx = (sample_idx - 1) * n_segments + segment_idx;
    segment = reshape(absolute(row_idx, :), T.feature_dim, [])';
    segments(:, :, segment_idx) = segment(:, 1:2);
    first_point = (segment_idx - 1) * (n_points - 1) + 1;
    polyline(first_point:first_point + n_points - 1, :) = segment(:, 1:2);
end

constraint = S.saved_segment_rollout_constraint;
geometry = constraint.track_boundary_geometry;
track.left = geometry.curves(1).control_points;
track.right = geometry.curves(2).control_points;
track.raceline = nan(size(track.left));

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.18, 0.10, 0.58, 0.78]);
hold on;
draw_track_segment(track, 'HandleVisibility', 'off');
draw_obstacles(constraint.obstacle_physical_geometry, ...
    'HandleVisibility', 'off');

orange = [0.8660, 0.3290, 0.0000];
for segment_idx = 1:n_segments
    segment = segments(:, :, segment_idx);
    plot(segment(:, 1), segment(:, 2), '-o', 'Color', orange, ...
        'MarkerFaceColor', orange, 'MarkerSize', 5, 'LineWidth', 1.6, ...
        'DisplayName', sprintf('Segment %d', segment_idx));
end

anchor_indices = 1:4:17;
plot(polyline(anchor_indices, 1), polyline(anchor_indices, 2), 's', ...
    'Color', [0.10 0.35 0.85], 'MarkerFaceColor', 'w', ...
    'MarkerSize', 8, 'LineWidth', 1.4, 'DisplayName', ...
    'First-level anchors');
for point_idx = 1:size(polyline, 1)
    text(polyline(point_idx, 1), polyline(point_idx, 2), ...
        sprintf('  P%d', point_idx), 'Color', [0.15 0.15 0.15], ...
        'FontSize', 8, 'Clipping', 'on');
end

axis equal;
grid on;
xlabel('x');
ylabel('y');
title(sprintf('Second-level Sample %d: 4 Segments, 17 Ordered Points', ...
    sample_idx));
legend('Location', 'bestoutside');

padding = 0.04;
limits_x = [min(polyline(:, 1)), max(polyline(:, 1))];
limits_y = [min(polyline(:, 2)), max(polyline(:, 2))];
xlim(limits_x + [-padding, padding]);
ylim(limits_y + [-padding, padding]);

output_path = fullfile(fileparts(cache_path), ...
    sprintf('SecondLevel_Sample_%03d.png', sample_idx));
exportgraphics(fig, output_path, 'Resolution', 220);
fprintf('Wrote %s\n', output_path);
end
