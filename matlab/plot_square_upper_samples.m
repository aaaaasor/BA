function output_path = plot_square_upper_samples
%PLOT_SQUARE_UPPER_SAMPLES Isolate the orange curves passing above square.

sample_indices = [44, 65, 93];
root_dir = fileparts(mfilename('fullpath'));
cache_path = fullfile(root_dir, 'outputs', ...
    'Racing_SecondLevel_Rollout_SerialTest.mat');
S = load(cache_path, 'segment_traj_path_10d', 'segment_data_transform', ...
    'saved_segment_rollout_constraint');
T = S.segment_data_transform;
standardized = squeeze(S.segment_traj_path_10d(end, :, :));
absolute = standardized .* T.std' + T.mean';
constraint = S.saved_segment_rollout_constraint;

n_segments = 4;
n_points = 5;
curves = zeros(17, 2, numel(sample_indices));
for curve_idx = 1:numel(sample_indices)
    sample_idx = sample_indices(curve_idx);
    for segment_idx = 1:n_segments
        row_idx = (sample_idx - 1) * n_segments + segment_idx;
        segment = reshape(absolute(row_idx, :), T.feature_dim, [])';
        first_point = (segment_idx - 1) * (n_points - 1) + 1;
        curves(first_point:first_point + n_points - 1, :, curve_idx) = ...
            segment(:, 1:2);
    end
end

geometry = constraint.track_boundary_geometry;
track.left = geometry.curves(1).control_points;
track.right = geometry.curves(2).control_points;
track.raceline = nan(size(track.left));
orange = [0.8660, 0.3290, 0.0000];

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.10, 0.16, 0.80, 0.62]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
draw_background(track, constraint.obstacle_physical_geometry);
draw_curves(curves, sample_indices, orange, true);
axis equal;
grid on;
xlabel('x'); ylabel('y');
title('Complete Second-level Curves');
legend('Location', 'best');

nexttile;
draw_background(track, constraint.obstacle_physical_geometry);
draw_curves(curves, sample_indices, orange, true);
square_center = constraint.obstacle_physical_geometry.centers(:, 1);
square_axes = constraint.obstacle_physical_geometry.semi_axes(:, 1);
axis equal;
xlim(square_center(1) + 2.2 * square_axes(1) * [-1, 1]);
ylim(square_center(2) + 2.8 * square_axes(2) * [-1, 1]);
grid on;
xlabel('x'); ylabel('y');
title('Square Upper-side Detail');

sgtitle('Orange Curves Near Square: Samples 44, 65, and 93');
output_path = fullfile(root_dir, 'outputs', ...
    'SecondLevel_SquareUpper_Samples044_065_093.png');
exportgraphics(fig, output_path, 'Resolution', 220);
fprintf('Wrote %s\n', output_path);
end

function draw_background(track, obstacle)
hold on;
draw_track_segment(track, 'HandleVisibility', 'off');
draw_obstacles(obstacle, 'HandleVisibility', 'off');
end

function draw_curves(curves, sample_indices, color, add_labels)
styles = {'-', '--', '-.'};
for curve_idx = 1:numel(sample_indices)
    curve = curves(:, :, curve_idx);
    plot(curve(:, 1), curve(:, 2), styles{curve_idx}, ...
        'Color', color, 'Marker', 'o', 'MarkerFaceColor', color, ...
        'MarkerSize', 4.5, 'LineWidth', 1.8, ...
        'DisplayName', sprintf('Sample %d', sample_indices(curve_idx)));
    if add_labels
        for point_idx = 1:size(curve, 1)
            text(curve(point_idx, 1), curve(point_idx, 2), ...
                sprintf(' %d:P%d', sample_indices(curve_idx), point_idx), ...
                'FontSize', 7, 'Color', [0.20 0.20 0.20], ...
                'Clipping', 'on');
        end
    end
end
end
