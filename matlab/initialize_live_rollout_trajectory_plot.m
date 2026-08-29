function plot_state = initialize_live_rollout_trajectory_plot( ...
    constraint_cfg, sample_idx, x0, t0, n_steps)
%INITIALIZE_LIVE_ROLLOUT_TRAJECTORY_PLOT Set up one incremental RK4 plot.

plot_state = struct('enabled', false);
if ~struct_field_default(constraint_cfg, ...
        'live_trajectory_plot_enabled', false)
    return;
end
sample_indices = struct_field_default(constraint_cfg, ...
    'live_trajectory_plot_sample_indices', 1);
if ~ismember(sample_idx, sample_indices)
    return;
end
requested_position = find(sample_indices == sample_idx, 1, 'first');
group_size = max(1, round(struct_field_default(constraint_cfg, ...
    'live_trajectory_group_size', numel(sample_indices))));
group_idx = ceil(requested_position / group_size);
segment_in_group = mod(requested_position - 1, group_size) + 1;
group_positions = (group_idx-1)*group_size + ...
    (1:min(group_size,numel(sample_indices)-(group_idx-1)*group_size));
present_positions = group_positions(sample_indices(group_positions)>0);
is_group_start = requested_position == present_positions(1);
is_group_end = requested_position == present_positions(end);
original_indices = struct_field_default(constraint_cfg, ...
    'live_trajectory_original_sample_indices', []);
display_sample_idx = sample_idx;
if ~isempty(original_indices)
    display_sample_idx = original_indices(sample_idx);
end
group_labels = struct_field_default(constraint_cfg, ...
    'live_trajectory_group_labels', 1:ceil(numel(sample_indices) / group_size));
if numel(group_labels) >= group_idx
    group_label = group_labels(group_idx);
else
    group_label = group_idx;
end

if isfield(constraint_cfg, 'live_trajectory_point_maps')
    point_maps = constraint_cfg.live_trajectory_point_maps;
elseif isfield(constraint_cfg, 'track_boundary_point_maps')
    point_maps = constraint_cfg.track_boundary_point_maps;
elseif isfield(constraint_cfg, 'obstacle_point_maps')
    point_maps = constraint_cfg.obstacle_point_maps;
else
    warning('Live RK4 trajectory plot skipped: no physical point maps.');
    return;
end
[~, order] = sort([point_maps.point_index]);
point_maps = point_maps(order);
xy0 = state_to_xy(x0, point_maps);

track = struct_field_default(constraint_cfg, ...
    'live_trajectory_track_segment', []);
level_label = char(string(struct_field_default(constraint_cfg, ...
    'live_trajectory_level_label', 'First-level')));
level_key = regexprep(level_label, '[^A-Za-z0-9]', '');
figure_tag = [level_key, 'LiveRK4Trajectory'];
axes_tag = [level_key, 'LiveRK4Axes'];
fig = findall(groot, 'Type', 'figure', 'Tag', figure_tag);
if isempty(fig)
    fig = figure('Name', [level_label, ' RK4 generation'], ...
        'Tag', figure_tag, 'Color', 'w', 'WindowStyle', 'normal', ...
        'Units', 'normalized', 'Position', [0.16, 0.10, 0.60, 0.78]);
    movegui(fig, 'center');
    ax = axes(fig, 'Tag', axes_tag);
else
    fig = fig(1);
    ax = findall(fig, 'Type', 'axes', 'Tag', axes_tag);
    if isempty(ax)
        ax = axes(fig, 'Tag', axes_tag);
    else
        ax = ax(1);
    end
    figure(fig);
end
set(fig, 'Name', sprintf('%s RK4 generation - segment sample %d', ...
    level_label, display_sample_idx));
set(fig, 'CurrentAxes', ax);

% Rebuild the static layer only for the first requested sample (or after a
% user has cleared the figure). Later samples reuse this same background.
first_requested_sample = sample_indices(find(sample_indices>0,1));
background_ready = isappdata(ax, 'joint_softmin_background_ready') && ...
    getappdata(ax, 'joint_softmin_background_ready');
if sample_idx == first_requested_sample || ~background_ready
    cla(ax);
    hold(ax, 'on');
    draw_joint_softmin_safe_set_background(ax, constraint_cfg, track);
    setappdata(ax, 'joint_softmin_background_ready', true);
else
    delete(findall(ax, 'Tag', 'LiveRK4Dynamic'));
    if is_group_start
        delete(findall(ax, 'Tag', 'LiveRK4Completed'));
    end
    legend(ax, 'off');
    hold(ax, 'on');
end

initial_handle = plot(ax, xy0(:, 1), xy0(:, 2), '--o', ...
    'Color', [0.45, 0.45, 0.45], 'MarkerFaceColor', [0.75, 0.75, 0.75], ...
    'LineWidth', 1.1, 'DisplayName', 'initial curve', ...
    'Tag', 'LiveRK4Dynamic');
trace_colors = lines(numel(point_maps));
trace_handles = gobjects(numel(point_maps), 1);
for point_idx = 1:numel(point_maps)
    trace_handles(point_idx) = animatedline(ax, ...
        'Color', 0.72 .* trace_colors(point_idx, :) + 0.28, ...
        'LineWidth', 0.9, 'HandleVisibility', 'off');
    set(trace_handles(point_idx), 'Tag', 'LiveRK4Dynamic');
    addpoints(trace_handles(point_idx), xy0(point_idx, 1), xy0(point_idx, 2));
end
segment_color = [0.00, 0.35, 0.85];
if struct_field_default(constraint_cfg, ...
        'live_trajectory_alternating_segment_colors', false)
    alternating_colors = [0.85, 0.10, 0.10; 0.10, 0.65, 0.20];
    segment_color = alternating_colors(mod(segment_in_group - 1, 2) + 1, :);
end
current_handle = plot(ax, xy0(:, 1), xy0(:, 2), '-o', ...
    'Color', segment_color, 'MarkerFaceColor', segment_color, ...
    'LineWidth', 2.0, 'MarkerSize', 5.5, ...
    'DisplayName', 'current generated curve', 'Tag', 'LiveRK4Dynamic');

axis(ax, 'equal');
if ~isempty(track) && isfield(track, 'left') && isfield(track, 'right')
    all_xy = [track.left; track.right];
    x_pad = max(0.025, 0.04 * (max(all_xy(:, 1)) - min(all_xy(:, 1))));
    y_pad = max(0.025, 0.04 * (max(all_xy(:, 2)) - min(all_xy(:, 2))));
    xlim(ax, [min(all_xy(:, 1)) - x_pad, max(all_xy(:, 1)) + x_pad]);
    ylim(ax, [min(all_xy(:, 2)) - y_pad, max(all_xy(:, 2)) + y_pad]);
end
xlabel(ax, 'x');
ylabel(ax, 'y');
grid(ax, 'off');
legend(ax, [initial_handle, current_handle], 'Location', 'best');
title(ax, sprintf(['%s trajectory %d, segment %d/%d ', ...
    '(sample %d): RK4 step 0/%d, t = %.4f'], ...
    level_label, group_label, segment_in_group, group_size, ...
    display_sample_idx, n_steps, t0));
drawnow;

% One VideoWriter is shared by the child segments of one parent trajectory.
% Start a fresh numbered MP4 at the first child of every parent.
video_enabled = struct_field_default(constraint_cfg, ...
    'live_trajectory_video_enabled', false);
if video_enabled && is_group_start
    video_cfg = constraint_cfg;
    output_pattern = struct_field_default(constraint_cfg, ...
        'live_trajectory_video_output_pattern', '');
    if ~isempty(output_pattern)
        [video_dir, video_name_pattern, video_ext] = fileparts( ...
            char(string(output_pattern)));
        video_filename = sprintf( ...
            [video_name_pattern, video_ext], group_label);
        video_cfg.live_trajectory_video_output_path = ...
            fullfile(video_dir, video_filename);
    end
    live_rollout_video_writer('start', fig, video_cfg);
end

plot_state = struct( ...
    'enabled', true, ...
    'figure', fig, ...
    'axes', ax, ...
    'point_maps', point_maps, ...
    'level_label', level_label, ...
    'group_idx', group_idx, ...
    'group_label', group_label, ...
    'segment_in_group', segment_in_group, ...
    'group_size', group_size, ...
    'video_group_end', is_group_end, ...
    'display_sample_idx', display_sample_idx, ...
    'current_handle', current_handle, ...
    'trace_handles', trace_handles);
end

function xy = state_to_xy(x, point_maps)
xy = zeros(numel(point_maps), 2);
for point_idx = 1:numel(point_maps)
    p = point_maps(point_idx).M * x(:) + point_maps(point_idx).o;
    xy(point_idx, :) = p(:)';
end
end
