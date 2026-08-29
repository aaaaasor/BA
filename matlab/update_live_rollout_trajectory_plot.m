function plot_state = update_live_rollout_trajectory_plot( ...
    plot_state, constraint_cfg, sample_idx, step_idx, n_steps, t_now, x_now)
%UPDATE_LIVE_ROLLOUT_TRAJECTORY_PLOT Draw the newly committed sample_path row.

if ~isstruct(plot_state) || ~struct_field_default(plot_state, 'enabled', false)
    return;
end
if ~isgraphics(plot_state.figure) || ~isgraphics(plot_state.current_handle)
    plot_state.enabled = false;
    return;
end
stride = max(1, round(struct_field_default(constraint_cfg, ...
    'live_trajectory_plot_stride', 1)));
if mod(step_idx, stride) ~= 0 && step_idx ~= n_steps
    return;
end

xy = state_to_xy(x_now, plot_state.point_maps);
set(plot_state.current_handle, 'XData', xy(:, 1), 'YData', xy(:, 2));
for point_idx = 1:numel(plot_state.trace_handles)
    addpoints(plot_state.trace_handles(point_idx), ...
        xy(point_idx, 1), xy(point_idx, 2));
end
level_label = struct_field_default(plot_state, 'level_label', 'First-level');
group_idx = struct_field_default(plot_state, 'group_label', ...
    struct_field_default(plot_state, 'group_idx', 1));
display_sample_idx = struct_field_default(plot_state, 'display_sample_idx', sample_idx);
segment_in_group = struct_field_default(plot_state, 'segment_in_group', 1);
group_size = struct_field_default(plot_state, 'group_size', 1);
title(plot_state.axes, sprintf([ ...
    '%s trajectory %d, segment %d/%d (sample %d): ', ...
    'RK4 step %d/%d, t = %.4f'], ...
    level_label, group_idx, segment_in_group, group_size, display_sample_idx, ...
    step_idx, n_steps, t_now));
drawnow;
delay = struct_field_default(constraint_cfg, ...
    'live_trajectory_plot_delay', 0.0);
if delay > 0
    pause(delay);
end

% In third-level live mode, retain the completed child segment in the same
% axes. The next sample initialization deletes only LiveRK4Dynamic objects,
% so the 16 final segments accumulate into one complete trajectory.
if step_idx == n_steps && struct_field_default(constraint_cfg, ...
        'live_trajectory_accumulate_segments', false)
    set(plot_state.current_handle, 'Tag', 'LiveRK4Completed', ...
        'Marker', 'none', 'LineWidth', 1.6, ...
        'HandleVisibility', 'off');
    drawnow;
end

% Append the visible figure after every displayed step. At the final step
% of the final requested child segment, close the shared writer so MATLAB
% finalizes the MP4 container while leaving the figure open.
if struct_field_default(constraint_cfg, ...
        'live_trajectory_video_enabled', false)
    live_rollout_video_writer('write', plot_state.figure, constraint_cfg);
    sample_indices = struct_field_default(constraint_cfg, ...
        'live_trajectory_plot_sample_indices', sample_idx);
    requested_position = find(sample_indices == sample_idx, 1, 'first');
    is_group_end = mod(requested_position, group_size) == 0 || ...
        requested_position == numel(sample_indices);
    is_group_end = struct_field_default(plot_state, 'video_group_end', is_group_end);
    is_final_video_frame = step_idx == n_steps && is_group_end;
    if is_final_video_frame
        live_rollout_video_writer('finish', plot_state.figure, ...
            constraint_cfg);
    end
end

if step_idx == n_steps && struct_field_default(constraint_cfg, ...
        'live_trajectory_save_enabled', false)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    output_path = fullfile(output_dir, sprintf( ...
        'FirstLevel_RK4_Generation_Sample_%03d.png', sample_idx));
    exportgraphics(plot_state.figure, output_path, 'Resolution', 180);
    disp(['Saved live RK4 trajectory plot: ', output_path]);
    if struct_field_default(constraint_cfg, ...
            'live_trajectory_close_after_save', false)
        close(plot_state.figure);
        plot_state.enabled = false;
    end
end
end

function xy = state_to_xy(x, point_maps)
xy = zeros(numel(point_maps), 2);
for point_idx = 1:numel(point_maps)
    p = point_maps(point_idx).M * x(:) + point_maps(point_idx).o;
    xy(point_idx, :) = p(:)';
end
end
