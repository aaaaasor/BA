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
title(plot_state.axes, sprintf('Sample %d: RK4 step %d/%d, t = %.4f', ...
    sample_idx, step_idx, n_steps, t_now));
drawnow;
delay = struct_field_default(constraint_cfg, ...
    'live_trajectory_plot_delay', 0.0);
if delay > 0
    pause(delay);
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
