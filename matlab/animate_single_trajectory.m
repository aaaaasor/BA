% Animate one trajectory-space ODE roll out.
function output_path = animate_single_trajectory(cfg, rollout_times, ...
    traj_path_single, source_points)
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
if cfg.output.enabled && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_filename = struct_field_default(cfg.animation, ...
    'output_filename', 'single_trajectory_rollout.gif');
output_path = fullfile(output_dir, output_filename);

%% Plot Limits
feature_dim = size(source_points, 2);
all_rollout_states = reshape(traj_path_single', feature_dim, [])';
all_rollout_points = all_rollout_states(:, 1:2);
use_segment_plot = isfield(cfg.animation, 'segment_plot_path') && ...
    ~isempty(cfg.animation.segment_plot_path);
if use_segment_plot
    % 展开所有帧的所有segment点，用于计算轴限
    seg_path = cfg.animation.segment_plot_path;
    n_pts = cfg.animation.segment_plot_points_per_segment;
    seg_fdim = size(seg_path, 3) / n_pts;
    seg_xy = zeros(size(seg_path, 1) * size(seg_path, 2) * n_pts, 2);
    write_idx = 1;
    for time_idx = 1:size(seg_path, 1)
        seg_rows = squeeze(seg_path(time_idx, :, :));
        if size(seg_rows, 1) == 1; seg_rows = reshape(seg_rows, 1, []); end
        for seg_idx = 1:size(seg_rows, 1)
            seg_curve = reshape(seg_rows(seg_idx, :), seg_fdim, [])';
            next_idx = write_idx + n_pts - 1;
            seg_xy(write_idx:next_idx, :) = seg_curve(:, 1:2);
            write_idx = next_idx + 1;
        end
    end
    all_rollout_points = [all_rollout_points; seg_xy];
end
source_points_xy = source_points(:, 1:2);
all_points = [all_rollout_points; source_points_xy];
x_pad = 0.08 * (max(all_points(:, 1)) - min(all_points(:, 1)));
y_pad = 0.08 * (max(all_points(:, 2)) - min(all_points(:, 2)));
x_limits = [min(all_points(:, 1)) - x_pad, max(all_points(:, 1)) + x_pad];
y_limits = [min(all_points(:, 2)) - y_pad, max(all_points(:, 2)) + y_pad];

%% Figure Layout
fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.18, 0.18, 0.58, 0.58]);
movegui(fig, 'center');
ax = axes(fig);
set(ax, 'Position', [0.10, 0.12, 0.62, 0.76]);
hold(ax, 'on');
source_plot_xy = source_points_xy;
break_source_segments = struct_field_default(cfg.animation, ...
    'break_source_segments', true);
if use_segment_plot && break_source_segments
    n_source_segments = cfg.animation.segment_plot_count;
    n_source_points = cfg.animation.segment_plot_points_per_segment;
    if n_source_segments > 1 && ...
            size(source_points_xy, 1) == n_source_segments * n_source_points
        % Each source segment is sampled independently.  Insert NaN rows so
        % MATLAB does not draw artificial links between adjacent segments.
        source_plot_xy = nan(size(source_points_xy, 1) + ...
            n_source_segments - 1, 2);
        for segment_idx = 1:n_source_segments
            input_start = (segment_idx - 1) * n_source_points + 1;
            input_end = input_start + n_source_points - 1;
            output_start = (segment_idx - 1) * (n_source_points + 1) + 1;
            output_end = output_start + n_source_points - 1;
            source_plot_xy(output_start:output_end, :) = ...
                source_points_xy(input_start:input_end, :);
        end
    end
end
h_source = plot(ax, source_plot_xy(:, 1), source_plot_xy(:, 2), '--', ...
    'Color', [0.45, 0.45, 0.45], 'LineWidth', 1.1, 'DisplayName', 'source');
h_rollout = plot(ax, nan, nan, '.-', ...
    'Color', [0.10, 0.35, 0.90], 'LineWidth', 1.8, ...
    'MarkerSize', 14, 'DisplayName', 'rollout');
if use_segment_plot
    n_segments = cfg.animation.segment_plot_count;
    delete(h_rollout);
    h_rollout = gobjects(n_segments, 1);
    for segment_idx = 1:n_segments
        if segment_idx == 1
            h_rollout(segment_idx) = plot(ax, nan, nan, '.-', ...
                'Color', [0.10, 0.35, 0.90], 'LineWidth', 1.8, ...
                'MarkerSize', 14, 'DisplayName', 'rollout');
        else
            h_rollout(segment_idx) = plot(ax, nan, nan, '.-', ...
                'Color', [0.10, 0.35, 0.90], 'LineWidth', 1.8, ...
                'MarkerSize', 14, 'HandleVisibility', 'off');
        end
    end
end
grid(ax, 'on');
axis(ax, 'equal');
xlim(ax, x_limits);
ylim(ax, y_limits);
xlabel(ax, 'x');
ylabel(ax, 'y');
rollout_legend_handle = h_rollout(1);
source_legend_handle = h_source(1);
legend(ax, [source_legend_handle, rollout_legend_handle], ...
    'Location', struct_field_default(cfg.animation, 'legend_location', 'best'), 'AutoUpdate', 'off');

%% GIF Frames
frame_idx_set = 1:cfg.animation.frame_stride:numel(rollout_times);
if frame_idx_set(end) ~= numel(rollout_times)
    frame_idx_set = [frame_idx_set, numel(rollout_times)];
end
% 某些关键状态（例如第二层端点 snap 后、随后的尾段 FM 步）不能被
% frame_stride 跳过。调用方可通过 forced_frame_indices 显式保留它们。
forced_frame_indices = struct_field_default(cfg.animation, ...
    'forced_frame_indices', zeros(1, 0));
forced_frame_indices = forced_frame_indices(:)';
forced_frame_indices = forced_frame_indices( ...
    forced_frame_indices >= 1 & forced_frame_indices <= numel(rollout_times));
frame_idx_set = unique([frame_idx_set, forced_frame_indices], 'sorted');
for frame_nr = 1:numel(frame_idx_set)
    time_idx = frame_idx_set(frame_nr);
    s_now = rollout_times(time_idx);
    current_curve = reshape(traj_path_single(time_idx, :), feature_dim, [])';
    if use_segment_plot
        segment_rows = squeeze(cfg.animation.segment_plot_path(time_idx, :, :));
        if size(segment_rows, 1) == 1
            segment_rows = reshape(segment_rows, 1, []);
        end
        segment_feature_dim = size(segment_rows, 2) / ...
            cfg.animation.segment_plot_points_per_segment;
        for segment_idx = 1:numel(h_rollout)
            segment_curve = reshape(segment_rows(segment_idx, :), ...
                segment_feature_dim, [])';
            set(h_rollout(segment_idx), 'XData', segment_curve(:, 1), ...
                'YData', segment_curve(:, 2));
        end
    else
        set(h_rollout, 'XData', current_curve(:, 1), ...
            'YData', current_curve(:, 2));
    end
    state_dim = size(current_curve, 1) * size(current_curve, 2);
    title(ax, sprintf('Single %dD ODE Rollout in %s, s = %.3f', ...
        state_dim, struct_field_default(cfg.animation, 'space_label', 'trajectory space'), s_now));
    drawnow;
    frame = getframe(fig);
    [image_rgb, ~] = frame2im(frame);
    [image_indexed, color_map] = rgb2ind(image_rgb, 256);
    if cfg.output.enabled
        if frame_nr == 1
            imwrite(image_indexed, color_map, output_path, 'gif', ...
                'LoopCount', inf, 'DelayTime', cfg.animation.delay_time);
        else
            imwrite(image_indexed, color_map, output_path, 'gif', ...
                'WriteMode', 'append', 'DelayTime', cfg.animation.delay_time);
        end
    end
end

end
