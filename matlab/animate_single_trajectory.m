% Animate one 10D trajectory-space ODE rollout.
function output_path = animate_single_trajectory(cfg, rollout_times, ...
    traj_path_single, source_points)
%% Output Path
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
output_enabled = ~isfield(cfg, 'output') || ...
    ~isfield(cfg.output, 'enabled') || cfg.output.enabled;
if output_enabled && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_path = fullfile(output_dir, 'single_trajectory_rollout.gif');
if ~output_enabled
    output_path = "";
end

%% Plot Limits
all_rollout_points = reshape(traj_path_single', 2, [])';
all_points = [all_rollout_points; source_points];
x_pad = 0.08 * (max(all_points(:, 1)) - min(all_points(:, 1)));
y_pad = 0.08 * (max(all_points(:, 2)) - min(all_points(:, 2)));
if x_pad <= 0
    x_pad = 0.1;
end
if y_pad <= 0
    y_pad = 0.1;
end
x_limits = [min(all_points(:, 1)) - x_pad, max(all_points(:, 1)) + x_pad];
y_limits = [min(all_points(:, 2)) - y_pad, max(all_points(:, 2)) + y_pad];

%% Figure Layout
fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.25, 0.20, 0.45, 0.55]);
movegui(fig, 'center');
ax = axes(fig);
hold(ax, 'on');
h_source = plot(ax, source_points(:, 1), source_points(:, 2), '--', ...
    'Color', [0.45, 0.45, 0.45], 'LineWidth', 1.1, ...
    'DisplayName', 'source');
h_rollout = plot(ax, nan, nan, '.-', ...
    'Color', [0.10, 0.35, 0.90], 'LineWidth', 1.8, ...
    'MarkerSize', 14, 'DisplayName', 'rollout');
h_anchor = plot(ax, nan, nan, 'ks', ...
    'LineWidth', 1.3, 'MarkerSize', 7, 'MarkerFaceColor', 'w', ...
    'DisplayName', 'first-level anchors');
grid(ax, 'on');
axis(ax, 'equal');
xlim(ax, x_limits);
ylim(ax, y_limits);
xlabel(ax, 'x');
ylabel(ax, 'y');
legend(ax, [h_source, h_rollout, h_anchor], 'Location', 'best', ...
    'AutoUpdate', 'off');

%% GIF Frames
frame_idx_set = 1:cfg.animation.frame_stride:numel(rollout_times);
if frame_idx_set(end) ~= numel(rollout_times)
    frame_idx_set = [frame_idx_set, numel(rollout_times)];
end

for frame_nr = 1:numel(frame_idx_set)
    time_idx = frame_idx_set(frame_nr);
    s_now = rollout_times(time_idx);
    current_curve = reshape(traj_path_single(time_idx, :), 2, [])';
    anchor_idx = 1:(cfg.segment_points_per_segment - 1): ...
        size(current_curve, 1);

    set(h_rollout, 'XData', current_curve(:, 1), ...
        'YData', current_curve(:, 2));
    set(h_anchor, 'XData', current_curve(anchor_idx, 1), ...
        'YData', current_curve(anchor_idx, 2));
    title(ax, sprintf('Single 10D ODE Rollout in Standardized Space, s = %.2f', ...
        s_now));
    drawnow;

    frame = getframe(fig);
    [image_rgb, ~] = frame2im(frame);
    [image_indexed, color_map] = rgb2ind(image_rgb, 256);
    if output_enabled
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
