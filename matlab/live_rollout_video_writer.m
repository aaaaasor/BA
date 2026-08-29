function [active, output_path] = live_rollout_video_writer( ...
    action, fig, constraint_cfg)
%LIVE_ROLLOUT_VIDEO_WRITER Persist one MP4 writer across segment rollouts.
% Video output must never abort numerical rollout. Invalid/closed figures
% and encoder failures stop recording with a warning, while the caller
% continues normally.

persistent video_writer saved_output_path
active = ~isempty(video_writer);
output_path = saved_output_path;
action = lower(string(action));

switch action
case "start"
    close_writer();
    if ~valid_figure(fig)
        warning('Live RK4 video skipped: figure handle is no longer valid.');
        return;
    end
    output_path = char(string(struct_field_default(constraint_cfg, ...
        'live_trajectory_video_output_path', fullfile( ...
        fileparts(mfilename('fullpath')), 'outputs', ...
        'Live_RK4_Trajectory.mp4'))));
    output_dir = fileparts(output_path);
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    try
        video_writer = VideoWriter(output_path, 'MPEG-4');
        video_writer.FrameRate = struct_field_default(constraint_cfg, ...
            'live_trajectory_video_frame_rate', 20);
        video_writer.Quality = struct_field_default(constraint_cfg, ...
            'live_trajectory_video_quality', 95);
        open(video_writer);
        saved_output_path = output_path;
        writeVideo(video_writer, getframe(fig));
        active = true;
        fprintf('Recording live RK4 animation: %s\n', output_path);
    catch video_error
        close_writer();
        warning('Live RK4 video recording disabled: %s', ...
            video_error.message);
    end

case "write"
    if isempty(video_writer)
        return;
    end
    if ~valid_figure(fig)
        warning(['Live RK4 video recording stopped because the figure ', ...
            'was closed or became invalid.']);
        close_writer();
        return;
    end
    try
        writeVideo(video_writer, getframe(fig));
        active = true;
    catch video_error
        close_writer();
        warning('Live RK4 video recording stopped: %s', ...
            video_error.message);
    end

case "finish"
    output_path = saved_output_path;
    was_active = ~isempty(video_writer);
    close_writer();
    active = false;
    if was_active && ~isempty(output_path)
        disp(['Saved live RK4 animation: ', output_path]);
    end

otherwise
    error('Unknown live video writer action "%s".', action);
end

    function close_writer()
        if ~isempty(video_writer)
            try
                close(video_writer);
            catch
            end
        end
        video_writer = [];
        saved_output_path = '';
        active = false;
    end
end

function tf = valid_figure(fig)
tf = ~isempty(fig) && isscalar(fig) && isgraphics(fig, 'figure');
end
