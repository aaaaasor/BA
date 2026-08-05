function output_paths = plot_racing_obstacle_layout(cfg)
%PLOT_RACING_OBSTACLE_LAYOUT Draw the racing corridor and obstacle layout.

output_paths = strings(0, 1);
track_segment = struct_field_default(cfg, 'track_segment', []);
if isempty(track_segment) || ~isfield(cfg, 'obstacle') || ...
        ~struct_field_default(cfg.obstacle, 'enabled', false)
    return;
end

fig = figure('Name', 'Racing obstacle layout', 'Color', 'w', ...
    'WindowStyle', 'normal', 'Units', 'normalized', ...
    'Position', [0.16, 0.12, 0.58, 0.72]);
movegui(fig, 'center');
ax = axes(fig);
hold(ax, 'on');
set(fig, 'CurrentAxes', ax);
draw_track_segment(track_segment, 'HandleVisibility', 'off');
draw_obstacles(cfg.obstacle, 'HandleVisibility', 'off');
grid(ax, 'off');
axis(ax, 'equal');
xlim(ax, [0, 1]);
ylim(ax, [0, 1]);
xlabel(ax, 'x');
ylabel(ax, 'y');
title(ax, 'Racing Track with One Square Obstacle');

if cfg.output.enabled
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    emf_path = fullfile(output_dir, 'Racing_Obstacle_Layout.emf');
    export_graphics_compat(fig, emf_path);
    output_paths = string(emf_path);
    disp(['Saved racing obstacle layout: ', emf_path]);
end
end
