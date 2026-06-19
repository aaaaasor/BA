% Plot the main MATLAB summary figure:
function plot_results(cfg, target_points, source_points, reconstructed_points)
%% Output Path
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
output_enabled = ~isfield(cfg, 'output') || ...
    ~isfield(cfg.output, 'enabled') || cfg.output.enabled;
if output_enabled && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_path = fullfile(output_dir, 'gp_flow_matching_demo_matlab.emf');

%% Figure Layout
fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.08, 0.18, 0.84, 0.48]);
movegui(fig, 'center');

tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Shared Target/Rollout Axis Limits
target_xy = reshape(target_points(:, :, 1:2), [], 2);
rollout_xy = reshape(reconstructed_points(:, :, 1:2), [], 2);
comparison_xy = [target_xy; rollout_xy];
comparison_xy = comparison_xy(all(isfinite(comparison_xy), 2), :);
x_limits = [min(comparison_xy(:, 1)), max(comparison_xy(:, 1))];
y_limits = [min(comparison_xy(:, 2)), max(comparison_xy(:, 2))];
x_padding = 0.05 * max(diff(x_limits), eps);
y_padding = 0.05 * max(diff(y_limits), eps);
x_limits = x_limits + [-x_padding, x_padding];
y_limits = y_limits + [-y_padding, y_padding];

%% Target Trajectories
nexttile;
hold on;
n_training_curves = size(target_points, 2);
for idx = 1:n_training_curves
    training_curve = squeeze(target_points(:, idx, 1:2));
    plot(training_curve(:, 1), training_curve(:, 2), '.-', 'LineWidth', 0.8, ...
        'HandleVisibility', 'off');
end
grid on;
axis equal;
xlim(x_limits);
ylim(y_limits);
xlabel('x');
ylabel('y');
title(sprintf('Target Trajectory Data (2D, %d Points)', size(target_points, 1)));

%% Source Trajectories
nexttile;
hold on;
max_curves = min(size(source_points, 2), cfg.n_trajectories);
for idx = 1:max_curves
    source_curve = squeeze(source_points(:, idx, 1:2));
    plot(source_curve(:, 1), source_curve(:, 2), '--', 'LineWidth', 0.9);
end
grid on;
axis equal;
xlabel('x');
ylabel('y');
title('ODE Source Trajectories');

%% Rollout Trajectories
nexttile;
hold on;
max_curves = min(size(reconstructed_points, 2), cfg.n_trajectories);
fprintf('Plotting rollout sample curves: %d / %d\n', ...
    max_curves, size(reconstructed_points, 2));
rollout_colors = lines(max(max_curves, 1));
for idx = 1:max_curves
    reconstructed_curve = squeeze(reconstructed_points(:, idx, 1:2));
    if idx == 1
        plot(reconstructed_curve(:, 1), reconstructed_curve(:, 2), ...
            'Color', rollout_colors(idx, :), 'LineWidth', 1.1, ...
            'DisplayName', get_cfg_label(cfg, ...
            'sample_curve_label', 'Stage 2 sample curves'));
    else
        plot(reconstructed_curve(:, 1), reconstructed_curve(:, 2), ...
            'Color', rollout_colors(idx, :), 'LineWidth', 1.1, ...
            'HandleVisibility', 'off');
    end
end
if max_curves > 0
    anchor_idx = 1:(cfg.segment_points_per_segment - 1): ...
        size(reconstructed_points, 1);
    refined_idx = setdiff(1:size(reconstructed_points, 1), anchor_idx);
    for idx = 1:max_curves
        refined_curve = squeeze(reconstructed_points(refined_idx, idx, :));
        refined_curve = refined_curve(:, 1:2);
        if idx == 1
            plot(refined_curve(:, 1), refined_curve(:, 2), 'o', ...
                'Color', rollout_colors(idx, :), ...
                'MarkerFaceColor', rollout_colors(idx, :), ...
                'MarkerSize', 4, ...
                'LineStyle', 'none', ...
                'DisplayName', get_cfg_label(cfg, ...
                'generated_point_label', 'Stage 2 generated points'));
        else
            plot(refined_curve(:, 1), refined_curve(:, 2), 'o', ...
                'Color', rollout_colors(idx, :), ...
                'MarkerFaceColor', rollout_colors(idx, :), ...
                'MarkerSize', 4, ...
                'LineStyle', 'none', ...
                'HandleVisibility', 'off');
        end
    end
    if isfield(cfg, 'reference_points') && ~isempty(cfg.reference_points)
        reference_points = cfg.reference_points;
        n_reference_curves = min(size(reference_points, 2), max_curves);
        reference_label = 'Training reference curve';
        if isfield(cfg, 'reference_label')
            reference_label = cfg.reference_label;
        end
        if isfield(cfg, 'reference_curve_count')
            n_reference_curves = min(n_reference_curves, ...
                cfg.reference_curve_count);
        end
        for idx = 1:n_reference_curves
            reference_curve = squeeze(reference_points(:, idx, :));
            reference_curve = reference_curve(:, 1:2);
            if idx == 1
                plot(reference_curve(:, 1), reference_curve(:, 2), ...
                    'Color', [0.9, 0.0, 0.0], 'LineStyle', '-', ...
                    'LineWidth', 3.0, 'DisplayName', reference_label);
            else
                plot(reference_curve(:, 1), reference_curve(:, 2), ...
                    'Color', [0.9, 0.0, 0.0], 'LineStyle', '-', ...
                    'LineWidth', 2.0, 'HandleVisibility', 'off');
            end
        end
    end
    for idx = 1:max_curves
        anchor_curve = squeeze(reconstructed_points(anchor_idx, idx, :));
        anchor_curve = anchor_curve(:, 1:2);
        if idx == 1
            plot(anchor_curve(:, 1), anchor_curve(:, 2), 'ks-', ...
                'LineWidth', 1.5, 'MarkerSize', 7, ...
                'MarkerFaceColor', 'w', ...
                'DisplayName', get_cfg_label(cfg, ...
                'anchor_label', 'Stage 1 fixed anchors'));
        else
            plot(anchor_curve(:, 1), anchor_curve(:, 2), 'ks-', ...
                'LineWidth', 1.0, 'MarkerSize', 5, ...
                'MarkerFaceColor', 'w', ...
                'HandleVisibility', 'off');
        end
    end
end
grid on;
axis equal;
xlim(x_limits);
ylim(y_limits);
xlabel('x');
ylabel('y');
title(sprintf('ODE Rollout Trajectories (%d Points, %d Samples)', ...
    size(reconstructed_points, 1), max_curves));
legend_location = 'northeastoutside';
if isfield(cfg, 'legend_location') && ~isempty(cfg.legend_location)
    legend_location = cfg.legend_location;
end
legend_font_size = 8;
if isfield(cfg, 'legend_font_size') && ~isempty(cfg.legend_font_size)
    legend_font_size = cfg.legend_font_size;
end
lgd = legend('Location', legend_location);
lgd.FontSize = legend_font_size;
lgd.ItemTokenSize = [14, 8];

if output_enabled
    exportgraphics(fig, output_path, 'ContentType', 'vector');
end
end

function label = get_cfg_label(cfg, field_name, default_label)
label = default_label;
if isfield(cfg, field_name) && ~isempty(cfg.(field_name))
    label = cfg.(field_name);
end
end
