function plot_training_trajectory_utilization(cfg, model_collection, level_label)
if ~isfield(model_collection, 'training_trajectory_utilization') || ...
        isempty(model_collection.training_trajectory_utilization)
    disp(['Skipping ', level_label, ...
        ' training trajectory utilization plot: statistics unavailable.']);
    return;
end

utilization = model_collection.training_trajectory_utilization(:);
trajectory_idx = (1:numel(utilization))';
moving_window = struct_field_default(cfg, ...
    'training_utilization_moving_average_window', 25);
moving_window = max(1, round(moving_window));
moving_window = min(moving_window, numel(utilization));
moving_average = movmean(utilization, moving_window, 'Endpoints', 'shrink');
plot_label = level_label;
file_label = strrep(level_label, ' ', '_');
if strcmp(level_label, 'first-level') && ...
        isfield(cfg, 'first_level_use_tangent_features')
    if cfg.first_level_use_tangent_features
        feature_label = 'tangent';
    else
        feature_label = 'position-only';
    end
    plot_label = [level_label, ' (', feature_label, ')'];
    file_label = [file_label, '_', feature_label];
end

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.18, 0.20, 0.58, 0.42]);
movegui(fig, 'center');
bar(trajectory_idx, utilization, 0.82, ...
    'FaceColor', [0.20, 0.45, 0.85], 'EdgeColor', 'none', ...
    'DisplayName', 'trajectory');
hold on;
plot(trajectory_idx, utilization, '-', 'Color', [0.08, 0.18, 0.34], ...
    'LineWidth', 1.0, 'DisplayName', 'trend');
plot(trajectory_idx, moving_average, '-', 'Color', [0.95, 0.55, 0.10], ...
    'LineWidth', 2.2, 'DisplayName', ...
    ['moving avg (', num2str(moving_window), ')']);

overall_utilization = sum(model_collection.n_added_per_trajectory) ./ ...
    max(sum(model_collection.n_added_per_trajectory + ...
    model_collection.n_skipped_per_trajectory), 1);
yline(overall_utilization, '--', 'Color', [0.85, 0.20, 0.20], ...
    'LineWidth', 1.2, 'DisplayName', 'overall');

grid on;
xlabel('training trajectory index');
ylabel('utilization ratio');
y_margin = 0.05;
plot_values = [utilization; moving_average];
y_lower = max(min(plot_values) - y_margin, 0);
y_upper = min(max(plot_values) + y_margin, 1);
if y_upper - y_lower < 2 * y_margin
    y_lower = max(y_lower - y_margin, 0);
    y_upper = min(y_upper + y_margin, 1);
end
ylim([y_lower, y_upper]);
title([plot_label, ': retained training-pair ratio per trajectory'], ...
    'Interpreter', 'none');
legend('Location', 'best', 'Interpreter', 'none');
disp([plot_label, ' utilization moving-average window: ', ...
    num2str(moving_window)]);

if cfg.output.enabled
    this_file = mfilename('fullpath');
    output_dir = fullfile(fileparts(this_file), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    output_path = fullfile(output_dir, ...
        ['trajectory_gp_training_utilization_', ...
        file_label, '_matlab.emf']);
    export_graphics_compat(fig, output_path);
end
end
