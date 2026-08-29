% Plot one GP predictive-mean panel per output dimension.
function plot_gp_mean_per_output(cfg, rollout_times, rollout_mean, level_label)
n_outputs = size(rollout_mean, 3);
n_samples = size(rollout_mean, 2);
n_columns = min(4, n_outputs);
n_rows = ceil(n_outputs / n_columns);
zero_tolerance = 1e-8;
output_max_abs = zeros(n_outputs, 1);
zero_curve_masks = false(n_samples, n_outputs);

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.03, 0.05, 0.94, 0.86]);
movegui(fig, 'center');
tiledlayout(n_rows, n_columns, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
for output_idx = 1:n_outputs
    ax = nexttile;
    hold(ax, 'on');
    mean_now = rollout_mean(:, :, output_idx);
    output_max_abs(output_idx) = max(abs(mean_now), [], 'all');
    % One blue curve is one rollout sample.  Mark a curve as zero only
    % when that sample remains numerically zero at every saved time.
    zero_curve_masks(:, output_idx) = ...
        max(abs(mean_now), [], 1)' <= zero_tolerance;
    for sample_idx = 1:n_samples
        plot(ax, rollout_times, mean_now(:, sample_idx), ...
            'Color', [0.55, 0.72, 0.92], 'LineWidth', 0.6, ...
            'HandleVisibility', 'off');
    end
    % Mean absolute prediction cannot be cancelled by positive and negative
    % rollout samples, so a zero curve really indicates a zero GP output.
    plot(ax, rollout_times, mean(abs(mean_now), 2), 'k-', ...
        'LineWidth', 1.5, 'DisplayName', 'mean absolute prediction');
    yline(ax, 0.0, ':', 'Color', [0.45, 0.45, 0.45], ...
        'HandleVisibility', 'off');
    grid(ax, 'on');
    title(ax, sprintf('output %d: max|mu|=%.3g, zero curves=%d/%d', ...
        output_idx, output_max_abs(output_idx), ...
        sum(zero_curve_masks(:, output_idx)), n_samples), ...
        'Interpreter', 'none');
    if mod(output_idx - 1, n_columns) == 0
        ylabel(ax, '\mu_i(t,x(t))');
    end
    if output_idx > n_outputs - n_columns
        xlabel(ax, 't');
    end
    if output_idx == 1
        legend(ax, 'Location', 'best', 'Interpreter', 'none');
    end
end
sgtitle(sprintf('%s GP predictive mean along rollout states', level_label), ...
    'Interpreter', 'none');

zero_curve_count = sum(zero_curve_masks, 'all');
if zero_curve_count == 0
    fprintf(['%s GP mean zero-curve check: none of %d blue curves x ', ...
        '%d outputs is identically zero over time (tolerance %.1e).\n'], ...
        level_label, n_samples, n_outputs, zero_tolerance);
else
    fprintf(['WARNING: %s GP mean contains %d zero blue curve(s) ', ...
        '(tolerance %.1e):\n'], level_label, zero_curve_count, ...
        zero_tolerance);
    for output_idx = 1:n_outputs
        zero_sample_indices = find(zero_curve_masks(:, output_idx));
        if ~isempty(zero_sample_indices)
            fprintf('  output %d, rollout sample(s): %s\n', output_idx, ...
                mat2str(zero_sample_indices(:)'));
        end
    end
end
fprintf('%s GP mean max-abs value per output: %s\n', level_label, ...
    mat2str(output_max_abs(:)', 5));

if cfg.output.enabled
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    if strcmpi(struct_field_default(cfg, 'scenario', ''), 'racing') && ...
            strcmpi(level_label, 'third-level')
        output_name = 'Racing_ThirdLevel_GP_Mean_Per_Output.emf';
    else
        output_name = ['trajectory_gp_mean_per_output_', ...
            strrep(level_label, ' ', '_'), '_matlab.emf'];
    end
    output_path = fullfile(output_dir, output_name);
    export_graphics_compat(fig, output_path);
    png_path = fullfile(output_dir, strrep(output_name, '.emf', '.png'));
    exportgraphics(fig, png_path, 'Resolution', 200);
    disp(['Saved GP predictive-mean panels: ', output_path]);
    disp(['Saved GP predictive-mean PNG: ', png_path]);
end
end
