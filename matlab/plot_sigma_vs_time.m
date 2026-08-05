function plot_sigma_vs_time(cfg, traj_times, uncertainty_values, level_label, ...
    ~, signal_std_vec)
if nargin < 6 || isempty(signal_std_vec)
    error('plot_sigma_vs_time:MissingPriorScale', ...
        'signal_std_vec is required for dimensionless OOD evaluation.');
end

n_outputs = size(uncertainty_values, 3);
signal_std_vec = signal_std_vec(:);
if numel(signal_std_vec) ~= n_outputs
    error('plot_sigma_vs_time:PriorScaleSizeMismatch', ...
        'Expected %d GP signal standard deviations, received %d.', ...
        n_outputs, numel(signal_std_vec));
end
prior_variance = reshape(max(signal_std_vec .^ 2, eps), 1, 1, []);
normalized_variance = uncertainty_values ./ prior_variance;
sigma_by_sample = sqrt(aggregate_variance_values( ...
    normalized_variance) ./ n_outputs);

ood_threshold = struct_field_default(cfg.variance_constraint, ...
    'ood_normalized_sigma_threshold', 0.8);
ood_quantile = struct_field_default(cfg.variance_constraint, ...
    'ood_quantile', 0.95);
if ~(isscalar(ood_threshold) && isfinite(ood_threshold) && ...
        ood_threshold > 0)
    error('plot_sigma_vs_time:InvalidOODThreshold', ...
        'OOD normalized-sigma threshold must be a positive finite scalar.');
end
if ~(isscalar(ood_quantile) && isfinite(ood_quantile) && ...
        ood_quantile > 0 && ood_quantile <= 1)
    error('plot_sigma_vs_time:InvalidOODQuantile', ...
        'OOD quantile must lie in (0, 1].');
end
ood_score = prctile(sigma_by_sample(:), 100 * ood_quantile);
is_ood = ood_score > ood_threshold;
fprintf(['%s normalized OOD score (q=%.3g): %.4f; threshold %.4f; ', ...
    'classification: %s\n'], level_label, ood_quantile, ood_score, ...
    ood_threshold, string(classification_label(is_ood)));

fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.14, 0.18, 0.60, 0.46]);
movegui(fig, 'center');
hold on;
for sample_idx = 1:size(sigma_by_sample, 2)
    plot(traj_times, sigma_by_sample(:, sample_idx), ...
        'Color', [0.20, 0.45, 0.85], 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
end
plot(nan, nan, '-', 'Color', [0.20, 0.45, 0.85], ...
    'LineWidth', 1.2, 'DisplayName', ...
    'normalized RMS GP uncertainty');
grid on;
xlabel('s');
ylabel('sqrt(mean_i(\sigma_i^2 / \Sigma_{F,i}^2))');
title(sprintf('%s: normalized GP uncertainty', level_label));
legend('Location', 'best', 'Interpreter', 'none');

if cfg.output.enabled
    this_file = mfilename('fullpath');
    output_dir = fullfile(fileparts(this_file), 'outputs');
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    if strcmpi(struct_field_default(cfg, 'scenario', ''), 'racing') && ...
            strcmpi(level_label, 'third-level')
        output_filename = 'Racing_ThirdLevel_Variance.emf';
    else
        output_filename = ['trajectory_gp_sigma_vs_time_', ...
            strrep(level_label, ' ', '_'), '_matlab.emf'];
    end
    output_path = fullfile(output_dir, output_filename);
    export_graphics_compat(fig, output_path);
end
end

function label = classification_label(is_ood)
if is_ood
    label = 'OOD';
else
    label = 'in-distribution';
end
end
