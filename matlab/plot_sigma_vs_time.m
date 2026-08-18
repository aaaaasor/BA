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
sigma_by_sample = aggregate_variance_values(normalized_variance);
% Unnormalized counterpart of the displayed OOD score.  This is the summed
% predictive variance computed directly from the GP variances:
% sum_i(sigma_i^2), without division by the prior Sigma_F,i^2.
raw_sigma_by_sample = aggregate_variance_values(uncertainty_values);
[terminal_beta_cap, terminal_cap_label] = terminal_variance_cap( ...
    cfg, traj_times, raw_sigma_by_sample, level_label);

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
    'normalized summed GP variance');
grid on;
xlabel('s');
ylabel('\Sigma_i(\sigma_i^2 / \Sigma_{F,i}^2)');
title(sprintf('%s: normalized GP uncertainty', level_label));
legend('Location', 'best', 'Interpreter', 'none');

fig_raw = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.16, 0.16, 0.60, 0.46]);
movegui(fig_raw, 'center');
hold on;
for sample_idx = 1:size(raw_sigma_by_sample, 2)
    plot(traj_times, raw_sigma_by_sample(:, sample_idx), ...
        'Color', [0.20, 0.45, 0.85], 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
end
plot(nan, nan, '-', 'Color', [0.20, 0.45, 0.85], ...
    'LineWidth', 1.2, 'DisplayName', 'summed GP variance');
if ~isempty(terminal_beta_cap)
    plot(traj_times, terminal_beta_cap, '--', ...
        'Color', [0.85, 0.20, 0.15], 'LineWidth', 2.2, ...
        'DisplayName', terminal_cap_label);
    excess = raw_sigma_by_sample - terminal_beta_cap(:);
    fprintf(['%s terminal variance PTCBF cap: max(raw beta-cap)=%.6g; ', ...
        'violating plotted points=%d / %d\n'], level_label, ...
        max(excess(:)), nnz(excess > 1e-8), numel(excess));
end
grid on;
xlabel('s');
ylabel('\Sigma_i(\sigma_i^2)');
title(sprintf('%s: raw GP uncertainty vs terminal PTCBF cap', level_label));
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
    raw_output_filename = ['trajectory_gp_sigma_raw_vs_time_', ...
        strrep(level_label, ' ', '_'), '_matlab.emf'];
    raw_output_path = fullfile(output_dir, raw_output_filename);
    export_graphics_compat(fig_raw, raw_output_path);
    raw_png_path = fullfile(output_dir, strrep( ...
        raw_output_filename, '.emf', '.png'));
    exportgraphics(fig_raw, raw_png_path, 'Resolution', 200);
    disp(['Saved raw GP uncertainty trace: ', raw_output_path]);
    disp(['Saved raw GP uncertainty PNG: ', raw_png_path]);
end
end

function [beta_cap, label] = terminal_variance_cap( ...
    cfg, traj_times, raw_beta, level_label)
beta_cap = [];
label = '';
level_prefix = level_config_prefix(level_label);
if level_prefix == ""
    return;
end
vcfg = cfg.variance_constraint;
enabled = struct_field_default(vcfg, ...
    [char(level_prefix), '_ptcbf_enabled'], false);
if ~enabled
    return;
end
beta_final = struct_field_required(vcfg, ...
    [char(level_prefix), '_terminal_variance_beta_final']);
margin = struct_field_default(vcfg, ...
    [char(level_prefix), '_terminal_variance_ptzf_initial_margin'], 1e-6);
gamma = struct_field_required(vcfg, ...
    [char(level_prefix), '_terminal_variance_ptzf_gamma']);
terminal_time = struct_field_default(vcfg, ...
    [char(level_prefix), '_terminal_variance_ptzf_terminal_time'], ...
    struct_field_default(vcfg, 'terminal_variance_ptzf_terminal_time', ...
    cfg.rollout_t_max));

% Match rk4_rollout exactly: one common initial envelope is constructed
% from the largest initial raw beta over every sample in this rollout.
hbar0 = max(max(raw_beta(1, :)) - beta_final, 0.0) + margin;
t_eff = traj_times(:) ./ terminal_time;
hbar = zeros(size(t_eff));
before_terminal = t_eff < 1.0;
remaining = 1.0 - t_eff(before_terminal);
shape = t_eff(before_terminal) ./ remaining;
hbar(before_terminal) = hbar0 .* exp(-gamma .* shape);
beta_cap = beta_final + hbar;
label = sprintf('terminal PTCBF cap (raw beta, final %.3g)', beta_final);
end

function prefix = level_config_prefix(level_label)
switch lower(string(level_label))
    case "first-level"
        prefix = "first_level";
    case "second-level"
        prefix = "second_level";
    case "third-level"
        prefix = "third_level";
    otherwise
        prefix = "";
end
end

function label = classification_label(is_ood)
if is_ood
    label = 'OOD';
else
    label = 'in-distribution';
end
end
