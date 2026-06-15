% Plot per-output LoG-GP predictive variance along an ODE rollout.
function plot_uncertainty_vs_time(cfg, traj_times, traj_gp_vars, plot_title, ...
    threshold)
if nargin < 4 || isempty(plot_title)
    plot_title = 'LoG-GP Predictive Variance Along ODE Rollout';
end
if nargin < 5 || isempty(threshold)
    threshold = cfg.variance_constraint.uncertainty_max;
end
%% Output Path
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
output_enabled = ~isfield(cfg, 'output') || ...
    ~isfield(cfg.output, 'enabled') || cfg.output.enabled;
if output_enabled && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_path = fullfile(output_dir, ...
    'trajectory_gp_uncertainty_vs_time_matlab.emf');

%% Figure Layout
n_outputs = size(traj_gp_vars, 3);
n_cols = 2;
n_rows = ceil(n_outputs / n_cols);
fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.08, 0.08, 0.84, 0.78]);
movegui(fig, 'center');
tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Per-Dimension Variance Curves
per_output_threshold = (threshold ^ 2) / max(n_outputs, 1);
for output_idx = 1:n_outputs
    nexttile;
    hold on;
    output_vars = traj_gp_vars(:, :, output_idx);
    for path_idx = 1:size(output_vars, 2)
        plot(traj_times, output_vars(:, path_idx), ...
            'Color', [0.55, 0.72, 0.92], 'LineWidth', 0.8, ...
            'HandleVisibility', 'off');
    end
    yline(per_output_threshold, '--', ...
        'Color', [0.85, 0.20, 0.20], 'LineWidth', 1.0, ...
        'DisplayName', 'variance threshold');
    set(gca, 'YScale', 'log');
    grid on;
    xlabel('s');
    ylabel(sprintf('\\sigma_%d^2', output_idx));
    title(sprintf('Output %d variance', output_idx));
    if output_idx == 1
        legend('Location', 'best');
    end
end
sgtitle(plot_title);
if output_enabled
    drawnow;
    if isgraphics(fig, 'figure')
        try
            exportgraphics(fig, output_path, 'ContentType', 'vector');
        catch export_error
            warning('Vector export failed: %s. Falling back to saveas.', ...
                export_error.message);
            saveas(fig, output_path);
        end
    else
        warning('Uncertainty figure handle is invalid. Skipping export.');
    end
end
end
