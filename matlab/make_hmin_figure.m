function T = make_hmin_figure(metrics_path, run_name, out_dir, file_prefix)
%MAKE_HMIN_FIGURE Per-trajectory minimum barrier value, in the same format as
% the FM / SafeFlow archives (Racing_FM_NN_hmin.emf, Racing_SafeFlow_NN_hmin.emf).
%
%   make_hmin_figure(metrics_mat, 'Racing Baseline', out_dir, 'Racing_Baseline')
%
% Reads safety_details from a *_Metrics_Variance_U.mat produced by
% evaluate_and_save_safeflow_metrics.  Nothing is recomputed: track_min_h and
% obstacle_min_h are the values the reported Safety metric itself used, i.e.
% the true track/obstacle geometry with no inward margin or inflation.
%
% Two panels: full range, and a zoom around zero (the full range is useless on
% its own once a trajectory has left the track by O(1e5)).

if nargin < 4 || isempty(file_prefix), file_prefix = run_name; end
S = load(metrics_path, 'safety_details', 'seed_bundle', 'safe_flow_metrics');
d = S.safety_details;
n = numel(d.track_min_h);
idx = (1:n)';

seed = nan(n, 1);
if isfield(S, 'seed_bundle') && isfield(S.seed_bundle, 'first_level_seeds')
    s1 = double(S.seed_bundle.first_level_seeds);
    if numel(s1) == n, seed = s1(:); end
end

T = table(idx, seed, d.safe_mask(:), d.obstacle_min_h(:), d.track_min_h(:), ...
    'VariableNames', {'trajectory_index', 'seed', 'safe', ...
    'min_obstacle_h', 'min_boundary_h'});

safety_pct = 100 * mean(d.safe_mask);
point_pct  = 100 * mean(d.point_safe_mask(:));

if ~exist(out_dir, 'dir'), mkdir(out_dir); end
writetable(T, fullfile(out_dir, [file_prefix '_hmin.csv']));

f = figure('Color', 'w', 'Position', [80 80 1180 400], 'Visible', 'off');

for panel = 1:2
    subplot(1, 2, panel); hold on;
    plot(T.trajectory_index, T.min_obstacle_h, '.-', 'MarkerSize', 9, ...
        'DisplayName', 'Obstacle min h');
    plot(T.trajectory_index, T.min_boundary_h, '.-', 'MarkerSize', 9, ...
        'DisplayName', 'Track boundary min h');
    yline(0, 'k--', 'HandleVisibility', 'off');
    grid on;
    xlabel('Trajectory index');
    ylabel('min h (true geometry, no margin)');
    if panel == 1
        legend('Location', 'best');
        title('Full range');
    else
        v = [T.min_obstacle_h; T.min_boundary_h];
        lo = prctile(v, 5); hi = prctile(v, 95);
        pad = max(0.15 * (hi - lo), 1e-4);
        ylim([lo - pad, hi + pad]);
        title(sprintf('Zoom  [%.3g, %.3g]', lo - pad, hi + pad));
    end
end
sgtitle(sprintf(['%s   Per-Trajectory Safety Margin   ' ...
    '(Safety %.0f%% of trajectories, %.2f%% of points)'], ...
    run_name, safety_pct, point_pct));

export_graphics_compat(f, fullfile(out_dir, [file_prefix '_hmin.emf']));
close(f);

fprintf('%-22s  safety %6.2f%% traj  %6.2f%% points | track min %.4g  obst min %.4g\n', ...
    run_name, safety_pct, point_pct, min(T.min_boundary_h), min(T.min_obstacle_h));
fprintf('  wrote %s\n', fullfile(out_dir, [file_prefix '_hmin.emf']));
fprintf('  wrote %s\n', fullfile(out_dir, [file_prefix '_hmin.csv']));
end
