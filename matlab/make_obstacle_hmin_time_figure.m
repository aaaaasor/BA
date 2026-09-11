function stats = make_obstacle_hmin_time_figure(rollout_mat, run_name, out_emf)
%MAKE_OBSTACLE_HMIN_TIME_FIGURE  Third-level obstacle h_min over generation time.
%
% Reproduces the figure main_demo.m writes as
% ThirdLevel_Obstacle_CBF_Value_Over_Time.emf, archived at the outputs root as
% "<racing>baseline hmin.emf": one blue line per rollout sample, obstacle
% h_min evaluated at the saved rollout nodes, plus the h = 0 boundary.

S = load(rollout_mat, 'third_traj_path_10d', 'third_rollout_times', ...
    'saved_third_segment_variance_constraint');
P = S.third_traj_path_10d;
t = S.third_rollout_times;
c = S.saved_third_segment_variance_constraint;
assert(all(struct_field_default(c, 'obstacle_constraint_inflations', 0) == 0), ...
    'This figure must use uninflated (true) obstacle geometry.');

H = evaluate_rollout_obstacle_hmin(P, c);      % (n_times x n_samples)
assert(any(isfinite(H(:))), 'obstacle h_min came back all-Inf.');

f = figure('Name', 'Third-level obstacle CBF value over time', 'Color', 'w', ...
    'WindowStyle', 'normal', 'Units', 'normalized', ...
    'Position', [0.17, 0.18, 0.60, 0.48], 'Visible', 'off');
hold on;
for s = 1:size(H, 2)
    plot(t, H(:, s), 'Color', [0.00, 0.45, 0.74], 'LineWidth', 0.8, ...
        'HandleVisibility', 'off');
end
plot(nan, nan, '-', 'Color', [0.00, 0.45, 0.74], 'LineWidth', 1.2, ...
    'DisplayName', 'minimum obstacle CBF value (rollout nodes)');
yline(0, '--', 'Color', [0.85, 0.20, 0.20], 'LineWidth', 1.2, ...
    'DisplayName', 'h = 0 safety boundary');
grid on; xlabel('t'); ylabel('h_{min}(t)');
title('Third-level obstacle CBF value over time');
legend('Location', 'best');
export_graphics_compat(f, out_emf);
close(f);

He = H(end, :);
stats = struct('n_samples', size(H, 2), 'terminal_min', min(He), ...
    'terminal_median', median(He), 'terminal_negative', sum(He < 0), ...
    'ever_negative', sum(any(H < 0, 1)), 'global_min', min(H(:)));
fprintf(['%-20s n=%4d | terminal min %-12.4g median %-11.4g #<0 %4d | ', ...
    'ever<0 %4d | global min %.4g\n'], run_name, stats.n_samples, ...
    stats.terminal_min, stats.terminal_median, stats.terminal_negative, ...
    stats.ever_negative, stats.global_min);
fprintf('  wrote %s\n', out_emf);
end
