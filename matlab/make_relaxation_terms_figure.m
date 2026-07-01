function fig = make_relaxation_terms_figure(traj_times, h_bar, h_bar_dot, ...
    h_bar_ddot, plot_title)
fig = figure('Color', 'w', 'WindowStyle', 'normal', ...
    'Units', 'normalized', 'Position', [0.16, 0.16, 0.62, 0.62]);
movegui(fig, 'center');
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(traj_times, h_bar, 'Color', [0.20, 0.45, 0.85], ...
    'LineWidth', 1.8);
grid on;
xlabel('s');
ylabel('$\bar{h}(s)$', 'Interpreter', 'latex');
title('prescribed-time relaxation term');

nexttile;
plot(traj_times, h_bar_dot, 'Color', [0.85, 0.25, 0.15], ...
    'LineWidth', 1.8);
grid on;
xlabel('s');
ylabel('$\dot{\bar{h}}(s)$', 'Interpreter', 'latex');
title('first derivative');

nexttile;
plot(traj_times, h_bar_ddot, 'Color', [0.10, 0.55, 0.25], ...
    'LineWidth', 1.8);
grid on;
xlabel('s');
ylabel('$\ddot{\bar{h}}(s)$', 'Interpreter', 'latex');
title('second derivative');

sgtitle(plot_title, 'Interpreter', 'latex');
end
