% Generate PNG plots from the last tune_first_level_last.mat cache.
project_dir = 'C:\Users\JieLi\OneDrive - LBGruppe\Documents\New project\matlab';
cd(project_dir);
data = load(fullfile(project_dir, 'outputs', 'tune_first_level_last.mat'));
trace = data.first_rollout_diagnostics.hocbf;
tt = trace.trace_t(:);
smp = trace.trace_sample_idx(:);
ib = trace.trace_integral_bound(:);
tb = trace.trace_terminal_bound(:);
hres_raw = trace.trace_hocbf_constraint_residual(:);
if isfield(trace,'trace_hocbf_relaxed_constraint_residual') && ~isempty(trace.trace_hocbf_relaxed_constraint_residual)
    hres = trace.trace_hocbf_relaxed_constraint_residual(:);
else
    hres = hres_raw;
end
tres_raw = trace.trace_terminal_constraint_residual(:);
if isfield(trace,'trace_terminal_relaxed_constraint_residual') && ~isempty(trace.trace_terminal_relaxed_constraint_residual)
    tres = trace.trace_terminal_relaxed_constraint_residual(:);
else
    tres = tres_raw;
end

% sort by (sample, t) so lines are connected properly
[~, so] = sortrows([smp, tt]);
smp_s = smp(so); tt_s = tt(so); ib_s = ib(so); tb_s = tb(so);
hres_s = hres(so); tres_s = tres(so);

fig = figure('Color','w','Visible','off','Position',[100 100 1200 800]);
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
nexttile; hold on;
for si = unique(smp_s)'
    m = smp_s == si;
    plot(tt_s(m), ib_s(m), 'Color',[0.10,0.55,0.25], 'LineWidth', 0.9);
    plot(tt_s(m), tb_s(m), 'Color',[0.85,0.20,0.20], 'LineWidth', 0.9);
end
plot(nan,nan,'Color',[0.10,0.55,0.25],'DisplayName','integral bound (HOCBF)');
plot(nan,nan,'Color',[0.85,0.20,0.20],'DisplayName','terminal bound (PTCBF)');
grid on; xlabel('s'); ylabel('raw QP upper bound');
title('QP bounds: smaller = tighter'); legend show;
ylim_now = ylim; if diff(ylim_now)>500; ylim([-150 150]); end

nexttile; hold on;
for si = unique(smp_s)'
    m = smp_s == si;
    plot(tt_s(m), hres_s(m), 'Color',[0.10,0.55,0.25], 'LineWidth', 0.9);
    plot(tt_s(m), tres_s(m), 'Color',[0.85,0.20,0.20], 'LineWidth', 0.9);
end
yline(0,'--','Color',[0 0 0]);
plot(nan,nan,'Color',[0.10,0.55,0.25],'DisplayName','HOCBF residual');
plot(nan,nan,'Color',[0.85,0.20,0.20],'DisplayName','PTCBF residual');
grid on; xlabel('s'); ylabel('constraint residual (should be <= 0)');
title('Residuals after QP'); legend show;

qp_emf = fullfile(project_dir,'outputs','tune_first_level_qp.emf');
export_graphics_compat(fig, qp_emf);
close(fig);

% Trajectory panel: reconstructed vs target
target_points = data.target_points;
rec = data.reconstructed;
fig2 = figure('Color','w','Visible','off','Position',[100 100 1000 800]);
hold on;
% Draw target trajectories (all n_train) as grey
for ti = 1:size(target_points,2)
    plot(target_points(:,ti,1), target_points(:,ti,2), 'Color',[0.7 0.7 0.7], 'LineWidth', 0.6);
end
% Draw reconstructed as colored
n_eval = size(rec,2);
c = lines(n_eval);
for si = 1:n_eval
    plot(rec(:,si,1), rec(:,si,2), '-o', 'Color', c(si,:), 'LineWidth', 1.4, ...
        'MarkerSize', 4, 'DisplayName', sprintf('rollout %d', si));
end
grid on; axis equal; xlabel('x'); ylabel('y');
title(sprintf('First-level rollout vs %d target trajectories', size(target_points,2)));
legend('Location','best');
traj_emf = fullfile(project_dir,'outputs','tune_first_level_traj.emf');
export_graphics_compat(fig2, traj_emf);
close(fig2);

fprintf('saved %s\n', qp_png);
fprintf('saved %s\n', traj_png);
exit(0);
