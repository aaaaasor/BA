function safeflow_nn_demo(n_train_steps, n_gen, include_ustg, train_opts)
% train_opts: options forwarded to safeflow_nn_train (e.g. hidden_width,
% n_train).  Empty keeps the defaults, so existing calls are unchanged.  The
% resolved options travel on net.training_options and are written into each
% archive's run_reproduce.m, so an archive retrains at its own width.
if nargin < 4 || isempty(train_opts), train_opts = struct(); end
% include_ustg: also run the u_per_stage variant (default false).  That variant
% re-solves the QP at every RK4 stage, which is NOT what Algorithm 1 says and
% never enters the archive -- it exists only to quantify the cost of the
% deviation.  It is 4x the QP count and 4x the wall time, so it is off by
% default; pass true when that comparison is actually wanted.
if nargin < 3 || isempty(include_ustg), include_ustg = false; end
%SAFEFLOW_NN_DEMO FM (NN) 与 SafeFlow (NN) 的完整对比。
% Safety 判据用真实边界 h >= 0，不含引导用的 margin。
if nargin < 1 || isempty(n_train_steps), n_train_steps = 20000; end
if nargin < 2 || isempty(n_gen),         n_gen = 100; end
this_dir = fileparts(mfilename('fullpath'));
out_dir  = fullfile(this_dir, 'outputs');

net = safeflow_nn_train(n_train_steps, false, train_opts);

% Guidance activation time.  The paper's navigation experiment leaves the
% CBF-QP off until t = 0.5; the supervisor asked for the reproduction to
% constrain from the first step instead, so the whole rollout is guided.
% The paper's own 0.5 schedule is kept archived under
% outputs/<racing>safeflow_tbar05_paper for comparison.
activation_time = 0.0;
% High-gain stress configuration requested by the supervisor: activate the
% second-order pole from the first guided step so divergent pre-projection
% trajectories and terminal-projection failures are reported explicitly.
phi1_form = 'second_order';
phi1_omega = 20.0;
phi1_switch_time = 0.0;
slack_weight = 1e4;

% name, mode, u_per_stage
runs = { 'fm',       'fm',       false; ...
         'safeflow', 'safeflow', false };   % Alg.1 literal: one u per RK4 step
if include_ustg
    % Deviation from Algorithm 1, kept only to price the cost; never archived.
    runs(end+1,:) = { 'safeflow_ustg', 'safeflow', true };
end

R = struct();
for i = 1:size(runs,1)
    nm = runs{i,1};
    fprintf('\n===== %s  (u_per_stage=%d) =====\n', nm, runs{i,3});
    % No fmincon fallback: (32) is solved by sequential linearisation alone.
    % The paper states the terminal projection without naming a solver, so
    % adding a second one would be our implementation choice, not the method.
    % A point that stays infeasible is recorded as a terminal-filter failure
    % and shows up as an unsafe point in the Safety metric.
    r = safeflow_nn_rollout(net, runs{i,2}, n_gen, ...
        struct('u_per_stage', runs{i,3}, 'fmincon_fallback', false, ...
               'activation_time', activation_time, ...
               'phi1_form', phi1_form, ...
               'phi1_omega', phi1_omega, ...
               'phi1_switch_time', phi1_switch_time, ...
               'slack_weight', slack_weight));
    r.metrics = safeflow_nn_metrics(r);
    print_row(r);
    R.(nm) = r;
end

fprintf('\n---- u 更新频率对 Time 的影响 (SafeFlow, %d 条) ----\n', n_gen);
if include_ustg
a = R.safeflow.metrics; b = R.safeflow_ustg.metrics;
fprintf('  %-22s %10s %10s\n', '', '每步一次', '每级一次');
fprintf('  %-22s %10d %10d\n', 'QP 次数',    a.n_qp, b.n_qp);
fprintf('  %-22s %10.4f %10.4f\n', 'Time (s/条)', a.time_seconds, b.time_seconds);
fprintf('  %-22s %10.2f %10.2f\n', 'Safety (%)', a.safety*100, b.safety*100);
fprintf('  %-22s %10.4f %10.4f\n', 'KL', a.kl, b.kl);
fprintf('  %-22s %10.4f %10.4f\n', 'CS', a.cs, b.cs);
fprintf('  %-22s %10.4f %10.4f\n', 'AS', a.as, b.as);
fprintf('  倍数: QP %.2fx, Time %.2fx\n', b.n_qp/max(a.n_qp,1), ...
    b.time_seconds/max(a.time_seconds,eps));
end

%% ---------- 保存 ----------
mat = fullfile(out_dir, 'SafeFlowNN_results.mat');
cfg = get_config();
save(mat, 'R', 'net', 'cfg', 'n_train_steps', 'n_gen', '-v7.3');
fprintf('\n结果已保存: %s\n', mat);

%% ---------- 图 ----------
f = figure('Color','w','Position',[80 80 1200 520]);
ttl = {sprintf('FM (NN)  Safety %.1f%%', R.fm.metrics.safety*100), ...
       sprintf('SafeFlow (NN)  Safety %.1f%%', R.safeflow.metrics.safety*100)};
names = {'fm','safeflow'};   % 图只画论文口径的两条
for s = 1:2
    rr = R.(names{s});
    subplot(1,2,s); hold on;
    draw_track_segment(rr.segment, 'HandleVisibility','off');
    draw_obstacles(rr.obstacle, 'HandleVisibility','off');
    nshow = min(30, size(rr.points,2));
    for i = 1:nshow
        plot(rr.points(:,i,1), rr.points(:,i,2), '-', 'LineWidth', 1.1);
    end
    axis equal; grid on; title(ttl{s});
end
emf = fullfile(out_dir, 'SafeFlowNN_fm_vs_safeflow.emf');
export_graphics_compat(f, emf);
fprintf('图已保存: %s\n', emf);

%% ---------- 归档 ----------
archive_safeflow_nn_run(R, net, cfg, n_train_steps, n_gen);
end

% =====================================================================
function print_row(r)
m = r.metrics;
fprintf('  Safety %6.2f%%   KL %.4f   CS %.4f   AS %.4f\n', ...
    m.safety*100, m.kl, m.cs, m.as);
fprintf('  KL: 端点出界 %d/%d (%.1f%%), q 触底格点 %d\n', ...
    m.kl_details.out_of_grid_count, size(m.final_xy,1), ...
    m.kl_details.out_of_grid_rate*100, m.kl_details.q_at_realmin_cells);
fprintf('  min h: 障碍 %+.6f  边界 %+.6f  (真实边界, 无 margin)\n', ...
    m.min_obstacle_h, m.min_boundary_h);
fprintf('  Time %.4f s/条  = 采样 %.3f + rollout %.3f + 终端 %.3f\n', ...
    m.time_seconds, m.sample_seconds, m.rollout_seconds, m.terminal_seconds);
fprintf('  QP %d 次, slack 激活 %d, 终端滤波失败 %d\n', ...
    m.n_qp, m.slack_active, m.terminal_failed);
if ~isempty(r.u_trace)
    U = double(r.u_trace);                    % (步, 样本, 点, xy)
    nu = sqrt(sum(U.^2, 4));                  % 每个 (步,样本,点) 的 |u|
    fprintf('  |u| (物理空间): mean %.4e  p95 %.4e  max %.4e  非零 %.1f%%\n', ...
        mean(nu(:)), quantile(nu(:), 0.95), max(nu(:)), ...
        100*mean(nu(:) > 1e-12));
    fprintf('  u 轨迹已记录: %s, 时间 t=%.3f..%.3f, %.1f MB\n', ...
        mat2str(size(U)), r.u_trace_t(1), r.u_trace_t(end), ...
        numel(r.u_trace)*4/1e6);
end
fprintf('  seed: 数据 %d, rollout 噪声 %d, 网络初始化 %d\n', ...
    r.data_seed, r.rollout_seed, r.weight_init_seed);
end
