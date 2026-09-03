function safeflow_nn_demo(n_train_steps, n_gen)
%SAFEFLOW_NN_DEMO FM (NN) 与 SafeFlow (NN) 的完整对比。
% Safety 判据用真实边界 h >= 0，不含引导用的 margin。
if nargin < 1 || isempty(n_train_steps), n_train_steps = 20000; end
if nargin < 2 || isempty(n_gen),         n_gen = 100; end
this_dir = fileparts(mfilename('fullpath'));
out_dir  = fullfile(this_dir, 'outputs');

net = safeflow_nn_train(n_train_steps, false);

% name, mode, u_per_stage
runs = { 'fm',           'fm',       false
         'safeflow',     'safeflow', false     % Alg.1 字面：每步解一次 u
         'safeflow_ustg','safeflow', true  };  % 每个 RK4 级重解 u（偏离）

R = struct();
for i = 1:size(runs,1)
    nm = runs{i,1};
    fprintf('\n===== %s  (u_per_stage=%d) =====\n', nm, runs{i,3});
    r = safeflow_nn_rollout(net, runs{i,2}, n_gen, ...
        struct('u_per_stage', runs{i,3}));
    r.metrics = compute_metrics(r, net);
    print_row(r);
    R.(nm) = r;
end

fprintf('\n---- u 更新频率对 Time 的影响 (SafeFlow, %d 条) ----\n', n_gen);
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
function m = compute_metrics(r, ~)
P = r.points;                      % (65, n, 2)
[nPt, n, ~] = size(P);

% ---- Safety：真实边界 h >= 0，不含 margin ----
ok = true(1,n); hmin_o = inf; hmin_b = inf;
for i = 1:n
    for k = 1:nPt
        p = [P(k,i,1); P(k,i,2)];
        for jo = 1:size(r.obstacle.centers,2)
            h = obstacle_level_and_gradient(p, r.obstacle, jo);
            hmin_o = min(hmin_o, h);
            if h < -1e-8, ok(i) = false; end
        end
        for bi = 1:2
            hr = evaluate_track_implicit_field(r.geometry.implicit_fields, bi, p);
            hmin_b = min(hmin_b, hr);          % 不减 margin
            if hr < -1e-8, ok(i) = false; end
        end
    end
end
m.safety = mean(ok);
m.min_obstacle_h = hmin_o;
m.min_boundary_h = hmin_b;

% ---- CS / AS ----
cs = nan(n,1); as = nan(n,1);
for i = 1:n
    xy = squeeze(P(:,i,1:2));
    w  = diff(xy,1,1);
    L  = sqrt(sum(w.^2,2));
    den = L(1:end-1).*L(2:end);
    ct  = ones(size(den));
    v_  = den > eps;
    ct(v_) = sum(w(1:end-1,:).*w(2:end,:),2)./den(v_);
    cs(i) = mean(1 - min(max(ct,-1),1));
    acc = diff(xy,2,1);
    as(i) = mean(sqrt(sum(acc.^2,2)));
end
m.cs = mean(cs); m.as = mean(as);
m.cs_per = cs;   m.as_per = as;

% ---- Time：采样 + rollout + 终端滤波，除以条数 ----
m.time_seconds = r.total_seconds_per_traj;
m.sample_seconds = r.sample_seconds;
m.rollout_seconds = r.rollout_seconds;
m.terminal_seconds = r.terminal_seconds;
m.terminal_failed = r.terminal_failed;
m.slack_active = r.slack_active;
m.n_qp = r.n_qp;

% ---- KL：复用主流程的 Racing_KL_Reference.mat（同网格同带宽，跨方法可比）----
m.final_xy = squeeze(P(end,:,1:2));
[m.kl, m.kl_details] = safeflow_nn_kl(m.final_xy);
end

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
