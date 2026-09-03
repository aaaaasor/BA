function archive_safeflow_nn_run(R, net, cfg, n_train_steps, n_gen)
%ARCHIVE_SAFEFLOW_NN_RUN 把 FM / SafeFlow 两次运行归档成可复现的目录。
%
% 目录结构参照 outputs/赛车baseline：
%   outputs/赛车fm/           outputs/赛车safeflow/
%     <prefix>_Net.mat                 网络权重 + 标准化 + 训练种子
%     <prefix>_Rollout.mat             生成轨迹 + 初始噪声 + 每条种子 + u 轨迹
%     <prefix>_Metrics.mat             Safety / KL / CS / AS / Time
%     <prefix>_Trajectory_Seeds.csv    每条轨迹一行：种子 + 该条的指标
%     <prefix>_Trajectory_Seeds.mat
%     <prefix>_KL_Reference.mat        KL 参考网格（与主流程同一份的拷贝）
%     REPRODUCE.md                     复现步骤与全部种子
%     GIT_STATE.txt                    HEAD + 工作区状态
%     MATLAB_ENVIRONMENT.txt           MATLAB 版本 / OS / 已装工具箱
%     run_reproduce.m                  可从本目录直接执行的复现脚本
%     source_snapshot/                 依赖闭包内的 .m + trajectory_data/
%     SHA256SUMS.csv                   本目录所有文件的校验和

this_dir = fileparts(mfilename('fullpath'));
out_root = fullfile(this_dir, 'outputs');
% 无参数调用时，从已保存的结果重新归档（不必重训重跑）
if nargin == 0
    S = load(fullfile(out_root, 'SafeFlowNN_results.mat'));
    R = S.R;  net = S.net;  cfg = S.cfg;
    n_train_steps = S.n_train_steps;  n_gen = S.n_gen;
end
specs = { 'fm',       '赛车fm',       'Racing_FM_NN'
          'safeflow', '赛车safeflow', 'Racing_SafeFlow_NN' };

git_txt = capture_git_state(fileparts(this_dir));

for si = 1:size(specs,1)
    key = specs{si,1};  folder = specs{si,2};  prefix = specs{si,3};
    d = fullfile(out_root, folder);
    if exist(d, 'dir'), rmdir(d, 's'); end
    mkdir(d);  mkdir(fullfile(d, 'source_snapshot'));

    r = R.(key);
    m = r.metrics;

    % ---- 网络 ----
    save(fullfile(d, [prefix '_Net.mat']), 'net', 'n_train_steps', '-v7.3');

    % ---- rollout：轨迹 + 复现所需的一切 ----
    rollout = r;
    rollout.metrics = [];                       % 指标单独存，避免重复
    save(fullfile(d, [prefix '_Rollout.mat']), 'rollout', '-v7.3');

    % ---- 指标 ----
    metrics = m;  metrics.mode = key;  metrics.n_gen = n_gen;
    save(fullfile(d, [prefix '_Metrics.mat']), 'metrics', 'cfg', '-v7.3');

    % ---- 每条轨迹一行的种子表 ----
    T = per_trajectory_table(r, m);
    writetable(T, fullfile(d, [prefix '_Trajectory_Seeds.csv']));
    trajectory_seeds = r.trajectory_seeds;      %#ok<NASGU>
    seed_table = T;                             %#ok<NASGU>
    save(fullfile(d, [prefix '_Trajectory_Seeds.mat']), ...
        'trajectory_seeds', 'seed_table', '-v7.3');

    % ---- KL 参考（拷贝，保证归档自洽）----
    copyfile(m.kl_details.reference_path, ...
        fullfile(d, [prefix '_KL_Reference.mat']));

    % ---- 源码快照 ----
    snapshot_sources(this_dir, fullfile(d, 'source_snapshot'));

    % ---- 图 ----
    make_figures(d, prefix, key, r, m, T, net);

    % ---- 元数据 ----
    fid = fopen(fullfile(d, 'GIT_STATE.txt'), 'w');
    fwrite(fid, git_txt);  fclose(fid);
    write_reproduce_md(fullfile(d, 'REPRODUCE.md'), key, prefix, r, m, ...
        net, n_train_steps, n_gen);
    write_environment(fullfile(d, 'MATLAB_ENVIRONMENT.txt'));
    write_run_reproduce(fullfile(d, 'run_reproduce.m'), key, prefix, ...
        n_train_steps, n_gen);

    % ---- 校验和 ----
    write_sha256(d);
    fprintf('已归档: %s\n', d);
end
end

% =====================================================================
function make_figures(d, prefix, key, r, m, T, net)
% All figure text is English on purpose: these go straight into the thesis.
if strcmp(key,'fm'), name = 'FM (NN)'; else, name = 'SafeFlow (NN)'; end

nP = net.n_points;  nF = net.n_features_per_point;
target_xy = permute(reshape(net.X1, nF, nP, []), [2 3 1]);   % (nP, N, nF)
target_xy = target_xy(:,:,1:2);
% Source panel shows the raw standardized noise, matching plot_results.m
source_xy = permute(reshape(r.z0, nF, nP, []), [2 3 1]);
source_xy = source_xy(:,:,1:2);

% Shared limits for panels 1 and 3, track corridor included.
cmp = [reshape(target_xy, [], 2); reshape(r.points, [], 2); ...
       r.segment.left; r.segment.right];
cmp = cmp(all(isfinite(cmp), 2), :);
xl = [min(cmp(:,1)) max(cmp(:,1))];  yl = [min(cmp(:,2)) max(cmp(:,2))];
xl = xl + 0.05*max(diff(xl), eps)*[-1 1];
yl = yl + 0.05*max(diff(yl), eps)*[-1 1];

%% ---- ThreePanel: target / source / rollout ----
f = figure('Color','w','Position',[60 60 1500 520],'Visible','off');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile; hold on;
draw_track_segment(r.segment, 'HandleVisibility','off');
draw_obstacles(r.obstacle, 'HandleVisibility','off');
for i = 1:size(target_xy,2)
    plot(target_xy(:,i,1), target_xy(:,i,2), '.-', 'LineWidth', 0.8, ...
        'HandleVisibility','off');
end
grid on; axis equal; xlim(xl); ylim(yl); xlabel('x'); ylabel('y');
title(sprintf('Target Trajectory Data (2D, %d Points)', nP));

nexttile; hold on;
for i = 1:size(source_xy,2)
    plot(source_xy(:,i,1), source_xy(:,i,2), '--', 'LineWidth', 0.9, ...
        'HandleVisibility','off');
end
grid on; axis equal; xlabel('x'); ylabel('y');
title('ODE Source Trajectories');

nexttile; hold on;
rollout_track = r.segment;
if isfield(rollout_track, 'raceline'), rollout_track.raceline(:) = nan; end
draw_track_segment(rollout_track, 'HandleVisibility','off');
draw_obstacles(r.obstacle, 'HandleVisibility','off');
unsafe = ~T.safe;
for i = find(~unsafe)'
    plot(r.points(:,i,1), r.points(:,i,2), '-', 'Color', [0 .45 .74 .35], ...
        'LineWidth', .9, 'HandleVisibility','off');
end
for i = find(unsafe)'
    plot(r.points(:,i,1), r.points(:,i,2), '-', 'Color', [.85 .1 .1 .6], ...
        'LineWidth', 1.1, 'HandleVisibility','off');
end
plot(nan,nan,'-','Color',[0 .45 .74],'DisplayName', ...
    sprintf('Safe (%d)', nnz(~unsafe)));
if any(unsafe)
    plot(nan,nan,'-','Color',[.85 .1 .1],'DisplayName', ...
        sprintf('Unsafe (%d)', nnz(unsafe)));
end
scatter(squeeze(r.points(end,:,1)), squeeze(r.points(end,:,2)), 14, 'k', ...
    'filled', 'DisplayName', 'Endpoints');
grid on; axis equal; xlim(xl); ylim(yl); legend('Location','best');
xlabel('x'); ylabel('y');
title(sprintf('%s Rollout (%d curves, Safety %.2f%%)', ...
    name, r.n_gen, 100*m.safety));
save_fig(f, fullfile(d, [prefix '_ThreePanel.emf']));

%% ---- Per-trajectory minimum barrier value ----
f = figure('Color','w','Position',[80 80 900 380],'Visible','off');
hold on;
plot(T.trajectory_index, T.min_obstacle_h, '.-', 'MarkerSize', 9, ...
    'DisplayName','Obstacle min h');
plot(T.trajectory_index, T.min_boundary_h, '.-', 'MarkerSize', 9, ...
    'DisplayName','Track boundary min h');
yline(0, 'k--', 'HandleVisibility','off');
grid on; legend('Location','best');
xlabel('Trajectory index'); ylabel('min h (true boundary, no margin)');
title(sprintf('%s  Per-Trajectory Safety Margin', name));
save_fig(f, fullfile(d, [prefix '_hmin.emf']));

%% ---- QP correction magnitude (SafeFlow only) ----
if ~isempty(r.u_trace)
    U = double(r.u_trace);                 % (step, sample, point, xy)
    nu = sqrt(sum(U.^2, 4));
    per_t = reshape(nu, size(nu,1), []);
    f = figure('Color','w','Position',[80 80 980 400],'Visible','off');
    subplot(1,2,1); hold on;
    plot(r.u_trace_t, mean(per_t,2), 'LineWidth', 1.5, 'DisplayName','mean');
    plot(r.u_trace_t, quantile(per_t,0.95,2), '--', 'LineWidth',1.2, ...
        'DisplayName','p95');
    plot(r.u_trace_t, max(per_t,[],2), ':', 'LineWidth',1.2, 'DisplayName','max');
    grid on; legend('Location','best');
    xlabel('Generation time t'); ylabel('|u|');
    title('QP Correction Magnitude (state space)');
    subplot(1,2,2);
    sl = double(r.u_trace_slack);
    plot(r.u_trace_t, 100*mean(reshape(sl, size(sl,1), []), 2), 'LineWidth', 1.5);
    grid on; xlabel('Generation time t'); ylabel('Slack activation rate (%)');
    title(sprintf('Overall slack activation %.1f%%', ...
        100*r.slack_active/max(r.n_qp,1)));
    save_fig(f, fullfile(d, [prefix '_u_norm.emf']));
end
end

function save_fig(f, p)
export_graphics_compat(f, p);
close(f);
end

% =====================================================================
function T = per_trajectory_table(r, m)
n = r.n_gen;
P = r.points;
seed = r.trajectory_seeds(:);
safe = false(n,1); min_obs = nan(n,1); min_bnd = nan(n,1);
for i = 1:n
    ho = inf; hb = inf;
    for k = 1:size(P,1)
        p = [P(k,i,1); P(k,i,2)];
        for jo = 1:size(r.obstacle.centers,2)
            ho = min(ho, obstacle_level_and_gradient(p, r.obstacle, jo));
        end
        for bi = 1:2
            hb = min(hb, evaluate_track_implicit_field( ...
                r.geometry.implicit_fields, bi, p));
        end
    end
    min_obs(i) = ho; min_bnd(i) = hb;
    safe(i) = (ho >= -1e-8) && (hb >= -1e-8);
end
% 该条的 |u| 统计
u_mean = nan(n,1); u_max = nan(n,1); u_nnz = nan(n,1);
if ~isempty(r.u_trace)
    U = double(r.u_trace);                      % (步, 样本, 点, xy)
    nu = sqrt(sum(U.^2, 4));                    % (步, 样本, 点)
    for i = 1:n
        v = reshape(nu(:,i,:), [], 1);
        u_mean(i) = mean(v); u_max(i) = max(v); u_nnz(i) = mean(v > 1e-12);
    end
end
T = table((1:n)', seed, safe, min_obs, min_bnd, ...
    m.cs_per(:), m.as_per(:), ...
    P(end,:,1)', P(end,:,2)', u_mean, u_max, u_nnz, ...
    'VariableNames', {'trajectory_index','seed','safe', ...
    'min_obstacle_h','min_boundary_h','cs','as', ...
    'final_x','final_y','u_norm_mean','u_norm_max','u_nonzero_rate'});
end

function snapshot_sources(src_dir, dst_dir)
% 入口点；其余 .m 依赖由 requiredFilesAndProducts 自动求闭包，避免手工漏项。
entries = { 'safeflow_nn_train.m', 'safeflow_nn_rollout.m', ...
            'safeflow_nn_demo.m', 'safeflow_nn_kl.m', ...
            'safeflow_nn_ablate_terminal.m', 'archive_safeflow_nn_run.m', ...
            'verify_qp2d.m', 'build_track_dataset.m' };
missing = {};
entry_paths = {};
for i = 1:numel(entries)
    p = fullfile(src_dir, entries{i});
    if isfile(p), entry_paths{end+1} = p; else, missing{end+1} = entries{i}; end %#ok<AGROW>
end

files = entry_paths;
try
    dep = matlab.codetools.requiredFilesAndProducts(entry_paths);
    files = unique([entry_paths, cellstr(dep(:))']);
catch err
    warning('archive:DependencyScan', ...
        '依赖自动扫描失败 (%s)，只归档入口文件。', err.message);
end
% 只保留工程目录内的文件（排除 MATLAB 自带函数）
src_full = string(java.io.File(src_dir).getCanonicalPath());
for i = 1:numel(files)
    p = files{i};
    if ~startsWith(string(java.io.File(p).getCanonicalPath()), src_full)
        continue;                     % MATLAB 内置/工具箱函数，不归档
    end
    copyfile(p, dst_dir);
end

% 赛道数据集与原始 CSV：不带这些就无法从归档独立重跑训练。
data_dst = fullfile(dst_dir, 'trajectory_data');
if ~exist(data_dst, 'dir'), mkdir(data_dst); end
data_files = { 'track_dataset_arena.mat', 'Nuerburgring.csv', ...
               'Nuerburgring_raceline.csv' };
for i = 1:numel(data_files)
    p = fullfile(src_dir, 'trajectory_data', data_files{i});
    if isfile(p), copyfile(p, data_dst); else, missing{end+1} = data_files{i}; end %#ok<AGROW>
end
if ~isempty(missing)
    warning('archive:MissingSnapshotFiles', ...
        '源码快照缺少: %s', strjoin(missing, ', '));
end
end

function write_environment(path)
L = {'# 运行环境（归档时记录）', ''};
L{end+1} = sprintf('MATLAB      : %s (%s)', version, version('-release'));
L{end+1} = sprintf('Computer    : %s', computer);
L{end+1} = sprintf('OS          : %s', os_string());
L{end+1} = sprintf('Archived at : %s', ...
    char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')));
L{end+1} = '';
L{end+1} = '## 必需的工具箱';
L{end+1} = '';
L{end+1} = ['- Optimization Toolbox — 仅用于两处：终端投影的 fmincon 兜底' ...
    '（正常配置下极少触发），以及 verify_qp2d 的 quadprog 校验。'];
L{end+1} = '  主 rollout 路径不需要它（QP 用自写的活跃集穷举求解）。';
L{end+1} = '- 不需要 Deep Learning Toolbox：网络的前向、反向与 Adam 都是手写的。';
L{end+1} = '';
L{end+1} = '## 归档时已安装的工具箱';
L{end+1} = '';
v = ver;
for i = 1:numel(v)
    L{end+1} = sprintf('- %s %s', v(i).Name, v(i).Version); %#ok<AGROW>
end
fid = fopen(path, 'w', 'n', 'UTF-8');
fwrite(fid, unicode2native(strjoin(L, newline), 'UTF-8'));
fclose(fid);
end

function s = os_string()
try
    if ispc
        [~, s] = system('ver');
    elseif ismac
        [~, s] = system('sw_vers -productVersion');
    else
        [~, s] = system('uname -a');
    end
    s = strtrim(strrep(s, newline, ' '));
catch
    s = 'unknown';
end
end

function write_run_reproduce(path, key, prefix, n_steps, n_gen)
L = {};
L{end+1} = 'function run_reproduce(retrain)';
L{end+1} = sprintf('%%RUN_REPRODUCE 从本归档目录独立重跑 %s。', prefix);
L{end+1} = '%';
L{end+1} = '%   cd 到本目录后 run_reproduce        — 复用网络重跑 + 单条重放';
L{end+1} = '%   run_reproduce(true)                — 另加从零重训并比对权重';
L{end+1} = '%';
L{end+1} = '% 依赖全部在 source_snapshot/ 内（含 trajectory_data/），不需要 BA 工程。';
L{end+1} = 'here = fileparts(mfilename(''fullpath''));';
L{end+1} = 'snap = fullfile(here, ''source_snapshot'');';
L{end+1} = 'addpath(snap);  old = cd(snap);  restore = onCleanup(@() cd(old));';
L{end+1} = '';
L{end+1} = '%% 1) 复用归档网络重跑 rollout（快，秒级）';
L{end+1} = sprintf('S = load(fullfile(here, ''%s_Net.mat''));', prefix);
L{end+1} = sprintf('r = safeflow_nn_rollout(S.net, ''%s'', %d);', key, n_gen);
L{end+1} = sprintf('A = load(fullfile(here, ''%s_Rollout.mat''));', prefix);
L{end+1} = 'd = max(abs(r.points(:) - A.rollout.points(:)));';
L{end+1} = 'fprintf(''复用网络重跑: 轨迹最大偏差 %.3e\n'', d);';
L{end+1} = '';
L{end+1} = '%% 2) 单条重放（用该条自己的 seed）';
L{end+1} = sprintf('T = readtable(fullfile(here, ''%s_Trajectory_Seeds.csv''));', prefix);
L{end+1} = 'i = 1;';
L{end+1} = sprintf(['r1 = safeflow_nn_rollout(S.net, ''%s'', 1, ...\n' ...
    '    struct(''trajectory_seeds'', T.seed(i)));'], key);
L{end+1} = 'd1 = max(abs(r1.points(:) - reshape(A.rollout.points(:,i,:), [], 1)));';
L{end+1} = 'fprintf(''第 %d 条单独重放: 最大偏差 %.3e\n'', i, d1);';
L{end+1} = '';
L{end+1} = '%% 3) 从零重训并比对权重（可选，约 2 分钟）';
L{end+1} = 'if nargin >= 1 && retrain';
L{end+1} = sprintf('    net2 = safeflow_nn_train(%d, false);', n_steps);
L{end+1} = '    dw = 0;';
L{end+1} = '    for j = 1:numel(S.net.P)';
L{end+1} = '        dw = max(dw, max(abs(net2.P{j}(:) - S.net.P{j}(:))));';
L{end+1} = '    end';
L{end+1} = '    fprintf(''从零重训: 权重最大偏差 %.3e\n'', dw);';
L{end+1} = 'else';
L{end+1} = '    fprintf(''从零重训已跳过 (run_reproduce(true) 启用, 约 2 分钟)\n'');';
L{end+1} = 'end';
L{end+1} = 'end';
fid = fopen(path, 'w', 'n', 'UTF-8');
fwrite(fid, unicode2native(strjoin(L, newline), 'UTF-8'));
fclose(fid);
end

function txt = capture_git_state(repo_dir)
lines = {sprintf('Archived at: %s', ...
    char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')))};
[s1, head] = system(sprintf('git -C "%s" rev-parse HEAD', repo_dir));
if s1 == 0
    lines{end+1} = sprintf('Git HEAD: %s', strtrim(head));
    [~, st] = system(sprintf('git -C "%s" status --porcelain', repo_dir));
    lines{end+1} = '';
    lines{end+1} = 'Working tree status at archive time:';
    lines{end+1} = strtrim(st);
else
    lines{end+1} = 'Git HEAD: (not a git repository)';
end
txt = strjoin(lines, newline);
end

function write_reproduce_md(path, key, prefix, r, m, net, n_steps, n_gen)
L = {};
L{end+1} = sprintf('# %s — 复现说明', prefix);
L{end+1} = '';
L{end+1} = sprintf('生成于 %s，MATLAB %s。', ...
    char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')), version('-release'));
L{end+1} = '';
L{end+1} = '## 结果';
L{end+1} = '';
L{end+1} = '| 指标 | 值 |';
L{end+1} = '|---|---|';
L{end+1} = sprintf('| Safety | %.2f%% |', 100*m.safety);
L{end+1} = sprintf('| KL | %.6f |', m.kl);
L{end+1} = sprintf('| CS | %.6f |', m.cs);
L{end+1} = sprintf('| AS | %.6f |', m.as);
L{end+1} = sprintf('| Time (s/条) | %.6f |', m.time_seconds);
L{end+1} = sprintf('| min h 障碍 | %+.6f |', m.min_obstacle_h);
L{end+1} = sprintf('| min h 边界 | %+.6f |', m.min_boundary_h);
L{end+1} = '';
L{end+1} = '## 全部种子';
L{end+1} = '';
L{end+1} = sprintf('- 训练数据 `cfg.random_seed` = **%d**', r.data_seed);
L{end+1} = sprintf('- 网络权重初始化 = **%d**', r.weight_init_seed);
L{end+1} = sprintf('- rollout 基种子 = **%d**（派生出 %d 个逐条种子）', ...
    r.rollout_seed, n_gen);
L{end+1} = sprintf('- 逐条种子: 见 `%s_Trajectory_Seeds.csv` 的 `seed` 列', prefix);
L{end+1} = '';
L{end+1} = sprintf('种子约定: %s', r.seed_convention);
L{end+1} = '';
L{end+1} = '## 复现步骤';
L{end+1} = '';
L{end+1} = '```matlab';
L{end+1} = '% 1) 全部重跑（网络 + 两个 rollout + 归档）';
L{end+1} = sprintf('safeflow_nn_demo(%d, %d);', n_steps, n_gen);
L{end+1} = '';
L{end+1} = '% 2) 只重跑本方法的 rollout（复用归档的网络）';
L{end+1} = sprintf('S = load(''%s_Net.mat'');', prefix);
L{end+1} = sprintf('r = safeflow_nn_rollout(S.net, ''%s'', %d);', key, n_gen);
L{end+1} = '';
L{end+1} = '% 3) 单独重放第 i 条轨迹（用它自己的种子）';
L{end+1} = sprintf('T = readtable(''%s_Trajectory_Seeds.csv'');', prefix);
L{end+1} = 'i = 57;';
L{end+1} = sprintf(['r_one = safeflow_nn_rollout(S.net, ''%s'', 1, ...\n' ...
    '    struct(''trajectory_seeds'', T.seed(i)));'], key);
L{end+1} = '% r_one.points 应与归档 Rollout.mat 的第 i 条逐位相同';
L{end+1} = '```';
L{end+1} = '';
L{end+1} = '## 配置';
L{end+1} = '';
L{end+1} = sprintf('- 状态维度 %d = %d 点 x %d 特征 [x, y, dx/ds, dy/ds]', ...
    numel(net.mu_d), net.n_points, net.n_features_per_point);
L{end+1} = sprintf('- 网络 %d -> 256^3 -> %d, %d 参数, %d 步训练', ...
    numel(net.mu_d)+1, numel(net.mu_d), ...
    sum(cellfun(@numel, net.P)), n_steps);
L{end+1} = sprintf('- 积分器 固定步 RK4, %d 步, t: 0 -> %.3f', ...
    r.n_rk_steps, r.t_max);
if strcmp(key, 'safeflow')
    L{end+1} = sprintf(['- CFMBF 激活 t >= %.2f; phi0 = %.1f; ' ...
        'phi1 = 1+4t^3 (t<%.1f), 1/(1-t) (t>=%.1f)'], ...
        r.activation_time, r.phi0, r.phi1_switch_time, r.phi1_switch_time);
    L{end+1} = sprintf('- QP %d 次, slack 激活 %d (%.1f%%)', ...
        r.n_qp, r.slack_active, 100*r.slack_active/max(r.n_qp,1));
    L{end+1} = sprintf(['- 终端安全滤波(式 32): 移动 %d 条, ' ...
        'fmincon 兜底 %d, 失败 %d'], r.terminal_corrected, ...
        r.terminal_rescued_by_fmincon, r.terminal_failed);
    L{end+1} = sprintf('- u 每个积分步解一次 (Algorithm 1 字面读法), u_per_stage=%d', ...
        r.u_per_stage);
    L{end+1} = sprintf('- 边界 margin %.4f（引导与投影用 h-m; Safety 用 h>=0）', ...
        r.margin);
end
L{end+1} = '';
L{end+1} = '## 说明';
L{end+1} = '';
L{end+1} = ['- **方差留空**：NN 骨干是确定性流场，没有后验方差。' ...
    'LoG-GP 的预测方差是 GP 特有的量，不适用于本方法。'];
L{end+1} = '- KL 使用与主流程同一份参考网格（本目录内附拷贝），跨方法可比。';
L{end+1} = ['- `source_snapshot/` 内含复现所需的 .m 源码，以及 ' ...
    '`trajectory_data/`（`track_dataset_arena.mat` 与原始的 ' ...
    'Nürburgring CSV，附生成脚本 `build_track_dataset.m`），' ...
    '因此可脱离 BA 工程独立重跑。'];
L{end+1} = ['- **Time 不可逐位复现**：它取决于 CPU、内存带宽和当时的系统负载。' ...
    'Safety / KL / CS / AS 与轨迹本身是确定性的，可逐位复现；' ...
    'Time 只应作数量级比较。'];
L{end+1} = ['- 运行环境见 `MATLAB_ENVIRONMENT.txt`；' ...
    '`run_reproduce.m` 可从本目录直接执行。'];
fid = fopen(path, 'w', 'n', 'UTF-8');
fwrite(fid, unicode2native(strjoin(L, newline), 'UTF-8'));
fclose(fid);
end

function write_sha256(d)
f = dir(fullfile(d, '**', '*'));
f = f(~[f.isdir]);
rows = cell(numel(f), 3);
for i = 1:numel(f)
    p = fullfile(f(i).folder, f(i).name);
    rows{i,1} = strrep(erase(p, [d filesep]), filesep, '/');
    rows{i,2} = f(i).bytes;
    rows{i,3} = sha256_of(p);
end
T = table(string(rows(:,1)), cell2mat(rows(:,2)), string(rows(:,3)), ...
    'VariableNames', {'Path','Bytes','SHA256'});
writetable(T, fullfile(d, 'SHA256SUMS.csv'));
end

function h = sha256_of(p)
fid = fopen(p, 'r');  b = fread(fid, inf, '*uint8');  fclose(fid);
md = java.security.MessageDigest.getInstance('SHA-256');
dg = typecast(md.digest(b), 'uint8');
h = upper(sprintf('%02X', dg));
end
