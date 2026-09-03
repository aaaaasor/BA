function archive_safeflow_nn_run(R, net, cfg, n_train_steps, n_gen)
%ARCHIVE_SAFEFLOW_NN_RUN 把 FM / SafeFlow 两次运行归档成可复现的目录。
%
% 目录结构参照 outputs/赛车baseline：
%   outputs/racing fm/        outputs/racing safeflow/
%     <prefix>_Net.mat                 网络权重 + 标准化 + 训练种子
%     <prefix>_Rollout.mat             生成轨迹 + 初始噪声 + 每条种子 + u 轨迹
%     <prefix>_Metrics.mat             Safety / KL / CS / AS / Time
%     <prefix>_Trajectory_Seeds.csv    每条轨迹一行：种子 + 该条的指标
%     <prefix>_Trajectory_Seeds.mat
%     <prefix>_KL_Reference.mat        KL 参考网格（与主流程同一份的拷贝）
%     REPRODUCE.md                     复现步骤与全部种子
%     GIT_STATE.txt                    HEAD + 工作区状态
%     source_snapshot/                 复现所需的全部 .m 源码
%     SHA256SUMS.csv                   本目录所有文件的校验和

this_dir = fileparts(mfilename('fullpath'));
out_root = fullfile(this_dir, 'outputs');
specs = { 'fm',       'racing fm',       'Racing_FM_NN'
          'safeflow', 'racing safeflow', 'Racing_SafeFlow_NN' };

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

    % ---- 元数据 ----
    fid = fopen(fullfile(d, 'GIT_STATE.txt'), 'w');
    fwrite(fid, git_txt);  fclose(fid);
    write_reproduce_md(fullfile(d, 'REPRODUCE.md'), key, prefix, r, m, ...
        net, n_train_steps, n_gen);

    % ---- 校验和 ----
    write_sha256(d);
    fprintf('已归档: %s\n', d);
end
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
files = { 'safeflow_nn_train.m', 'safeflow_nn_rollout.m', ...
          'safeflow_nn_demo.m', 'safeflow_nn_kl.m', ...
          'safeflow_nn_ablate_terminal.m', 'archive_safeflow_nn_run.m', ...
          'get_config.m', 'scenario_training_points.m', ...
          'build_track_boundary_geometry.m', 'configure_racing_obstacles.m', ...
          'obstacle_level_and_gradient.m', 'evaluate_track_implicit_field.m', ...
          'struct_field_default.m', 'export_graphics_compat.m', ...
          'draw_track_segment.m', 'draw_obstacles.m', 'verify_qp2d.m' };
for i = 1:numel(files)
    p = fullfile(src_dir, files{i});
    if isfile(p), copyfile(p, dst_dir); end
end
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
L{end+1} = '- `source_snapshot/` 内含复现所需的全部 .m 源码。';
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
