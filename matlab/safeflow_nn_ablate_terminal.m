function safeflow_nn_ablate_terminal(n_train_steps, n_gen)
%SAFEFLOW_NN_ABLATE_TERMINAL SafeFlow 去掉终端安全滤波后还剩多少安全率。
%
% 三个配置共用同一个网络、同一批初始噪声：
%   fm            无引导、无投影
%   safeflow_noTF CFMBF 引导，但不做式 32 的终端投影
%   safeflow      引导 + 终端投影（完整方法）
% 判据：真实边界 h >= 0，不含 margin。

if nargin < 1 || isempty(n_train_steps), n_train_steps = 20000; end
if nargin < 2 || isempty(n_gen),         n_gen = 100; end

net = safeflow_nn_train(n_train_steps, false);

runs = { 'fm',            'fm',       false
         'safeflow_noTF', 'safeflow', false
         'safeflow',      'safeflow', true  };

fprintf('\n%-14s %8s %10s %8s %12s %12s %10s\n', '配置', 'Safety', ...
    '违反点数', '占比', 'min h 障碍', 'min h 边界', 'Time(s)');
R = struct();
for i = 1:size(runs,1)
    nm = runs{i,1};
    r = safeflow_nn_rollout(net, runs{i,2}, n_gen, ...
        struct('terminal_filter', runs{i,3}));
    [sr, nbad, ntot, ho, hb] = safety_stats(r);
    R.(nm) = r;  R.(nm).safety = sr;
    fprintf('%-14s %7.1f%% %10d %7.2f%% %+12.6f %+12.6f %10.4f\n', ...
        nm, 100*sr, nbad, 100*nbad/ntot, ho, hb, r.total_seconds_per_traj);
end

fprintf('\nslack 激活: %d / %d QP (%.1f%%)\n', R.safeflow.slack_active, ...
    R.safeflow.n_qp, 100*R.safeflow.slack_active/max(R.safeflow.n_qp,1));
fprintf('终端投影: 移动 %d 条, 失败 %d, fmincon 兜底 %d\n', ...
    R.safeflow.terminal_corrected, R.safeflow.terminal_failed, ...
    R.safeflow.terminal_rescued_by_fmincon);

out = fullfile(fileparts(mfilename('fullpath')), 'outputs', ...
    'SafeFlowNN_TerminalAblation.mat');
save(out, 'R', 'n_gen', 'n_train_steps', '-v7.3');
fprintf('已保存 %s\n', out);
end

function [rate, nbad, ntot, hmin_o, hmin_b] = safety_stats(r)
P = r.points;  [nPt, n, ~] = size(P);
ok = true(1,n); nbad = 0; ntot = nPt*n; hmin_o = inf; hmin_b = inf;
for i = 1:n
    for k = 1:nPt
        p = [P(k,i,1); P(k,i,2)];  bad = false;
        for jo = 1:size(r.obstacle.centers,2)
            h = obstacle_level_and_gradient(p, r.obstacle, jo);
            hmin_o = min(hmin_o, h);
            if h < -1e-8, bad = true; end
        end
        for bi = 1:2
            hr = evaluate_track_implicit_field(r.geometry.implicit_fields, bi, p);
            hmin_b = min(hmin_b, hr);
            if hr < -1e-8, bad = true; end
        end
        if bad, nbad = nbad + 1; ok(i) = false; end
    end
end
rate = mean(ok);
end
