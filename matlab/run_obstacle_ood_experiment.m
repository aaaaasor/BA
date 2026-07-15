% SafeFlow 避障 A/B 实验: 三层始终开避障(整条曲线绕障)，只切换第三层的
% "OOD 控制"(方差 HOCBF + 末端方差 PTCBF)，观察无 OOD 时第三层内部点被
% 障碍推离训练流形后的表现。
%
% 复用 main_demo.m: 通过在工作区注入 cfg_external，让 main_demo 用它作为
% 配置并跳过默认 clear。L1/L2 两变体完全一致(缓存共享)，只有 L3 换缓存路径。
%
% fix 7: 不再只看 diverged 计数。每个变体额外报告——
%   离散 h_min / 连续 h_min / 穿障样本数 (evaluate_trajectory_obstacle_safety)
%   GP 不确定度 max/mean、correction norm、障碍 rollout min-h / max-slack
%   曲率平滑度 CS、加速度平滑度 AS (论文式 35/37)
clear; clc;

variants = struct( ...
    'label', {'A: with OOD', 'B: no OOD'}, ...
    'ood',   {true, false}, ...
    'cache', {'LoG_GP_ThirdLevel_Rollout_ObstacleOOD_A.mat', ...
              'LoG_GP_ThirdLevel_Rollout_ObstacleOOD_B.mat'}, ...
    'color', {[0 0.45 0.74], [0.85 0.10 0.10]});

ood_results = cell(1, numel(variants));
ood_obstacle = [];
ood_target = [];
for v = 1:numel(variants)
    clearvars -except variants ood_results ood_obstacle ood_target v
    cfg_external = get_config();
    cfg_external.obstacle.enabled = true;                 % 避障总开关: 始终开
    cfg_external.variance_constraint.third_level_hocbf_enabled = variants(v).ood;
    cfg_external.variance_constraint.third_level_ptcbf_enabled = variants(v).ood;
    cfg_external.cache.third_level_rollout_path = ...
        fullfile('outputs', variants(v).cache);
    cfg_external.animation.enabled = false;               % A/B 不需要动画
    fprintf('\n===== Obstacle OOD variant %s (L3 hocbf/ptcbf = %d) =====\n', ...
        variants(v).label, variants(v).ood);
    main_demo;                                            % 用 cfg_external 跑一遍

    % ---- 指标汇总 (fix 4/7) ----
    recon = reconstructed_points_plot;
    res = struct();
    res.recon = recon;
    res.diverged = third_level_diverged_sample_count;
    % 障碍安全评价: 原始 h(离散点 + 相邻线段连续极小), 与 QP slack 无关。
    safety = evaluate_trajectory_obstacle_safety(recon, cfg_external.obstacle);
    res.discrete_h_min = safety.discrete_h_min;
    res.continuous_h_min = safety.continuous_h_min;
    res.n_crossing_samples = safety.n_crossing_samples;
    res.n_crossing_segments = safety.n_crossing_segments;
    % 平滑度 CS / AS (逐样本平均)。
    n_s = size(recon, 2);
    cs_list = zeros(n_s, 1);
    as_list = zeros(n_s, 1);
    for s = 1:n_s
        [cs_list(s), as_list(s)] = trajectory_smoothness(squeeze(recon(:, s, 1:2)));
    end
    res.cs = mean(cs_list);
    res.as = mean(as_list);
    % GP 不确定度 (偏离流形的直接信号)。
    if exist('uncertainty_values', 'var') && ~isempty(uncertainty_values)
        res.sigma_max = max(uncertainty_values(:));
        res.sigma_mean = mean(uncertainty_values(:));
    else
        res.sigma_max = nan;
        res.sigma_mean = nan;
    end
    % rollout 诊断: correction norm / 障碍 rollout min-h / max-slack。
    if exist('third_rollout_diagnostics', 'var') && ...
            isfield(third_rollout_diagnostics, 'hocbf')
        dh = third_rollout_diagnostics.hocbf;
        res.max_correction_norm = struct_field_default(dh, ...
            'max_correction_norm', nan);
        res.obstacle_rollout_h_min = struct_field_default(dh, ...
            'min_obstacle_h', nan);
        res.obstacle_max_slack = struct_field_default(dh, ...
            'max_obstacle_slack', nan);
    else
        res.max_correction_norm = nan;
        res.obstacle_rollout_h_min = nan;
        res.obstacle_max_slack = nan;
    end
    ood_results{v} = res;
    ood_obstacle = cfg_external.obstacle;
    ood_target = target_points_plot;
end

%% 叠加图: 有 OOD vs 无 OOD
figure('Name', 'SafeFlow obstacle avoidance: with vs without OOD control');
hold on;
draw_obstacles(ood_obstacle);
for s_idx = 1:size(ood_target, 2)
    plot(ood_target(:, s_idx, 1), ood_target(:, s_idx, 2), '-', ...
        'Color', [0.8 0.8 0.8], 'LineWidth', 0.5, 'HandleVisibility', 'off');
end
h_legend = gobjects(1, numel(variants));
for v = 1:numel(variants)
    recon = ood_results{v}.recon;
    for s_idx = 1:size(recon, 2)
        hh = plot(recon(:, s_idx, 1), recon(:, s_idx, 2), '.-', ...
            'Color', variants(v).color, 'MarkerSize', 6, 'LineWidth', 0.8);
        if s_idx == 1
            h_legend(v) = hh;
        else
            set(hh, 'HandleVisibility', 'off');
        end
    end
end
axis equal; grid on;
xlabel('x'); ylabel('y');
title('SafeFlow obstacle avoidance: with vs without OOD control');
legend_labels = arrayfun(@(v) sprintf('%s (cross=%d, diverged=%d)', ...
    variants(v).label, ood_results{v}.n_crossing_samples, ...
    ood_results{v}.diverged), 1:numel(variants), 'UniformOutput', false);
legend(h_legend, legend_labels, 'Location', 'best');
hold off;

%% 指标对比表
fprintf('\n============ Obstacle OOD experiment metrics ============\n');
fprintf('%-34s | %12s | %12s\n', 'metric', variants(1).label, variants(2).label);
fprintf('%s\n', repmat('-', 1, 62));
print_row = @(name, f) fprintf('%-34s | %12.4g | %12.4g\n', name, ...
    ood_results{1}.(f), ood_results{2}.(f));
print_row('diverged sample count', 'diverged');
print_row('crossing sample count', 'n_crossing_samples');
print_row('crossing segment count', 'n_crossing_segments');
print_row('discrete h_min (>0 safe)', 'discrete_h_min');
print_row('continuous h_min (>0 safe)', 'continuous_h_min');
print_row('GP uncertainty max', 'sigma_max');
print_row('GP uncertainty mean', 'sigma_mean');
print_row('max correction norm |u|', 'max_correction_norm');
print_row('obstacle rollout min RAW h', 'obstacle_rollout_h_min');
print_row('obstacle max slack', 'obstacle_max_slack');
print_row('curvature smoothness CS', 'cs');
print_row('acceleration smoothness AS', 'as');
fprintf('%s\n', repmat('-', 1, 62));
fprintf(['注: h_min/continuous_h_min 为原始 h(与 QP slack 无关); ', ...
    'continuous_h_min<0 表示"点在外但连线穿障"。\n']);
