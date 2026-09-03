function compare_u_freeze(n_samples)
%COMPARE_U_FREEZE 我们的方法：每级解 u vs 每步解一次 u（SafeFlow Alg.1 口径）
%
% 同一个模型、同一批初始状态，只改 rk4_rollout 的 u_per_stage。
% 在第一层上做，因为它约束齐全（PTCLF + 几何安全 + 方差）且最快。
% 种子筛选关闭：否则两种配置会接受不同的种子，比较就没有意义了。
%
% 关心三件事：
%   1. rollout 耗时降多少
%   2. 轨迹是否发生可察觉的改变
%   3. 几何安全与终点是否退化（切换点粗化到外层网格的代价）

if nargin < 1 || isempty(n_samples), n_samples = 20; end
this_dir = fileparts(mfilename('fullpath'));

%% ---------- 复刻 main_demo 的第一层准备（全部走缓存）----------
cfg = get_config();
rng(cfg.random_seed);
[first_level_target_points, track_segment] = scenario_training_points(cfg, 5, cfg.n_train);
cfg.track_segment = track_segment;
cfg.obstacle = configure_racing_obstacles(track_segment, cfg.obstacle);
cfg.track_boundary.geometry = build_track_boundary_geometry(track_segment, ...
    struct_field_default(cfg.track_boundary, 'n_spline_points', 400), ...
    struct_field_default(cfg.track_boundary, 'spline_type', 'spline'));

rng(cfg.first_level_data_seed);
[s_slices, x_slices, y_slices, target_points, ~, ~, source_data, ~, ...
    data_transform] = build_training_data(cfg.t_min, 1.0, ...
    cfg.n_time_slices, first_level_target_points);
first_level_feature_dim = size(target_points, 3);

rng(cfg.first_level_hyperparameter_seed);
first_gp = cfg.gp;
first_gp.o_ratio = cfg.gp.first_level_o_ratio;
first_gp.hyperparameter_mat_path = cfg.cache.first_level_hyperparameter_path;
first_gp.n_pretrain = cfg.gp.first_level_n_pretrain;
first_gp = optimize_gp_hyperparameters(x_slices, y_slices, first_gp, s_slices);
first_gp.training_accuracy_threshold = cfg.gp.first_level_training_accuracy_threshold;

rng(cfg.first_level_fit_seed);
model_collection = fit_or_load_loggp_model(s_slices, x_slices, y_slices, ...
    first_gp, fullfile(this_dir, cfg.cache.first_level_model_path), 'first-level');
model_collection = strip_model_for_prediction(model_collection, 'First-level');

n_rows = size(source_data, 1);
n_points_first = n_rows / first_level_feature_dim;
ccfg0 = make_level_variance_constraint(cfg, 'first_level');
if ccfg0.obstacle_enabled
    ccfg0.obstacle_point_maps = build_obstacle_point_maps(data_transform, ...
        first_level_feature_dim, n_points_first, ccfg0.obstacle_points, 'absolute');
end
if ccfg0.track_boundary_enabled
    ccfg0.track_boundary_point_maps = build_obstacle_point_maps(data_transform, ...
        first_level_feature_dim, n_points_first, ccfg0.track_boundary_points, 'absolute');
    [ccfg0.track_boundary_reference_s_min_targets, ...
     ccfg0.track_boundary_reference_s_max_targets] = ...
        build_track_boundary_reference_windows(n_samples, 1, n_points_first, ...
        ccfg0.track_boundary_points);
end

rng(cfg.first_level_rollout_seed);
x_init = randn(n_samples, n_rows);
fprintf('第一层: %d 条 x %d 维, %d 步\n', n_samples, n_rows, ...
    cfg.first_level_time_steps);

%% ---------- 两种配置 ----------
% name, u_per_stage, u_freeze_before_time
variants = { 'per_stage', true,  0.0      % 现状：每级解 u
             'per_step',  false, 0.0      % 全程冻结（SafeFlow Alg.1 口径）
             'hybrid80',  true,  0.80     % t<0.80 冻结，末段逐级重解
             'hybrid90',  true,  0.90 };

res = struct();
for vi = 1:size(variants,1)
    v = variants{vi,1};
    ccfg = ccfg0;
    ccfg.u_per_stage         = variants{vi,2};
    ccfg.u_freeze_before_time = variants{vi,3};
    fprintf('\n===== %s (u_per_stage=%d, freeze_before=%.2f) =====\n', ...
        v, ccfg.u_per_stage, ccfg.u_freeze_before_time);
    t0 = tic;
    [times, path] = rk4_rollout(model_collection, x_init, cfg.t_min, ...
        cfg.rollout_t_max, cfg.first_level_time_steps, ccfg, [], []);
    secs = toc(t0);
    res.(v) = struct('times', times, 'path', path, 'seconds', secs);
    fprintf('  rollout %.2f s  (%.4f s/条)\n', secs, secs/n_samples);
end

%% ---------- 对比：全部以 per_stage 为基准 ----------
A  = res.per_stage.path;
sa = res.per_stage.seconds;
fprintf('\n%-10s %11s %11s %11s %9s %8s %11s %11s\n', '配置', ...
    '相对偏差均值', '相对偏差最大', '终点位移均值', '耗时(s)', '加速', ...
    'min h 障碍', 'min h 边界');
for vi = 1:size(variants,1)
    v = variants{vi,1};
    B = res.(v).path;
    d = zeros(n_samples,1); df = zeros(n_samples,1); sc = zeros(n_samples,1);
    for i = 1:n_samples
        Ai = squeeze(A(:,i,:));  Bi = squeeze(B(:,i,:));
        d(i)  = max(abs(Ai - Bi), [], 'all');
        df(i) = norm(Ai(end,:) - Bi(end,:));
        sc(i) = max(abs(Ai), [], 'all');
    end
    [sr, ho, hb] = geom_safety(B, data_transform, ...
        first_level_feature_dim, n_points_first, cfg);
    res.(v).safety = sr;  res.(v).rel_dev = d./sc;  res.(v).final_dev = df;
    fprintf('%-10s %10.3f%% %10.3f%% %11.4e %9.2f %7.2fx %+11.6f %+11.6f\n', ...
        v, 100*mean(d./sc), 100*max(d./sc), mean(df), ...
        res.(v).seconds, sa/res.(v).seconds, ho, hb);
end
fprintf('\nSafety (真实边界, 无 margin): ');
for vi = 1:size(variants,1)
    fprintf('%s %.1f%%  ', variants{vi,1}, 100*res.(variants{vi,1}).safety);
end
fprintf('\n');

save(fullfile(this_dir, 'outputs', 'Racing_UFreeze_Comparison.mat'), ...
    'res', 'x_init', 'variants', 'n_samples', '-v7.3');
fprintf('\n已保存 outputs/Racing_UFreeze_Comparison.mat\n');
end

% =====================================================================
function [rate, hmin_o, hmin_b] = geom_safety(path, data_transform, fdim, nPt, cfg)
n = size(path, 2);
ok = true(1,n); hmin_o = inf; hmin_b = inf;
fields = cfg.track_boundary.geometry.implicit_fields;
for i = 1:n
    xs = squeeze(path(end, i, :));
    pts = decode_points(xs, data_transform, fdim, nPt);
    for k = 1:nPt
        p = pts(:,k);
        for jo = 1:size(cfg.obstacle.centers, 2)
            h = obstacle_level_and_gradient(p, cfg.obstacle, jo);
            hmin_o = min(hmin_o, h);
            if h < -1e-8, ok(i) = false; end
        end
        for bi = 1:2
            hr = evaluate_track_implicit_field(fields, bi, p);
            hmin_b = min(hmin_b, hr);
            if hr < -1e-8, ok(i) = false; end
        end
    end
end
rate = mean(ok);
end

function pts = decode_points(xs, data_transform, fdim, nPt)
% 与 build_obstacle_point_maps 的 'absolute' 模式同一口径: p = x.*std + mean，
% 行序为 feature 最快（[f1p1; f2p1; ...; f1p2; ...]）。
x_abs = xs(:) .* data_transform.std(:) + data_transform.mean(:);
M = reshape(x_abs, fdim, nPt);
pts = M(1:2, :);
end
