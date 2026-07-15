function report = evaluate_trajectory_obstacle_safety(points, obstacle)
% 对最终生成曲线做障碍安全评价(fix 1/4): 不依赖 QP slack，直接量原始 h。
%   points : n_points x n_samples x feature_dim (物理坐标, 前 2 列为位置)
%   obstacle: 含 centers (2 x n_obs) / semi_axes (2 x n_obs)
% 输出 report:
%   discrete_h_min          所有离散点的最小 h (h<0 = 点在障碍内)
%   continuous_h_min        所有相邻线段的连续最小 h (解析求二次极小)
%   n_crossing_segments     连续 h<0 的线段总数(线段穿过椭圆)
%   n_crossing_samples      至少有一段穿障的样本数
%   per_sample_continuous_h_min  每个样本的连续最小 h
% 说明: 相邻点取"样本内点序"(第三层端点已被硬覆盖成共享 L2 锚点，
%       段边界点重合，连成折线合理)。这补上了"点在外、连线仍穿障"的评价。
report.discrete_h_min = inf;
report.continuous_h_min = inf;
report.n_crossing_segments = 0;
report.n_crossing_samples = 0;
report.per_sample_continuous_h_min = [];
if isempty(obstacle) || ~isfield(obstacle, 'centers')
    return;
end
n_samples = size(points, 2);
report.per_sample_continuous_h_min = inf(n_samples, 1);
for s = 1:n_samples
    xy = squeeze(points(:, s, 1:2));
    if size(xy, 2) ~= 2
        xy = reshape(xy, [], 2);
    end
    n_pts = size(xy, 1);
    sample_cont_hmin = inf;
    sample_has_crossing = false;
    for i = 1:n_pts
        report.discrete_h_min = min(report.discrete_h_min, ...
            local_point_min_h(xy(i, :)', obstacle));
    end
    for i = 1:(n_pts - 1)
        seg_hmin = local_segment_min_h(xy(i, :)', xy(i + 1, :)', obstacle);
        sample_cont_hmin = min(sample_cont_hmin, seg_hmin);
        if seg_hmin < 0
            report.n_crossing_segments = report.n_crossing_segments + 1;
            sample_has_crossing = true;
        end
    end
    report.per_sample_continuous_h_min(s) = sample_cont_hmin;
    report.continuous_h_min = min(report.continuous_h_min, sample_cont_hmin);
    if sample_has_crossing
        report.n_crossing_samples = report.n_crossing_samples + 1;
    end
end
end

function hmin = local_point_min_h(p, obstacle)
hmin = inf;
for j = 1:size(obstacle.centers, 2)
    c = obstacle.centers(:, j);
    a = obstacle.semi_axes(1, j);
    b = obstacle.semi_axes(2, j);
    d = p - c;
    hmin = min(hmin, (d(1) / a)^2 + (d(2) / b)^2 - 1);
end
end

function hmin = local_segment_min_h(pa, pb, obstacle)
% 每个椭圆: h(s) = d(s)'Q d(s) - 1, d(s) = pa + s(pb-pa) - c, s in [0,1]。
% 二次: h(s) = A s^2 + B s + C, 极小在顶点 s*=-B/(2A)(裁剪[0,1])或端点。
hmin = inf;
for j = 1:size(obstacle.centers, 2)
    c = obstacle.centers(:, j);
    a = obstacle.semi_axes(1, j);
    b = obstacle.semi_axes(2, j);
    Q = diag([1 / a^2, 1 / b^2]);
    d0 = pa - c;
    dd = pb - pa;
    A = dd' * Q * dd;
    B = 2 * (d0' * Q * dd);
    C = d0' * Q * d0 - 1;
    cand = [0; 1];
    if A > eps
        s_star = -B / (2 * A);
        if s_star > 0 && s_star < 1
            cand(end + 1) = s_star; %#ok<AGROW>
        end
    end
    for k = 1:numel(cand)
        s = cand(k);
        hmin = min(hmin, A * s^2 + B * s + C);
    end
end
end
