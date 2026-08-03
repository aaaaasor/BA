%EXTRACT_TRACK_SEGMENT 从赛道数据库中截取并规范化一段赛道几何。
%
%   segment = EXTRACT_TRACK_SEGMENT(track_csv_path, raceline_csv_path, opts)
%
% 原始数据（TUMFTM/racetrack-database）：
%   tracks/Nuerburgring.csv    -> x_m, y_m, w_tr_right_m, w_tr_left_m
%   racelines/Nuerburgring.csv -> x_m, y_m   (只有坐标，没有 s/psi/kappa/vx)
% 注意该文件是 5.1 km 的 GP-Strecke，不是 20.8 km 的 Nordschleife。
%
% 赛段有两种选择方式：
%   1. opts.s_range_m 非空：直接按中心线累计弧长截取 [起点,终点]；
%   2. opts.s_range_m 为空：平滑中心线曲率、提取左右弯序列，再匹配
%      opts.pattern（例如 'RLR'）。
%
% 输入：
%   track_csv_path    - 中心线及左右半宽 CSV 文件。
%   raceline_csv_path - 赛车线坐标 CSV 文件。
%   opts              - 可选参数结构体，字段说明见下方默认参数。
%
% 输出 segment：
%   center/right/left/raceline - 归一化后的各条二维曲线；
%   tangent                    - 中心线单位切向量；
%   half_width_left/right      - 归一化后的左右半宽；
%   s_center                   - 归一化中心线累计弧长；
%   transform                  - 米制坐标与归一化坐标之间的相似变换；
%   meta                       - 实际截取范围、长度和弯道序列。
%
% 坐标变换对 x、y 使用同一个比例，因此不会改变赛道的角度和长宽比例。
function segment = extract_track_segment(track_csv_path, raceline_csv_path, ...
    opts)
if nargin < 3 || isempty(opts)
    opts = struct();
end
% 直接给定弧长区间 [s_start, s_stop]（米）时跳过模式匹配。模式匹配只在
% 没给区间时用来自动定位（pattern 是按行驶方向的弯序列，如 'RLR'）。
s_range_m = struct_field_default(opts, 's_range_m', []);
pattern = struct_field_default(opts, 'pattern', 'RLR');
% 曲率平滑窗口（单位：米弧长）。该数据集采样约 5 m/点。
smooth_length_m = struct_field_default(opts, 'smooth_length_m', 25.0);
% 死区：|kappa| 小于该值视为直道，单位 rad/m。1/200 m 半径 = 0.005。
kappa_deadband = struct_field_default(opts, 'kappa_deadband', 0.004);
% 忽略短于该弧长的弯/直段（游程压缩前的去噪）。
min_run_length_m = struct_field_default(opts, 'min_run_length_m', 30.0);
% 匹配段前后各留出的直道余量，单位米。
margin_m = struct_field_default(opts, 'margin_m', 60.0);
% 归一化后段内重采样点数。
n_resample = struct_field_default(opts, 'n_resample', 400);
% 第几个匹配（同一条赛道可能有多处 LRR）。
match_index = struct_field_default(opts, 'match_index', 1);

%% Load Raw Data
% CSV 中以 # 开头的说明行由 read_racetrack_csv 跳过；这里只取所需列。
raceline = read_racetrack_csv(raceline_csv_path, {'x_m', 'y_m'});
track = read_racetrack_csv(track_csv_path, ...
    {'x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'});

center_all = [track.x_m(:), track.y_m(:)];
raceline_all = [raceline.x_m(:), raceline.y_m(:)];
w_left_all = track.w_tr_left_m(:);
w_right_all = track.w_tr_right_m(:);

% 数据里没有曲率列，从中心线几何计算。闭环填充用于避免整圈起终点处
% 因有限差分缺少邻点而出现异常曲率。
s_rl = cumulative_arclength(center_all);
kappa = closed_loop_curvature(center_all);

if ~isempty(s_range_m)
    s_start = max(s_range_m(1), s_rl(1));
    s_stop = min(s_range_m(2), s_rl(end));
    corner_only = '';
    matched_by_pattern = false;
else
    matched_by_pattern = true;
end

%% Curvature -> Corner Sequence
% 将米制平滑长度换算为采样点窗口，并把小曲率区域归为直道 0。
ds_mean = mean(diff(s_rl));
smooth_window = max(round(smooth_length_m / ds_mean), 1);
kappa_smooth = movmean(kappa, smooth_window, 'Endpoints', 'shrink');

sign_seq = zeros(size(kappa_smooth));
sign_seq(kappa_smooth > kappa_deadband) = 1;    % left
sign_seq(kappa_smooth < -kappa_deadband) = -1;  % right

[run_values, run_starts, run_stops] = run_length_encode(sign_seq);
run_lengths_m = s_rl(run_stops) - s_rl(run_starts);

% 短游程通常来自曲率在阈值附近抖动，将其并入相邻保留游程以去除伪弯。
keep = run_lengths_m >= min_run_length_m;
keep(1) = true;
[run_values, run_starts, run_stops] = merge_short_runs( ...
    run_values, run_starts, run_stops, keep);

corner_labels = repmat('S', 1, numel(run_values));
corner_labels(run_values == 1) = 'L';
corner_labels(run_values == -1) = 'R';

%% Pattern Match
% 去掉直道 S 后形成纯弯道字符串，例如 'RLRRL'，再在其中寻找目标模式。
corner_mask = corner_labels ~= 'S';
corner_idx = find(corner_mask);
corner_only = corner_labels(corner_idx);

if matched_by_pattern
    match_positions = strfind(corner_only, pattern);
    if isempty(match_positions)
        error('extract_track_segment:noMatch', ...
            'No occurrence of corner pattern "%s". Corner sequence: %s', ...
            pattern, corner_only);
    end
    if match_index > numel(match_positions)
        error('extract_track_segment:matchIndex', ...
            'Requested match %d but only %d occurrence(s) of "%s" found.', ...
            match_index, numel(match_positions), pattern);
    end

    first_run = corner_idx(match_positions(match_index));
    last_run = corner_idx(match_positions(match_index) + numel(pattern) - 1);
    s_start = max(s_rl(run_starts(first_run)) - margin_m, s_rl(1));
    s_stop = min(s_rl(run_stops(last_run)) + margin_m, s_rl(end));
end

%% Cut Centerline, Project Raceline Onto The Same Span
% 中心线用弧长范围直接索引；赛车线缺少对应 s，因此用欧氏最近点确定端点。
center_idx = find(s_rl >= s_start & s_rl <= s_stop);
center_xy = center_all(center_idx, :);
w_left = w_left_all(center_idx);
w_right = w_right_all(center_idx);

rl_start = nearest_point_index(raceline_all, center_xy(1, :));
rl_stop = nearest_point_index(raceline_all, center_xy(end, :));
if rl_start <= rl_stop
    rl_idx = rl_start:rl_stop;
else
    rl_idx = [rl_start:size(raceline_all, 1), 1:rl_stop];  % 跨过起跑线
end
raceline_xy = raceline_all(rl_idx, :);

%% Arc-Length Reparameterization + Boundary Reconstruction
% 先按弧长等距重采样中心线，再把左右宽度插值到相同弧长位置。
s_center_raw = cumulative_arclength(center_xy);
[center_xy, s_center] = resample_by_arclength(center_xy, n_resample);
w_left = interp1(s_center_raw, w_left, s_center, 'linear', 'extrap');
w_right = interp1(s_center_raw, w_right, s_center, 'linear', 'extrap');

tangent = gradient_rows(center_xy);
tangent = tangent ./ max(vecnorm(tangent, 2, 2), eps);
normal_left = [-tangent(:, 2), tangent(:, 1)];  % 左法向（逆时针 90 度）

% 正宽度沿左法向得到左边界，负宽度沿左法向得到右边界。
left_xy = center_xy + normal_left .* w_left;
right_xy = center_xy - normal_left .* w_right;

raceline_xy = resample_by_arclength(raceline_xy, n_resample);

%% Similarity Transform Into The Unit Square
% 用所有几何元素的联合包围盒计算统一缩放，确保边界和赛车线也落入单位方框。
all_xy = [center_xy; left_xy; right_xy; raceline_xy];
min_xy = min(all_xy, [], 1);
max_xy = max(all_xy, [], 1);
scale = 1.0 / max(max_xy - min_xy);   % 统一缩放，不改变形状
offset = -min_xy * scale;
% 居中：把较短的那一维放到 [0,1] 中间
extent = (max_xy - min_xy) * scale;
offset = offset + (1.0 - extent) / 2.0;

to_unit = @(xy) xy * scale + offset;

segment.center = to_unit(center_xy);
segment.left = to_unit(left_xy);
segment.right = to_unit(right_xy);
segment.raceline = to_unit(raceline_xy);
segment.tangent = tangent;
segment.half_width_left = w_left * scale;
segment.half_width_right = w_right * scale;
segment.s_center = s_center * scale;
segment.transform.scale = scale;
segment.transform.offset = offset;
segment.transform.to_unit = to_unit;
segment.transform.to_metric = @(xy) (xy - offset) / scale;
segment.meta.pattern = pattern;
segment.meta.corner_sequence = corner_only;
segment.meta.s_range_m = [s_start, s_stop];
segment.meta.length_m = s_stop - s_start;
end

%% ------------------------------------------------------------------------
function data = read_racetrack_csv(csv_path, column_names)
%READ_RACETRACK_CSV 读取 racetrack-database 的数值列。
% 数据库文件用 '#' 开头的注释行保存表头，实际数据可能使用逗号或分号。
% column_names 既规定需要读取的前 n 列，也作为返回结构体的字段名。
if ~isfile(csv_path)
    error('extract_track_segment:missingFile', 'File not found: %s', csv_path);
end
raw = fileread(csv_path);
lines = strsplit(raw, {sprintf('\r\n'), sprintf('\n')});
lines = lines(~cellfun(@isempty, strtrim(lines)));
lines = lines(~startsWith(strtrim(lines), '#'));

first = lines{1};
if contains(first, ';')
    delimiter = ';';
else
    delimiter = ',';
end

n_cols = numel(column_names);
values = nan(numel(lines), n_cols);
for line_idx = 1:numel(lines)
    parts = strsplit(strtrim(lines{line_idx}), delimiter);
    if numel(parts) < n_cols
        error('extract_track_segment:columnCount', ...
            'Line %d of %s has %d columns, expected %d.', ...
            line_idx, csv_path, numel(parts), n_cols);
    end
    values(line_idx, :) = str2double(parts(1:n_cols));
end

data = struct();
for col_idx = 1:n_cols
    data.(column_names{col_idx}) = values(:, col_idx);
end
end

function [values, starts, stops] = run_length_encode(seq)
%RUN_LENGTH_ENCODE 对离散符号序列执行游程编码。
% values 保存每段的符号，starts/stops 保存该段在原序列中的闭区间索引。
seq = seq(:);
change_idx = [1; find(diff(seq) ~= 0) + 1];
starts = change_idx;
stops = [change_idx(2:end) - 1; numel(seq)];
values = seq(starts);
end

function [values, starts, stops] = merge_short_runs(values, starts, stops, keep)
%MERGE_SHORT_RUNS 删除 keep=false 的短游程并扩展保留游程的覆盖范围。
% 本函数不重新计算符号，只保证输出游程连续覆盖原序列。被删除的短段会被
% 其前一个保留段吸收；开头游程由调用方强制保留，避免出现无前驱的情况。
keep_idx = find(keep);
new_values = values(keep_idx);
new_starts = starts(keep_idx);
new_stops = stops(keep_idx);
% 被丢弃的游程并入前一个保留游程：直接把 stop 延伸到下一个保留游程之前
new_stops(1:(end - 1)) = new_starts(2:end) - 1;
new_stops(end) = stops(end);
values = new_values;
starts = new_starts;
stops = new_stops;
end

function idx = nearest_point_index(points, query)
%NEAREST_POINT_INDEX 返回二维点集中与 query 欧氏距离最小的行索引。
d2 = sum((points - query) .^ 2, 2);
[~, idx] = min(d2);
end

function kappa = closed_loop_curvature(xy)
%CLOSED_LOOP_CURVATURE 用有限差分估计闭合二维曲线的有符号曲率。
% 首尾各环绕填充 pad 个点，以便赛道起终点也拥有连续邻域。正曲率对应左转，
% 负曲率对应右转；分母加 eps 下界避免直道上数值除零。
pad = 30;
padded = [xy((end - pad + 1):end, :); xy; xy(1:pad, :)];
dx = gradient(padded(:, 1));
dy = gradient(padded(:, 2));
ddx = gradient(dx);
ddy = gradient(dy);
denominator = max((dx .^ 2 + dy .^ 2) .^ 1.5, eps);
kappa = (dx .* ddy - dy .* ddx) ./ denominator;
kappa = kappa((pad + 1):(end - pad));
end

function s = cumulative_arclength(xy)
%CUMULATIVE_ARCLENGTH 计算折线从首点开始的累计欧氏弧长。
ds = vecnorm(diff(xy, 1, 1), 2, 2);
s = [0; cumsum(ds)];
end

function [xy_out, s_out] = resample_by_arclength(xy, n_points)
%RESAMPLE_BY_ARCLENGTH 将二维折线重采样为 n_points 个等弧长点。
% pchip 插值比普通线性插值更平滑，同时保持局部形状、减少弯道过冲。
s = cumulative_arclength(xy);
s_out = linspace(0, s(end), n_points)';
xy_out = [interp1(s, xy(:, 1), s_out, 'pchip'), ...
    interp1(s, xy(:, 2), s_out, 'pchip')];
end

function g = gradient_rows(xy)
%GRADIENT_ROWS 分别沿行索引计算 x、y 坐标的一阶有限差分。
g = [gradient(xy(:, 1)), gradient(xy(:, 2))];
end
