%EXTRACT_TRACK_SEGMENT 从赛道数据库中截取并规范化一段赛道几何。

function segment = extract_track_segment(track_csv_path, raceline_csv_path, ...
    opts)
if nargin < 3 || isempty(opts)
    opts = struct();
end
% 按中心线弧长截取赛段。
if ~isfield(opts, 's_range_m') || isempty(opts.s_range_m)
    error('extract_track_segment:missingRange', ...
        'opts.s_range_m is required, e.g. struct(''s_range_m'', [250, 1150]).');
end
s_range_m = opts.s_range_m;
% 重采样点数。
n_resample = struct_field_default(opts, 'n_resample', 400);

%% Load Raw Data
% 读取赛车线和中心线。
raceline = read_racetrack_csv(raceline_csv_path, {'x_m', 'y_m'});
track = read_racetrack_csv(track_csv_path, ...
    {'x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'});

center_all = [track.x_m(:), track.y_m(:)];
raceline_all = [raceline.x_m(:), raceline.y_m(:)];
w_left_all = track.w_tr_left_m(:);
w_right_all = track.w_tr_right_m(:);

s_rl = cumulative_arclength(center_all);
if numel(s_range_m) ~= 2 || s_range_m(2) <= s_range_m(1)
    error('extract_track_segment:badRange', ...
        'opts.s_range_m must be [s_start, s_stop] with s_stop > s_start.');
end
s_start = max(s_range_m(1), s_rl(1));
s_stop = min(s_range_m(2), s_rl(end));

%% Cut Centerline, Project Raceline Onto The Same Span
% 截取中心线和赛车线。
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
% 重采样并重建左右边界。
s_center_raw = cumulative_arclength(center_xy);
[center_xy, s_center] = resample_by_arclength(center_xy, n_resample);
w_left = interp1(s_center_raw, w_left, s_center, 'linear', 'extrap');
w_right = interp1(s_center_raw, w_right, s_center, 'linear', 'extrap');

tangent = gradient_rows(center_xy);
tangent = tangent ./ max(vecnorm(tangent, 2, 2), eps);
normal_left = [-tangent(:, 2), tangent(:, 1)];  % 左法向（逆时针 90 度）

left_xy = center_xy + normal_left .* w_left;
right_xy = center_xy - normal_left .* w_right;

raceline_xy = resample_by_arclength(raceline_xy, n_resample);

%% Similarity Transform Into The Unit Square
% 等比例缩放到单位方框。
all_xy = [center_xy; left_xy; right_xy; raceline_xy];
min_xy = min(all_xy, [], 1);
max_xy = max(all_xy, [], 1);
scale = 1.0 / max(max_xy - min_xy);   % 统一缩放，不改变形状
offset = -min_xy * scale;
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
segment.meta.s_range_m = [s_start, s_stop];
segment.meta.length_m = s_stop - s_start;
end

%% ------------------------------------------------------------------------
function data = read_racetrack_csv(csv_path, column_names)
% 读取赛道 CSV 数值列。
if ~isfile(csv_path)
    error('extract_track_segment:missingFile', 'File not found: %s', csv_path);
end
raw = fileread(csv_path);
lines = strsplit(raw, {sprintf('\r\n'), newline});
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

function s = cumulative_arclength(xy)
% 计算累计弧长。
ds = vecnorm(diff(xy, 1, 1), 2, 2);
s = [0; cumsum(ds)];
end

function [xy_out, s_out] = resample_by_arclength(xy, n_points)
% 按弧长重采样二维曲线。
s = cumulative_arclength(xy);
s_out = linspace(0, s(end), n_points)';
xy_out = [interp1(s, xy(:, 1), s_out, 'pchip'), ...
    interp1(s, xy(:, 2), s_out, 'pchip')];
end
