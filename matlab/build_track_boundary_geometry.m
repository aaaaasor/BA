function geometry = build_track_boundary_geometry(track_segment, ...
    n_control_points, spline_type)
%BUILD_TRACK_BOUNDARY_GEOMETRY Build left/right parametric boundary splines.
% Each boundary uses n_control_points knots, hence n_control_points-1 spline
% intervals. The center spline is retained only to select the inward side.
%
% spline_type:
%   'pchip'  - 分段三次 Hermite, 保形, 但只有 C1: c'' 在结点处跳变(实测中位
%              1.489, 最大 696)。
%   'spline' - not-a-knot 三次样条, C2: c'' 连续(跳变降到 3.7e-4)。
% 必须用 'spline': track_boundary_cbf_info 的梯度链式项需要 c'', 而 ds*/dp 的
% 分母 |c'|^2-(p-c)·c'' 在 pchip 下走廊内最小值为 -2.479(变号, 导数发散),
% spline 下为 +0.860。实测两者对原始折线的最大偏离只差 1 cm, 所以没有代价。

if nargin < 2 || isempty(n_control_points)
    n_control_points = 400;
end
if nargin < 3 || isempty(spline_type)
    spline_type = 'spline';
end
switch lower(string(spline_type))
    case "pchip"
        fit_curve = @pchip;
    case "spline"
        fit_curve = @spline;
    otherwise
        error('spline_type must be ''pchip'' or ''spline''.');
end
geometry.spline_type = char(lower(string(spline_type)));
if ~isscalar(n_control_points) || n_control_points < 4 || ...
        n_control_points ~= round(n_control_points)
    error('n_control_points must be an integer >= 4.');
end
required = {'left', 'right', 'center'};
if ~all(isfield(track_segment, required))
    error('track_segment must contain left, right, and center curves.');
end
n_track = size(track_segment.center, 1);
if n_track < n_control_points
    error('Track has %d points but %d boundary controls were requested.', ...
        n_track, n_control_points);
end

control_indices = round(linspace(1, n_track, n_control_points));
s_control = linspace(0.0, 1.0, n_control_points);
center_control = track_segment.center(control_indices, :);
geometry.n_control_points = n_control_points;
geometry.n_intervals = n_control_points - 1;
geometry.control_indices = control_indices;
geometry.s_control = s_control;
geometry.center_pp_x = fit_curve(s_control, center_control(:, 1)');
geometry.center_pp_y = fit_curve(s_control, center_control(:, 2)');
% 一阶/二阶导用于 CBF 梯度里的链式项 ds*/dp。PCHIP 在每个开区间内
% 二次可导；结点附近由 track_boundary_cbf_info 的灵敏度保护处理。
geometry.center_dpp_x = pp_derivative(geometry.center_pp_x);
geometry.center_dpp_y = pp_derivative(geometry.center_pp_y);
geometry.center_ddpp_x = pp_derivative(geometry.center_dpp_x);
geometry.center_ddpp_y = pp_derivative(geometry.center_dpp_y);

names = {'left', 'right'};
curves = repmat(struct(), 2, 1);
for curve_idx = 1:2
    points = track_segment.(names{curve_idx})(control_indices, :);
    curves(curve_idx).name = names{curve_idx};
    curves(curve_idx).control_points = points;
    curves(curve_idx).pp_x = fit_curve(s_control, points(:, 1)');
    curves(curve_idx).pp_y = fit_curve(s_control, points(:, 2)');
    curves(curve_idx).dpp_x = pp_derivative(curves(curve_idx).pp_x);
    curves(curve_idx).dpp_y = pp_derivative(curves(curve_idx).pp_y);
    distance_grid = sort([s_control, ...
        0.5 * (s_control(1:end-1) + s_control(2:end))]);
    curves(curve_idx).distance_s_grid = distance_grid;
    curves(curve_idx).distance_points = [ ...
        ppval(curves(curve_idx).pp_x, distance_grid); ...
        ppval(curves(curve_idx).pp_y, distance_grid)];
end
geometry.curves = curves;
geometry.implicit_fields = build_global_implicit_fields(curves);
end

function fields = build_global_implicit_fields(curves)
% Precompute two fixed functions h1(x,y), h2(x,y).  Each function is the
% signed distance to one rail.  Its inward orientation is obtained from the
% opposite rail at the same construction parameter; the centerline is not
% used.  Finite differences of the completed scalar fields provide gradients
% that are consistent with the h values used by the CBF.
grid_size = 401;
padding = 0.20;
all_points = [curves(1).distance_points, curves(2).distance_points];
x_min = min(all_points(1, :)) - padding;
x_max = max(all_points(1, :)) + padding;
y_min = min(all_points(2, :)) - padding;
y_max = max(all_points(2, :)) + padding;
x_grid = linspace(x_min, x_max, grid_size);
y_grid = linspace(y_min, y_max, grid_size);
[grid_x, grid_y] = meshgrid(x_grid, y_grid);
query_points = [grid_x(:), grid_y(:)];

fields.x_min = x_min;
fields.y_min = y_min;
fields.dx = x_grid(2) - x_grid(1);
fields.dy = y_grid(2) - y_grid(1);
fields.n_x = grid_size;
fields.n_y = grid_size;
% Only the value grid is stored.  evaluate_track_implicit_field differentiates
% the bilinear interpolant analytically; a separately stored finite-difference
% gradient grid is not the derivative of that interpolant and reintroduces the
% value/gradient inconsistency this field was meant to avoid.
fields.h = cell(2, 1);

for boundary_idx = 1:2
    rail = curves(boundary_idx).distance_points';
    opposite_idx = 3 - boundary_idx;
    opposite = curves(opposite_idx).distance_points';
    [~, nearest_idx] = nearest_sample_distance(query_points, rail);
    % Distance to the nearest *sample* overestimates the distance to the rail
    % itself by up to half the sample spacing.  With 1.131 m median spacing
    % that is 0.255 m at 0.5 m from the rail, i.e. 27% of the 0.936 m margin,
    % and the bias grows exactly where accuracy matters most (close to the
    % wall).  Measured effect: h was reported 0.07..0.28 m safer than the
    % truth, so the CBF pulled with only ~66% of the required strength.
    % Refining onto the two segments adjacent to the nearest sample gives the
    % exact distance to the polyline at no extra search cost.
    distance = polyline_distance(query_points, rail, nearest_idx);
    inward = opposite(nearest_idx, :) - rail(nearest_idx, :);
    inward_norm = sqrt(sum(inward .^ 2, 2));
    inward = inward ./ max(inward_norm, eps);
    offset = query_points - rail(nearest_idx, :);
    side = sign(sum(offset .* inward, 2));
    side(side == 0) = 1;
    fields.h{boundary_idx} = reshape(side .* distance, grid_size, grid_size);
end
end

function distance = polyline_distance(query, rail, nearest_idx)
% Exact distance to the rail polyline.  The closest point of a polyline to a
% query always lies on one of the two segments touching the closest vertex,
% so only those two need to be checked.
n_rail = size(rail, 1);
distance = inf(size(query, 1), 1);
for segment_offset = [-1, 0]
    first_idx = min(max(nearest_idx + segment_offset, 1), n_rail - 1);
    segment_start = rail(first_idx, :);
    segment_vector = rail(first_idx + 1, :) - segment_start;
    length_sq = max(sum(segment_vector .^ 2, 2), eps);
    projection = sum((query - segment_start) .* segment_vector, 2) ./ length_sq;
    projection = min(max(projection, 0), 1);
    foot = segment_start + projection .* segment_vector;
    distance = min(distance, sqrt(sum((query - foot) .^ 2, 2)));
end
end

function [distance, nearest_idx] = nearest_sample_distance(query, samples)
% Chunked nearest-neighbour calculation avoids a large temporary matrix and
% does not require Statistics and Machine Learning Toolbox.
n_query = size(query, 1);
nearest_idx = zeros(n_query, 1);
distance_sq = inf(n_query, 1);
chunk_size = 2000;
sample_norm_sq = sum(samples .^ 2, 2)';
for first_idx = 1:chunk_size:n_query
    last_idx = min(first_idx + chunk_size - 1, n_query);
    q = query(first_idx:last_idx, :);
    pair_distance_sq = sum(q .^ 2, 2) + sample_norm_sq - 2 * q * samples';
    [local_distance_sq, local_idx] = min(pair_distance_sq, [], 2);
    distance_sq(first_idx:last_idx) = max(local_distance_sq, 0);
    nearest_idx(first_idx:last_idx) = local_idx;
end
distance = sqrt(distance_sq);
end
