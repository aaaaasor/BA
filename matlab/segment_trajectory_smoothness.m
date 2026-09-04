function [cs_values, as_values, details] = segment_trajectory_smoothness( ...
    segment_data, n_segments, n_points, junction_tolerance)
% CS/AS on the reconstructed trajectory, in trajectory/segment order.
% Adjacent segments share one waypoint, so the duplicate copy is merged and
% each trajectory contributes n_segments*(n_points-1)+1 distinct points --
% the same 65-point reconstruction the Safety metric and downstream code
% consume.
%
% Junction handling: merging is now the default (junction_tolerance = inf).
% Keeping both copies inserted an edge of the junction gap length between
% two normal edges; because that gap is orders of magnitude shorter than the
% regular spacing, its direction is essentially noise and the two angles it
% creates dominated CS (measured: 0.4019 with both copies against 0.0711
% merged, i.e. ~93% of the reported value came from junction artefacts).
% It also made CS incomparable with single-shot baselines such as SafeFlow
% and UniConFlow, which have no junctions at all. The discontinuity is still
% measured and returned in details.junction_gaps, so nothing is hidden -- it
% is reported as its own quantity instead of being folded into a curvature
% metric.
% segment_data: one segment per row, interleaved point features [x y ...].
if nargin < 4
    junction_tolerance = inf;
end
validateattributes(segment_data, {'numeric'}, {'2d', 'nonempty', 'finite'});
validateattributes(n_segments, {'numeric'}, {'scalar', 'integer', 'positive'});
validateattributes(n_points, {'numeric'}, {'scalar', 'integer', '>=', 2});
validateattributes(junction_tolerance, {'numeric'}, {'scalar', 'nonnegative'});
assert(mod(size(segment_data, 1), n_segments) == 0, ...
    'Segment rows must contain complete parent trajectories.');
feature_dim = size(segment_data, 2) / n_points;
assert(feature_dim >= 2 && feature_dim == floor(feature_dim), ...
    'Each generated point must have at least x-y features.');
n_trajectories = size(segment_data, 1) / n_segments;
cs_values = nan(n_trajectories, 1);
as_values = nan(n_trajectories, 1);
point_counts = zeros(n_trajectories, 1);
junction_gaps = zeros(n_trajectories, n_segments - 1);
for ti = 1:n_trajectories
    xy = zeros(0, 2);
    for si = 1:n_segments
        row = (ti - 1) * n_segments + si;
        curve = reshape(segment_data(row, :), feature_dim, n_points)';
        curve = curve(:, 1:2);
        if si > 1
            gap = norm(curve(1, :) - xy(end, :));
            junction_gaps(ti, si - 1) = gap;
            if gap <= junction_tolerance
                curve = curve(2:end, :);
            end
        end
        xy = [xy; curve]; %#ok<AGROW>
    end
    point_counts(ti) = size(xy, 1);
    w = diff(xy, 1, 1);
    if size(w, 1) >= 2
        left = w(1:end-1, :);
        right = w(2:end, :);
        denominator = sqrt(sum(left.^2, 2)) .* sqrt(sum(right.^2, 2));
        valid = denominator > eps;
        cos_theta = ones(size(denominator));
        cos_theta(valid) = sum(left(valid, :) .* right(valid, :), 2) ./ denominator(valid);
        cos_theta = min(max(cos_theta, -1), 1);
        cs_values(ti) = mean(1 - cos_theta);
        acceleration = diff(xy, 2, 1);
        as_values(ti) = mean(sqrt(sum(acceleration.^2, 2)));
    end
end
details = struct( ...
    'definition', ['segment-order physical x-y points, including unmatched ', ...
    'junction endpoints and gap steps; only coincident junctions merged; ', ...
    'CS=mean(1-cos(theta)); AS=mean(norm(second difference))'], ...
    'junction_tolerance', junction_tolerance, ...
    'input_points_per_trajectory', n_segments * n_points, ...
    'points_per_trajectory', point_counts, ...
    'junction_gaps', junction_gaps, ...
    'cs_per_trajectory', cs_values, ...
    'as_per_trajectory', as_values);
end
