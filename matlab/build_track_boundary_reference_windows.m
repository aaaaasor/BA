function [s_min_targets, s_max_targets] = ...
    build_track_boundary_reference_windows(n_rollout_samples, n_segments, ...
    n_points_per_segment, apply_points)
%BUILD_TRACK_BOUNDARY_REFERENCE_WINDOWS Longitudinal bins for local points.
% Rollout rows are ordered by complete trajectory, then segment.  Each
% generated point owns the half-step interval around its expected global
% centerline parameter, preventing a folded track's other branches from
% being selected merely because they are spatially close.

if mod(n_rollout_samples, n_segments) ~= 0
    error('Rollout sample count must be divisible by the segment count.');
end
if n_segments < 1 || n_points_per_segment < 2
    error('n_segments and n_points_per_segment must be positive.');
end
n_maps = numel(apply_points);
s_min_targets = zeros(n_rollout_samples, n_maps);
s_max_targets = zeros(n_rollout_samples, n_maps);
global_step = 1.0 / (n_segments * (n_points_per_segment - 1));
for sample_idx = 1:n_rollout_samples
    segment_idx = mod(sample_idx - 1, n_segments) + 1;
    for map_idx = 1:n_maps
        point_idx = apply_points(map_idx);
        global_point_idx = (segment_idx - 1) * ...
            (n_points_per_segment - 1) + (point_idx - 1);
        s_reference = global_point_idx * global_step;
        s_min_targets(sample_idx, map_idx) = max( ...
            0.0, s_reference - 0.5 * global_step);
        s_max_targets(sample_idx, map_idx) = min( ...
            1.0, s_reference + 0.5 * global_step);
    end
end
end
