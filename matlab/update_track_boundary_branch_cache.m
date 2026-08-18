function cache = update_track_boundary_branch_cache(constraint_cfg, x)
%UPDATE_TRACK_BOUNDARY_BRANCH_CACHE Commit phase history after one RK4 step.
% Branch limits and nominal phases stay fixed. Only the soft temporal
% reference is updated after k1/k2/k3/k4 have used the same previous value.

cache = struct_field_default(constraint_cfg, ...
    'track_boundary_reference_cache', struct([]));
if isempty(cache) || ...
        ~struct_field_default(constraint_cfg, 'track_boundary_enabled', false)
    return;
end
geometry = constraint_cfg.track_boundary_geometry;
point_maps = constraint_cfg.track_boundary_point_maps;
if numel(cache) ~= numel(point_maps)
    error('Track boundary reference cache count must match point maps.');
end
for map_idx = 1:numel(point_maps)
    if ~strcmpi(struct_field_default(cache(map_idx), ...
            'matching_method', 'fixed_window'), 'branch_soft')
        continue;
    end
    p = point_maps(map_idx).M * x(:) + point_maps(map_idx).o;
    cache(map_idx).s_previous = nearest_parameter_in_branch( ...
        p, geometry, cache(map_idx));
end
end

function s_near = nearest_parameter_in_branch(p, geometry, cache)
scale = max(cache.phase_scale, eps);
objective_grid = (cache.x_grid - p(1)) .^ 2 + ...
    (cache.y_grid - p(2)) .^ 2 + ...
    cache.reference_weight .* ((cache.s_grid - cache.s_nominal) ./ scale) .^ 2 + ...
    cache.previous_weight .* ((cache.s_grid - cache.s_previous) ./ scale) .^ 2;
[~, nearest_idx] = min(objective_grid);
lo = cache.s_grid(max(nearest_idx - 1, 1));
hi = cache.s_grid(min(nearest_idx + 1, numel(cache.s_grid)));
objective = @(s) (ppval(geometry.center_pp_x, s) - p(1)) .^ 2 + ...
    (ppval(geometry.center_pp_y, s) - p(2)) .^ 2 + ...
    cache.reference_weight .* ((s - cache.s_nominal) ./ scale) .^ 2 + ...
    cache.previous_weight .* ((s - cache.s_previous) ./ scale) .^ 2;
if hi > lo
    s_near = fminbnd(objective, lo, hi, center_search_options());
else
    s_near = lo;
end
span = max(cache.s_max - cache.s_min, eps);
face_tol = 1e-6 * span;
if s_near - cache.s_min <= face_tol
    s_near = cache.s_min;
elseif cache.s_max - s_near <= face_tol
    s_near = cache.s_max;
end
end

function options = center_search_options()
persistent cached_options
if isempty(cached_options)
    cached_options = optimset('TolX', 1e-12);
end
options = cached_options;
end
