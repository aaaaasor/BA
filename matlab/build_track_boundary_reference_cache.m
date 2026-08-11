function cache = build_track_boundary_reference_cache(constraint_cfg)
%BUILD_TRACK_BOUNDARY_REFERENCE_CACHE Cache sample-constant phase searches.
% The nearest-point minimizer still runs at every velocity-field call. This
% cache only removes repeated spline-grid and local-speed setup, preserving
% the original objective and fminbnd bracket exactly.

cache = struct([]);
if ~struct_field_default(constraint_cfg, 'track_boundary_enabled', false) || ...
        ~isfield(constraint_cfg, 'track_boundary_geometry') || ...
        ~isfield(constraint_cfg, 'track_boundary_point_maps')
    return;
end
geometry = constraint_cfg.track_boundary_geometry;
n_maps = numel(constraint_cfg.track_boundary_point_maps);
reference_s_min = zeros(1, n_maps);
reference_s_max = ones(1, n_maps);
if isfield(constraint_cfg, 'track_boundary_reference_s_min') && ...
        isfield(constraint_cfg, 'track_boundary_reference_s_max')
    reference_s_min = constraint_cfg.track_boundary_reference_s_min(:)';
    reference_s_max = constraint_cfg.track_boundary_reference_s_max(:)';
end
if numel(reference_s_min) ~= n_maps || numel(reference_s_max) ~= n_maps
    error('Track boundary reference window count must match point maps.');
end
phase_search_half_steps = struct_field_default(constraint_cfg, ...
    'track_boundary_phase_search_half_steps', 2.0);
cache = repmat(struct('s_reference', 0, 's_min', 0, 's_max', 1, ...
    's_grid', [], 'x_grid', [], 'y_grid', []), 1, n_maps);
for map_idx = 1:n_maps
    s_reference = 0.5 * (reference_s_min(map_idx) + ...
        reference_s_max(map_idx));
    reference_half_width = max(0.5 * (reference_s_max(map_idx) - ...
        reference_s_min(map_idx)), eps);
    search_half_width = phase_search_half_steps * 2.0 * reference_half_width;
    s_reference = max(0.0, min(1.0, s_reference));
    s_min = max(0.0, s_reference - search_half_width);
    s_max = min(1.0, s_reference + search_half_width);
    in_window = geometry.s_control >= s_min & geometry.s_control <= s_max;
    s_grid = geometry.s_control(in_window);
    if numel(s_grid) < 2
        s_grid = linspace(s_min, s_max, 3);
    end
    x_grid = ppval(geometry.center_pp_x, s_grid);
    y_grid = ppval(geometry.center_pp_y, s_grid);
    cache(map_idx).s_reference = s_reference;
    cache(map_idx).s_min = s_min;
    cache(map_idx).s_max = s_max;
    cache(map_idx).s_grid = s_grid;
    cache(map_idx).x_grid = x_grid;
    cache(map_idx).y_grid = y_grid;
end
end
