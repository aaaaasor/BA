function cache = build_track_boundary_reference_cache(constraint_cfg)
%BUILD_TRACK_BOUNDARY_REFERENCE_CACHE Cache sample-constant phase searches.
% The nearest-point minimizer still runs at every velocity-field call. This
% cache only removes repeated spline-grid setup, preserving the original
% objective and fminbnd bracket exactly.

cache = struct([]);
if ~struct_field_default(constraint_cfg, 'track_boundary_enabled', false) || ...
        ~isfield(constraint_cfg, 'track_boundary_geometry') || ...
        ~isfield(constraint_cfg, 'track_boundary_point_maps')
    return;
end
constraint_method = lower(string(struct_field_default(constraint_cfg, ...
    'track_boundary_constraint_method', 'paired_cross_section')));
if constraint_method == "independent_signed_distance" || ...
        constraint_method == "global_implicit_fields"
    % Global boundary functions have no nominal/previous longitudinal phase.
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
matching_method = lower(string(struct_field_default(constraint_cfg, ...
    'track_boundary_phase_matching_method', 'fixed_window')));
branch_breaks = struct_field_default(constraint_cfg, ...
    'track_boundary_branch_breaks', [0.0, 1.0]);
branch_breaks = branch_breaks(:)';
if matching_method == "branch_soft"
    if numel(branch_breaks) < 2 || branch_breaks(1) ~= 0.0 || ...
            branch_breaks(end) ~= 1.0 || any(diff(branch_breaks) <= 0)
        error(['track_boundary_branch_breaks must increase strictly ', ...
            'from 0 to 1.']);
    end
elseif matching_method ~= "fixed_window"
    error(['track_boundary_phase_matching_method must be ', ...
        '''fixed_window'' or ''branch_soft''.']);
end
reference_weight = struct_field_default(constraint_cfg, ...
    'track_boundary_branch_reference_weight', 0.0);
previous_weight = struct_field_default(constraint_cfg, ...
    'track_boundary_branch_previous_weight', 0.0);
true_distance_enabled = struct_field_default(constraint_cfg, ...
    'track_boundary_true_distance_enabled', false);
true_distance_overlap_steps = struct_field_default(constraint_cfg, ...
    'track_boundary_true_distance_overlap_steps', 1.0);
if any(~isfinite([reference_weight, previous_weight])) || ...
        any([reference_weight, previous_weight] < 0)
    error('Track boundary branch weights must be finite and nonnegative.');
end
if ~isscalar(true_distance_overlap_steps) || ...
        ~isfinite(true_distance_overlap_steps) || ...
        true_distance_overlap_steps < 0
    error(['Track boundary true-distance overlap steps must be a ', ...
        'finite nonnegative scalar.']);
end
nominal_step = max(max(reference_s_max - reference_s_min), eps);
cache = repmat(struct('matching_method', char(matching_method), ...
    's_nominal', 0, 's_previous', 0, 'phase_scale', nominal_step, ...
    'reference_weight', reference_weight, 'previous_weight', previous_weight, ...
    'branch_index', 1, 's_reference', 0, 's_min', 0, 's_max', 1, ...
    'search_half_width', 0, 's_grid', [], 'x_grid', [], 'y_grid', [], ...
    'safety_s_grid', [], 'safety_center_points', [], ...
    'safety_left_points', [], 'safety_right_points', []), ...
    1, n_maps);
for map_idx = 1:n_maps
    s_reference = 0.5 * (reference_s_min(map_idx) + ...
        reference_s_max(map_idx));
    s_reference = max(0.0, min(1.0, s_reference));
    reference_half_width = max(0.5 * (reference_s_max(map_idx) - ...
        reference_s_min(map_idx)), eps);
    search_half_width = phase_search_half_steps * 2.0 * reference_half_width;
    if matching_method == "branch_soft"
        branch_idx = find(s_reference <= branch_breaks(2:end), 1, 'first');
        branch_idx = min(max(branch_idx, 1), numel(branch_breaks) - 1);
        s_min = max(branch_breaks(branch_idx), ...
            s_reference - search_half_width);
        s_max = min(branch_breaks(branch_idx + 1), ...
            s_reference + search_half_width);
    else
        s_min = max(0.0, s_reference - search_half_width);
        s_max = min(1.0, s_reference + search_half_width);
        branch_idx = 1;
    end
    in_window = geometry.s_control > s_min & geometry.s_control < s_max;
    % The window faces are valid minimizers and must be searched explicitly.
    % Omitting them can mistake a clamped minimum for an interior stationary
    % point and consequently use the wrong ds*/dp branch.
    s_grid = unique([s_min, geometry.s_control(in_window), s_max]);
    if numel(s_grid) < 3
        s_grid = unique([s_grid, 0.5 * (s_min + s_max)]);
    end
    x_grid = ppval(geometry.center_pp_x, s_grid);
    y_grid = ppval(geometry.center_pp_y, s_grid);
    cache(map_idx).s_nominal = s_reference;
    cache(map_idx).s_previous = s_reference;
    cache(map_idx).branch_index = branch_idx;
    cache(map_idx).s_reference = s_reference;
    cache(map_idx).s_min = s_min;
    cache(map_idx).s_max = s_max;
    cache(map_idx).search_half_width = search_half_width;
    cache(map_idx).s_grid = s_grid;
    cache(map_idx).x_grid = x_grid;
    cache(map_idx).y_grid = y_grid;
    if true_distance_enabled
        if matching_method == "branch_soft"
            safety_overlap = true_distance_overlap_steps * nominal_step;
            safety_s_min = max(0.0, ...
                branch_breaks(branch_idx) - safety_overlap);
            safety_s_max = min(1.0, ...
                branch_breaks(branch_idx + 1) + safety_overlap);
        else
            safety_s_min = s_min;
            safety_s_max = s_max;
        end
        safety_inner = geometry.s_control(geometry.s_control > safety_s_min & ...
            geometry.s_control < safety_s_max);
        safety_base = unique([safety_s_min, safety_inner, safety_s_max]);
        % Add interval midpoints so the geometric nearest-center search gets
        % a tight local bracket before its one-dimensional spline refinement.
        safety_mid = 0.5 * (safety_base(1:end-1) + safety_base(2:end));
        safety_s_grid = sort([safety_base, safety_mid]);
        cache(map_idx).safety_s_grid = safety_s_grid;
        cache(map_idx).safety_center_points = [ ...
            ppval(geometry.center_pp_x, safety_s_grid); ...
            ppval(geometry.center_pp_y, safety_s_grid)];
        cache(map_idx).safety_left_points = [ ...
            ppval(geometry.curves(1).pp_x, safety_s_grid); ...
            ppval(geometry.curves(1).pp_y, safety_s_grid)];
        cache(map_idx).safety_right_points = [ ...
            ppval(geometry.curves(2).pp_x, safety_s_grid); ...
            ppval(geometry.curves(2).pp_y, safety_s_grid)];
    end
end
end
