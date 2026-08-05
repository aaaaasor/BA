function h_min = evaluate_rollout_obstacle_hmin(rollout_path, constraint_cfg)
% Evaluate geometric obstacle h_min at saved rollout nodes without full diagnostics.
n_times = size(rollout_path, 1);
n_samples = size(rollout_path, 2);
state_dim = size(rollout_path, 3);
h_min = inf(n_times, n_samples);

if ~struct_field_default(constraint_cfg, 'obstacle_enabled', false) || ...
		~isfield(constraint_cfg, 'obstacle_point_maps') || ...
		empty_obstacle_geometry(constraint_cfg)
	return;
end

centers = constraint_cfg.obstacle_centers;
semi_axes = constraint_cfg.obstacle_semi_axes;
if isfield(constraint_cfg, 'obstacle_geometry')
	obstacle_geometry = constraint_cfg.obstacle_geometry;
else
	obstacle_geometry = struct('centers', centers, 'semi_axes', semi_axes);
end
if size(centers, 1) ~= 2
	centers = centers';
end
if size(semi_axes, 1) ~= 2
	semi_axes = semi_axes';
end
if size(centers, 1) ~= 2 || ~isequal(size(centers), size(semi_axes))
	error('evaluate_rollout_obstacle_hmin:InvalidObstacleGeometry', ...
		'Obstacle centers and semi-axes must both have size 2-by-n_obstacles.');
end
if any(semi_axes <= 0, 'all')
	error('evaluate_rollout_obstacle_hmin:InvalidSemiAxis', ...
		'Obstacle semi-axes must be positive.');
end

% Columns are ordered by time within each sample, matching reshape(...,
% n_times,n_samples) below.
state_columns = reshape(permute(rollout_path, [3, 1, 2]), state_dim, []);
point_maps = constraint_cfg.obstacle_point_maps;
for map_idx = 1:numel(point_maps)
	point_map = point_maps(map_idx);
	if size(point_map.M, 2) ~= state_dim
		error('evaluate_rollout_obstacle_hmin:StateDimensionMismatch', ...
			'Obstacle point map %d expects %d states, but rollout has %d.', ...
			map_idx, size(point_map.M, 2), state_dim);
	end
	positions = point_map.M * state_columns + point_map.o(:);
	for obstacle_idx = 1:size(centers, 2)
		angle = geometry_value(obstacle_geometry, 'angles', obstacle_idx, 0.0);
		exponent = geometry_value(obstacle_geometry, 'exponents', obstacle_idx, 2.0);
		rotation = [cos(angle), -sin(angle); sin(angle), cos(angle)];
		delta_local = rotation' * ...
			(positions - centers(:, obstacle_idx));
		scaled = delta_local ./ semi_axes(:, obstacle_idx);
		power_sum = sum(abs(scaled) .^ exponent, 1);
		if exponent > 2
			level_values = power_sum .^ (1 / exponent) - 1.0;
		else
			level_values = power_sum - 1.0;
		end
		h_now = reshape(level_values, ...
			n_times, n_samples);
		h_min = min(h_min, h_now);
	end
end
end

function value = geometry_value(geometry, field_name, obstacle_idx, default_value)
if ~isfield(geometry, field_name) || isempty(geometry.(field_name))
	value = default_value;
elseif isscalar(geometry.(field_name))
	value = geometry.(field_name);
else
	value = geometry.(field_name)(obstacle_idx);
end
end

function is_empty = empty_obstacle_geometry(constraint_cfg)
is_empty = ~isfield(constraint_cfg, 'obstacle_centers') || ...
	~isfield(constraint_cfg, 'obstacle_semi_axes') || ...
	empty_field(constraint_cfg, 'obstacle_centers') || ...
	empty_field(constraint_cfg, 'obstacle_semi_axes');
end

function tf = empty_field(value_struct, field_name)
tf = isempty(value_struct.(field_name));
end
