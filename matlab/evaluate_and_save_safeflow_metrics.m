function safe_flow_metrics = evaluate_and_save_safeflow_metrics(cfg, ...
	generated_points, dataset_points, constraint, rollout_times, ...
	variance_values, rollout_diagnostics, n_segments, seed_acceptance, ...
	evaluation_time, seed_bundle, cache_manifest, project_dir, generated_segment_data)
% Compute the SafeFlow metrics applicable to this geometric racing task and
% save them together with raw GP variance and the applied QP correction u.
% SD/ED are intentionally absent because this task has no externally fixed
% task-level start or goal. SR-A/AR/KC metrics require physical car actions.

eval_cfg = struct_field_default(cfg, 'safe_flow_evaluation', struct());
output_relative = struct_field_default(eval_cfg, 'output_path', ...
	fullfile('outputs', 'Racing_SafeFlow_Metrics_Variance_U.mat'));
output_path = fullfile(project_dir, output_relative);
output_dir = fileparts(output_path);
if ~exist(output_dir, 'dir')
	mkdir(output_dir);
end

validateattributes(generated_points, {'numeric'}, {'3d', 'finite'});
validateattributes(dataset_points, {'numeric'}, {'3d', 'finite'});
if size(generated_points, 3) < 2 || size(dataset_points, 3) < 2
	error('SafeFlow evaluation requires at least x-y features.');
end

%% Pointwise geometric safety at generated trajectory points
n_points = size(generated_points, 1);
n_trajectories = size(generated_points, 2);
tolerance = struct_field_default(eval_cfg, 'safety_tolerance', 1e-8);
safety_mask = false(n_trajectories, 1);
track_min_h = nan(n_trajectories, 1);
obstacle_min_h = nan(n_trajectories, 1);
point_safe_mask = false(n_points, n_trajectories);
track_point_h = nan(n_points, n_trajectories);
obstacle_point_h = nan(n_points, n_trajectories);
synthetic_domain = strcmpi(struct_field_default(cfg, 'scenario', ''), 'obstacle');
if synthetic_domain
	domain_bounds = struct_field_required(eval_cfg, 'synthetic_domain_bounds');
	validateattributes(domain_bounds, {'numeric'}, {'size', [2, 2], 'finite'});
	assert(all(domain_bounds(:, 1) < domain_bounds(:, 2)), ...
		'Synthetic domain lower bounds must be smaller than upper bounds.');
	safety_definition = ['pointwise generated-state constraint satisfaction: ', ...
		'points inside configured synthetic domain and outside physical obstacles; edges not checked'];
elseif isfield(constraint, 'track_boundary_geometry') && ...
		isfield(constraint.track_boundary_geometry, 'implicit_fields')
	track_fields = constraint.track_boundary_geometry.implicit_fields;
elseif isfield(cfg, 'track_boundary') && ...
		isfield(cfg.track_boundary, 'geometry') && ...
		isfield(cfg.track_boundary.geometry, 'implicit_fields')
	% Unguided baselines deliberately omit boundary data from the rollout
	% constraint.  Evaluation still uses the same physical geometry built in
	% cfg, without feeding it back into generation.
	track_fields = cfg.track_boundary.geometry.implicit_fields;
else
	error('Pointwise Safety requires the original implicit track fields.');
end
if ~synthetic_domain
	safety_definition = ['pointwise generated-state constraint satisfaction: ', ...
		'points inside track and outside physical obstacles; edges not checked'];
end
if isfield(constraint, 'obstacle_physical_geometry')
	obstacles = constraint.obstacle_physical_geometry;
elseif isfield(constraint, 'obstacle_geometry')
	obstacles = constraint.obstacle_geometry;
elseif isfield(cfg, 'obstacle') && isfield(cfg.obstacle, 'centers') && ...
		isfield(cfg.obstacle, 'semi_axes')
	% The physical obstacle description remains available in cfg when all
	% per-level obstacle-guidance switches are disabled.
	obstacles = cfg.obstacle;
else
	error('Pointwise Safety requires physical obstacle geometry.');
end
for trajectory_idx = 1:n_trajectories
	xy = squeeze(generated_points(:, trajectory_idx, 1:2));
	for point_idx = 1:n_points
		p = xy(point_idx, :)';
		if synthetic_domain
			% Same pointwise sign test, with the four faces of the synthetic
			% domain instead of racing-track splines. Never fed into rollout.
			track_point_h(point_idx, trajectory_idx) = ...
				min([p - domain_bounds(:, 1); domain_bounds(:, 2) - p]);
		else
			left_h = evaluate_track_implicit_field(track_fields, 1, p);
			right_h = evaluate_track_implicit_field(track_fields, 2, p);
			track_point_h(point_idx, trajectory_idx) = min(left_h, right_h);
		end
		obstacle_h = inf;
		for obstacle_idx = 1:size(obstacles.centers, 2)
			obstacle_h = min(obstacle_h, obstacle_level_and_gradient( ...
				p, obstacles, obstacle_idx));
		end
		obstacle_point_h(point_idx, trajectory_idx) = obstacle_h;
		point_safe_mask(point_idx, trajectory_idx) = ...
			track_point_h(point_idx, trajectory_idx) >= -tolerance && ...
			obstacle_h >= -tolerance;
	end
	track_min_h(trajectory_idx) = min(track_point_h(:, trajectory_idx));
	obstacle_min_h(trajectory_idx) = min( ...
		obstacle_point_h(:, trajectory_idx));
	safety_mask(trajectory_idx) = all(point_safe_mask(:, trajectory_idx));
end

%% SafeFlow CS and AS: retain unmatched segment endpoints
[cs_per_trajectory, as_per_trajectory, smoothness_details] = ...
    segment_trajectory_smoothness(generated_segment_data, n_segments, ...
    cfg.segment_points_per_segment);
assert(numel(cs_per_trajectory) == n_trajectories, ...
    'Smoothness segment data must match the evaluated parent trajectories.');
%% Endpoint-distribution KL via a common Gaussian KDE grid
generated_final_xy = squeeze(generated_points(end, :, 1:2));
dataset_final_xy = squeeze(dataset_points(end, :, 1:2));
generated_final_xy = reshape(generated_final_xy, [], 2);
dataset_final_xy = reshape(dataset_final_xy, [], 2);
[kl_divergence, kl_details] = endpoint_kde_kl( ...
	dataset_final_xy, generated_final_xy, eval_cfg, project_dir);

%% Accepted u trace only
if ~isfield(rollout_diagnostics, 'hocbf') || ...
		~isfield(rollout_diagnostics.hocbf, 'trace_u') || ...
		empty_trace(rollout_diagnostics.hocbf)
	error(['SafeFlow evaluation is enabled, but the rollout cache has no ', ...
		'nonempty u trace. Delete/rebuild the third-level rollout cache.']);
end
trace = rollout_diagnostics.hocbf;
accepted_trace_mask = accepted_u_trace_mask(trace, seed_acceptance, n_segments);
u_trace = struct( ...
	'u', trace.trace_u(accepted_trace_mask, :), ...
	't', trace.trace_t(accepted_trace_mask, :), ...
	'segment_sample_index', trace.trace_sample_idx(accepted_trace_mask, :), ...
	'rk4_step_index', trace.trace_step_idx(accepted_trace_mask, :), ...
	'rk4_stage_index', trace.trace_stage_idx(accepted_trace_mask, :), ...
	'coordinate_space', 'normalized trajectory-space flow correction', ...
	'accepted_attempts_only', true);
if isfield(trace, 'trace_seed_attempt')
	u_trace.seed_attempt = trace.trace_seed_attempt(accepted_trace_mask, :);
end

%% Compact metric result and raw export
if isempty(variance_values) || ~isnumeric(variance_values)
	error(['Variance comparison requires a nonempty numeric array with ', ...
		'layout time x segment sample x GP output.']);
end
mean_predictive_variance = mean(variance_values(:), 'omitnan');
terminal_variance_values = variance_values(end, :, :);
terminal_predictive_variance = mean(terminal_variance_values(:), 'omitnan');

safe_flow_metrics = struct();
safe_flow_metrics.safety_rate = mean(safety_mask);
safe_flow_metrics.safety_percent = 100 * safe_flow_metrics.safety_rate;
safe_flow_metrics.kl_divergence = kl_divergence;
safe_flow_metrics.cs = mean(cs_per_trajectory, 'omitnan');
safe_flow_metrics.as = mean(as_per_trajectory, 'omitnan');
safe_flow_metrics.mean_predictive_variance = mean_predictive_variance;
safe_flow_metrics.terminal_predictive_variance = ...
	terminal_predictive_variance;
safe_flow_metrics.time_seconds = evaluation_time.seconds_per_final_trajectory;
safe_flow_metrics.time = evaluation_time;

safety_details = struct( ...
	'safe_mask', safety_mask, ...
	'point_safe_mask', point_safe_mask, ...
	'track_point_h', track_point_h, ...
	'obstacle_point_h', obstacle_point_h, ...
	'track_min_h', track_min_h, ...
	'obstacle_min_h', obstacle_min_h, ...
	'cs_per_trajectory', cs_per_trajectory, ...
	'as_per_trajectory', as_per_trajectory);
variance_times = rollout_times;
evaluation_metadata = struct( ...
	'metric_set', ['Safety, KL, CS, AS, mean predictive variance, ', ...
	'terminal predictive variance, Time'], ...
	'safety_definition', safety_definition, ...
	'kl_definition', 'D_KL(dataset endpoint KDE || generated endpoint KDE)', ...
	'variance_metric_definition', ['raw third-level GP predictive variance; ', ...
	'MeanVar averages time x segment sample x output, TerminalVar averages ', ...
	'the final rollout slice x segment sample x output'], ...
	'smoothness_definition', smoothness_details.definition, ...
	'smoothness_junction_tolerance', smoothness_details.junction_tolerance, ...
	'smoothness_points_per_trajectory', smoothness_details.points_per_trajectory, ...
	'variance_layout', 'time x segment sample x GP output', ...
	'u_layout', 'RK4 substage record x normalized trajectory-space dimension', ...
	'n_final_trajectories', n_trajectories, ...
	'n_segments_per_trajectory', n_segments, ...
	'created_at', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
cache_manifest.evaluation_reference_paths = {kl_details.reference_path};
cache_manifest = describe_cache_files(cache_manifest);
saved_cfg = cfg;

fprintf(['Evaluation metrics: Safety %.2f%%, KL %.6g, CS %.6g, ', ...
	'AS %.6g, MeanVar %.6g, TerminalVar %.6g, ', ...
	'Time %.6g s/trajectory.\n'], ...
	safe_flow_metrics.safety_percent, safe_flow_metrics.kl_divergence, ...
	safe_flow_metrics.cs, safe_flow_metrics.as, ...
	safe_flow_metrics.mean_predictive_variance, ...
	safe_flow_metrics.terminal_predictive_variance, ...
	safe_flow_metrics.time_seconds);
save(output_path, 'safe_flow_metrics', 'safety_details', 'kl_details', ...
	'smoothness_details', ...
	'variance_values', 'variance_times', 'u_trace', ...
	'generated_final_xy', 'dataset_final_xy', 'seed_bundle', ...
	'cache_manifest', 'saved_cfg', 'evaluation_metadata', '-v7.3');
disp(['Saved SafeFlow metrics, variance, and u: ', output_path]);
end

function manifest = describe_cache_files(manifest)
groups = {'hyperparameter_paths', 'model_paths', 'rollout_paths', ...
	'seed_paths', 'evaluation_reference_paths'};
all_paths = cell(0, 1);
all_groups = cell(0, 1);
for group_idx = 1:numel(groups)
	paths = manifest.(groups{group_idx});
	all_paths = [all_paths; paths(:)]; %#ok<AGROW>
	all_groups = [all_groups; repmat(groups(group_idx), numel(paths), 1)]; %#ok<AGROW>
end
n = numel(all_paths);
exists = false(n, 1);
bytes = nan(n, 1);
last_modified = strings(n, 1);
for idx = 1:n
	info = dir(all_paths{idx});
	if ~isempty(info)
		exists(idx) = true;
		bytes(idx) = info(1).bytes;
		last_modified(idx) = string(info(1).date);
	end
end
manifest.file_table = table(string(all_groups), string(all_paths), ...
	exists, bytes, last_modified, 'VariableNames', ...
	{'cache_group', 'path', 'exists', 'bytes', 'last_modified'});
end

function mask = accepted_u_trace_mask(trace, acceptance, n_segments)
n = size(trace.trace_u, 1);
mask = true(n, 1);
if ~isfield(trace, 'trace_seed_attempt') || ...
		~struct_field_default(acceptance, 'enabled', false)
	return;
end
sample_idx = trace.trace_sample_idx(:);
attempt = trace.trace_seed_attempt(:);
valid_sample = isfinite(sample_idx) & sample_idx >= 1 & ...
	sample_idx == floor(sample_idx);
desired_attempt = nan(n, 1);
if isfield(acceptance, 'segment_attempt_counts')
	counts = acceptance.segment_attempt_counts;
	parent_idx = ceil(sample_idx(valid_sample) / n_segments);
	segment_idx = mod(sample_idx(valid_sample) - 1, n_segments) + 1;
	linear_idx = sub2ind(size(counts), parent_idx, segment_idx);
	desired_attempt(valid_sample) = counts(linear_idx);
else
	counts = acceptance.attempt_counts(:);
	parent_idx = ceil(sample_idx(valid_sample) / n_segments);
	desired_attempt(valid_sample) = counts(parent_idx);
end
mask = valid_sample & isfinite(attempt) & attempt == desired_attempt;
if ~any(mask)
	error('No accepted-attempt u records were found in the rollout trace.');
end
end

function [kl_value, details] = endpoint_kde_kl( ...
	data_xy, generated_xy, eval_cfg, project_dir)
grid_size = struct_field_default(eval_cfg, 'kl_grid_size', 128);
validateattributes(grid_size, {'numeric'}, ...
	{'scalar', 'integer', '>=', 32, '<=', 512});
reference_relative = struct_field_default(eval_cfg, 'kl_reference_path', ...
	fullfile('outputs', 'Racing_KL_Reference.mat'));
reference_path = fullfile(project_dir, reference_relative);
rebuild = struct_field_default(eval_cfg, 'kl_reference_rebuild', false);
reference_exists = isfile(reference_path);
reference_reused = false;
reference_migrated = false;
if reference_exists && ~rebuild
	loaded = load(reference_path, 'kl_reference');
	if ~isfield(loaded, 'kl_reference')
		error('KL reference file does not contain kl_reference: %s', ...
			reference_path);
	end
	if isfield(loaded.kl_reference, 'version') && ...
			loaded.kl_reference.version == 2
		reference = loaded.kl_reference;
		validate_kl_reference(reference, data_xy, grid_size, reference_path);
		reference_reused = true;
	else
		reference = create_kl_reference(data_xy, grid_size);
		save_kl_reference(reference_path, reference);
		reference_migrated = true;
	end
else
	reference = create_kl_reference(data_xy, grid_size);
	save_kl_reference(reference_path, reference);
end
bandwidth = reference.bandwidth;
x_grid = reference.x_grid;
y_grid = reference.y_grid;
[grid_x, grid_y] = meshgrid(x_grid, y_grid);
grid_xy = [grid_x(:), grid_y(:)];
p = reference.dataset_density(:);
q = gaussian_kde_on_grid(grid_xy, generated_xy, bandwidth);
q = max(q, realmin('double'));
grid_step = [x_grid(2) - x_grid(1), y_grid(2) - y_grid(1)];
cell_area = prod(grid_step);
% Evaluate log(p/q) as log(p)-log(q).  Forming p/q first can overflow when
% an unguided method places all generated endpoints far outside the fixed
% dataset reference grid and q reaches realmin on cells carrying P mass.
positive_p = p > 0;
kl_value = max(0, sum(p(positive_p) .* ...
	(log(p(positive_p)) - log(q(positive_p)))) * cell_area);
outside = generated_xy(:, 1) < x_grid(1) | ...
	generated_xy(:, 1) > x_grid(end) | ...
	generated_xy(:, 2) < y_grid(1) | ...
	generated_xy(:, 2) > y_grid(end);
details = struct('bandwidth', bandwidth, 'grid_size', grid_size, ...
	'x_grid', x_grid, 'y_grid', y_grid, ...
	'dataset_density', reshape(p, grid_size, grid_size), ...
	'generated_density', reshape(q, grid_size, grid_size), ...
	'generated_endpoint_out_of_grid_rate', mean(outside), ...
	'generated_endpoint_out_of_grid_count', nnz(outside), ...
	'reference_path', reference_path, ...
	'reference_reused', reference_reused, ...
	'reference_migrated', reference_migrated, ...
	'grid_step', grid_step, ...
	'bandwidth_to_grid_step_ratio', bandwidth ./ grid_step, ...
	'reference_basis', reference.basis);
end

function reference = create_kl_reference(data_xy, grid_size)
data_scale = std(data_xy, 0, 1);
data_span = max(data_xy, [], 1) - min(data_xy, [], 1);
scale_floor = max(data_span * 1e-3, 1e-8);
data_scale = max(data_scale, scale_floor);
bandwidth = data_scale * size(data_xy, 1)^(-1/6);
lower = min(data_xy, [], 1) - 6 * bandwidth;
upper = max(data_xy, [], 1) + 6 * bandwidth;
basis = 'dataset endpoint bounds plus six dataset-only bandwidths';
x_grid = linspace(lower(1), upper(1), grid_size);
y_grid = linspace(lower(2), upper(2), grid_size);
[grid_x, grid_y] = meshgrid(x_grid, y_grid);
grid_xy = [grid_x(:), grid_y(:)];
p = gaussian_kde_on_grid(grid_xy, data_xy, bandwidth);
grid_step = [x_grid(2) - x_grid(1), y_grid(2) - y_grid(1)];
p = p / (sum(p) * prod(grid_step));
reference = struct( ...
	'version', 2, ...
	'basis', basis, ...
	'bandwidth', bandwidth, ...
	'grid_size', grid_size, ...
	'x_grid', x_grid, ...
	'y_grid', y_grid, ...
	'dataset_density', reshape(p, grid_size, grid_size), ...
	'dataset_final_xy', data_xy, ...
	'created_at', char(datetime('now', ...
		'Format', 'yyyy-MM-dd HH:mm:ss')));
end

function save_kl_reference(reference_path, reference)
reference_dir = fileparts(reference_path);
if ~exist(reference_dir, 'dir')
	mkdir(reference_dir);
end
kl_reference = reference;
save(reference_path, 'kl_reference', '-v7');
end

function validate_kl_reference(reference, data_xy, grid_size, reference_path)
required = {'version', 'bandwidth', 'grid_size', 'x_grid', 'y_grid', ...
	'dataset_density', 'dataset_final_xy', 'basis'};
if ~all(isfield(reference, required))
	error(['KL reference has an obsolete layout. Set ', ...
		'cfg.safe_flow_evaluation.kl_reference_rebuild=true once: %s'], ...
		reference_path);
end
same_dataset = isequal(size(reference.dataset_final_xy), size(data_xy)) && ...
	max(abs(reference.dataset_final_xy - data_xy), [], 'all') <= 1e-12;
if reference.version ~= 2 || reference.grid_size ~= grid_size || ~same_dataset
	error(['KL reference does not match the configured grid or dataset. ', ...
		'All methods must share one reference. If this change is deliberate, ', ...
		'set cfg.safe_flow_evaluation.kl_reference_rebuild=true once and ', ...
		'then return it to false. Reference: %s'], reference_path);
end
end

function density = gaussian_kde_on_grid(grid_xy, samples, bandwidth)
density = zeros(size(grid_xy, 1), 1);
for sample_idx = 1:size(samples, 1)
	delta = (grid_xy - samples(sample_idx, :)) ./ bandwidth;
	density = density + exp(-0.5 * sum(delta.^2, 2));
end
density = density / max(size(samples, 1), 1);
density = density / (2 * pi * prod(bandwidth));
end

function tf = empty_trace(trace)
tf = ~isfield(trace, 'trace_t') || isempty(trace.trace_t) || ...
	~isfield(trace, 'trace_u') || isempty(trace.trace_u);
end
