% Generate exactly one accepted full trajectory per parent trajectory.
% seed_scope='trajectory' retains one seed for all adjacent segments;
% seed_scope='segment' assigns one independently reproducible seed to every
% segment. retry_scope='segment' freezes accepted segments and retries only
% failed segments; all segments must pass before a full trajectory completes.
function [times, accepted_path, diagnostics, accepted_x_init, ...
	accepted_seeds, acceptance, accepted_uncertainty, exhaustion] = ...
	generate_valid_second_level_rollouts(model_collection, base_constraint, ...
	n_parents, n_segments, state_dim, n_points_per_segment, data_transform, ...
	t0, t1, n_steps, refine_cfg, ...
	parallel_cfg, filter_cfg)

if ~struct_field_default(base_constraint, 'ptcbf_enabled', false)
	error(['Variance rejection sampling requires ', ...
		'ptcbf_enabled=true.']);
end
fixed_hbar0 = struct_field_default(base_constraint, ...
	'terminal_variance_ptzf_hbar0_fixed', []);
if isempty(fixed_hbar0)
	error(['Variance rejection sampling requires one fixed ', ...
		'terminal_variance_ptzf_hbar0_fixed across candidate batches.']);
end

base_seed = struct_field_required(filter_cfg, 'base_seed');
max_attempts = struct_field_default(filter_cfg, 'max_attempts_per_trajectory', 100);
return_on_exhaustion = struct_field_default(filter_cfg, ...
	'return_on_exhaustion', false);
seed_scope = lower(string(struct_field_default(filter_cfg, ...
	'seed_scope', 'trajectory')));
if ~any(seed_scope == ["trajectory", "segment"])
	error('seed_scope must be trajectory or segment.');
end
retry_scope = lower(string(struct_field_default(filter_cfg, ...
	'retry_scope', 'trajectory')));
if ~any(retry_scope == ["trajectory", "segment"])
	error('retry_scope must be trajectory or segment.');
end
full_parent_count = n_parents;
full_segment_count = n_segments;
fixed_seed_matrix = struct_field_default(filter_cfg, ...
	'fixed_seed_matrix', []);
segment_retry = retry_scope == "segment" && n_segments > 1;
if segment_retry
	if seed_scope ~= "segment"
		error('Segment-local retry requires seed_scope=segment.');
	end
	% A retry unit is now one original segment row, not an entire parent.
	% Keep the original grouping for seed allocation, logging and exports.
	n_parents = full_parent_count * full_segment_count;
	n_segments = 1;
end
if ~isempty(fixed_seed_matrix)
	[times, accepted_path, diagnostics, accepted_x_init, ...
		accepted_seeds, acceptance, accepted_uncertainty, exhaustion] = ...
		replay_fixed_seed_rollouts(model_collection, base_constraint, ...
		fixed_seed_matrix, full_parent_count, full_segment_count, ...
		n_parents, n_segments, state_dim, t0, t1, n_steps, ...
		refine_cfg, parallel_cfg, ...
		filter_cfg, seed_scope, retry_scope, segment_retry, fixed_hbar0);
	return;
end
tolerance = struct_field_default(filter_cfg, 'tolerance', 1e-8);
track_point_filter_enabled = struct_field_default(filter_cfg, ...
	'final_track_point_filter_enabled', false);
track_point_tolerance = struct_field_default(filter_cfg, ...
	'final_track_point_tolerance', 1e-8);
track_point_include_margin = struct_field_default(filter_cfg, ...
	'final_track_point_include_margin', false);
track_acceptance_margin = 0.0;
track_margin_reference = 'raw_implicit_boundary';
if track_point_include_margin
	track_acceptance_margin = struct_field_default(base_constraint, ...
		'track_boundary_margin', 0.0);
	track_margin_reference = 'controller_margin_subtracted';
end
obstacle_point_filter_enabled = struct_field_default(filter_cfg, ...
	'final_obstacle_point_filter_enabled', false);
obstacle_point_tolerance = struct_field_default(filter_cfg, ...
	'final_obstacle_point_tolerance', 1e-8);
track_line_filter_enabled = struct_field_default(filter_cfg, ...
	'final_track_line_filter_enabled', false);
obstacle_line_filter_enabled = struct_field_default(filter_cfg, ...
	'final_obstacle_line_filter_enabled', false);
line_filter_enabled = track_line_filter_enabled || obstacle_line_filter_enabled;
geometry_internal_points_only = struct_field_default(filter_cfg, ...
	'final_geometry_internal_points_only', true);
geometry_state_mode = lower(string(struct_field_default(filter_cfg, ...
	'final_geometry_state_mode', 'absolute')));
if ~any(geometry_state_mode == ["absolute", "increment"])
	error('final_geometry_state_mode must be absolute or increment.');
end
if geometry_internal_points_only
	geometry_point_indices = 2:(n_points_per_segment - 1);
else
	geometry_point_indices = 1:n_points_per_segment;
end
if max_attempts < 1 || max_attempts ~= floor(max_attempts)
	error('max_attempts_per_trajectory must be a positive integer.');
end

if seed_scope == "segment"
	accepted_seeds = nan(n_parents, n_segments);
else
	accepted_seeds = nan(n_parents, 1);
end
attempt_counts = zeros(n_parents, 1);
search_max_excess = nan(n_parents, 1);
final_max_excess = nan(n_parents, 1);
final_min_hocbf_psi0 = nan(n_parents, 1);
final_min_hocbf_psi1 = nan(n_parents, 1);
final_min_track_point_h = nan(n_parents, 1);
final_min_obstacle_point_h = nan(n_parents, 1);
final_min_track_line_h = nan(n_parents, 1);
final_min_obstacle_line_h = nan(n_parents, 1);
accepted_x_init = zeros(n_parents * n_segments, state_dim);
accepted_path = [];
accepted_uncertainty = [];
batch_diagnostics = cell(0, 1);
times = [];
beta_cap = [];
terminal_hard_time_mask = [];
hocbf_hard_time_mask = [];
batch_size = struct_field_default(filter_cfg, ...
	'parallel_trajectory_batch_size', 6);
if batch_size < 1 || batch_size ~= floor(batch_size)
	error('parallel_trajectory_batch_size must be a positive integer.');
end
if segment_retry
	batch_size = batch_size * full_segment_count;
end
batch_number = 0;
pending_queue = (1:n_parents)';
exhausted_units = zeros(0, 1);
while ~isempty(pending_queue)
	batch_number = batch_number + 1;
	n_batch = min(batch_size, numel(pending_queue));
	batch_parents = pending_queue(1:n_batch);
	pending_queue = pending_queue((n_batch + 1):end);
	pending_queue = pending_queue(:);
	can_attempt = attempt_counts(batch_parents) < max_attempts;
	if any(~can_attempt)
		exhausted_now = batch_parents(~can_attempt);
		if ~return_on_exhaustion
			if segment_retry
				failed_pairs = [floor((exhausted_now-1)/full_segment_count)+1, ...
					mod(exhausted_now-1,full_segment_count)+1];
				error(['Could not find a valid seed within %d attempts per segment. ', ...
					'Failed [trajectory segment] pairs: %s.'], ...
					max_attempts, mat2str(failed_pairs));
			end
			error(['Could not find a hard-variance-valid seed within %d ', ...
				'attempts for full trajectory indices %s.'], ...
				max_attempts, mat2str(exhausted_now'));
		end
		exhausted_units = unique([exhausted_units; exhausted_now], 'stable');
		batch_parents = batch_parents(can_attempt);
		n_batch = numel(batch_parents);
	end
	if n_batch == 0
		continue;
	end
	attempt_counts(batch_parents) = attempt_counts(batch_parents) + 1;
	[candidate_x_init, candidate_seeds] = seeded_initial_states( ...
		base_seed, batch_parents, attempt_counts(batch_parents), ...
		max_attempts, n_segments, state_dim, seed_scope, ...
		1 + double(segment_retry) * (full_segment_count - 1));
	candidate_constraint = subset_sample_targets(base_constraint, ...
		batch_parents, n_segments);
	if batch_number > 1
		% A partial retry must not overwrite a previous complete animation.
		pattern = struct_field_default(candidate_constraint, ...
			'live_trajectory_video_output_pattern', '');
		if ~isempty(pattern)
			[folder, stem, ext] = fileparts(pattern);
			candidate_constraint.live_trajectory_video_output_pattern = ...
				fullfile(folder, [stem, sprintf('_RetryBatch_%03d', batch_number), ext]);
		end
	end
	% A failed QP means this seed did not produce a usable trajectory.  Let
	% rk4_rollout return that sample as failed so the rejection queue can try
	% its next seed without aborting every other trajectory in the batch.
	candidate_constraint.seed_filter_reject_failed_rollouts = true;
	if segment_retry
		complete = all(reshape(isfinite(accepted_seeds), full_segment_count, [])', 2);
		fprintf(['Parallel variance seed filter: %d pending segments, ', ...
			'%d/%d segments accepted, %d/%d full trajectories complete...\n'], ...
			n_batch, nnz(isfinite(accepted_seeds)), n_parents, ...
			nnz(complete), full_parent_count);
	else
	fprintf(['Parallel variance seed filter: %d full trajectories, ', ...
		'%d segment samples, %d/%d already accepted...\n'], ...
		n_batch, n_batch * n_segments, ...
		nnz(all(isfinite(accepted_seeds), 2)), n_parents);
	end
	[candidate_times, candidate_path, candidate_diagnostics] = ...
		rk4_rollout(model_collection, candidate_x_init, t0, t1, n_steps, ...
		candidate_constraint, refine_cfg, parallel_cfg);
	original_rows = reshape(((batch_parents(:)-1)*n_segments + ...
		(1:n_segments))', [], 1);
	batch_diagnostics{end + 1, 1} = remap_batch_diagnostics( ...
		candidate_diagnostics, original_rows, ...
		repelem(attempt_counts(batch_parents), n_segments)); %#ok<AGROW>
	failed_segment_mask = struct_field_default(candidate_diagnostics, ...
		'failed_sample_mask', false(size(candidate_path, 2), 1));
	failed_segment_mask = failed_segment_mask(:);
	candidate_uncertainty = nan(numel(candidate_times), ...
		size(candidate_path, 2), ...
		numel(model_collection.model.output_models));
	successful_segment_mask = ~failed_segment_mask;
	if any(successful_segment_mask)
		candidate_uncertainty(:, successful_segment_mask, :) = ...
			evaluate_rollout_uncertainty(model_collection, candidate_times, ...
			candidate_path(:, successful_segment_mask, :));
	end
	candidate_beta = aggregate_variance_values(candidate_uncertainty);
	candidate_cap = terminal_variance_cap(candidate_times, base_constraint);
	candidate_terminal_hard_mask = terminal_variance_hard_mask( ...
		candidate_times, base_constraint);
	if isempty(times)
		times = candidate_times;
		accepted_path = zeros(numel(times), n_parents * n_segments, state_dim);
		accepted_uncertainty = zeros(numel(times), n_parents * n_segments, ...
			size(candidate_uncertainty, 3));
		beta_cap = candidate_cap;
		terminal_hard_time_mask = candidate_terminal_hard_mask;
	elseif ~isequal(times, candidate_times)
		error('Parallel candidate batches produced inconsistent time grids.');
	end

	for batch_idx = 1:n_batch
		parent_idx = batch_parents(batch_idx);
		if segment_retry
			candidate_label = sprintf('trajectory %d segment %d', ...
				floor((parent_idx-1)/full_segment_count)+1, ...
				mod(parent_idx-1,full_segment_count)+1);
		else
			candidate_label = sprintf('trajectory %d', parent_idx);
		end
		source_rows = (batch_idx - 1) * n_segments + (1:n_segments);
		if any(failed_segment_mask(source_rows))
			pending_queue = [pending_queue; parent_idx]; %#ok<AGROW>
			failed_local_rows = source_rows(failed_segment_mask(source_rows));
			failure_messages = struct_field_default(candidate_diagnostics, ...
				'failed_sample_messages', repmat({''}, size(candidate_path, 2), 1));
			failure_message = failure_messages{failed_local_rows(1)};
			fprintf(['  %s rejected (attempt %d, seeds %s): ', ...
				'rollout/QP failure: %s\n'], candidate_label, ...
				attempt_counts(parent_idx), ...
				seed_text(candidate_seeds(batch_idx, :)), ...
				failure_message);
			continue;
		end
		parent_beta = candidate_beta(:, source_rows);
		terminal_excess = max( ...
			parent_beta(candidate_terminal_hard_mask, :) - ...
			candidate_cap(candidate_terminal_hard_mask), [], 'all');
		[min_hocbf_psi0, min_hocbf_psi1, ...
			candidate_hocbf_hard_mask] = integral_hocbf_margins( ...
			candidate_times, parent_beta, base_constraint);
		min_track_point_h = final_track_point_margin( ...
			reshape(candidate_path(end, source_rows, :), numel(source_rows), state_dim), data_transform, ...
			n_segments, n_points_per_segment, base_constraint, ...
			track_point_filter_enabled, geometry_point_indices, ...
			geometry_state_mode, track_acceptance_margin);
		min_obstacle_point_h = final_obstacle_point_margin( ...
			reshape(candidate_path(end, source_rows, :), numel(source_rows), state_dim), data_transform, ...
			n_segments, n_points_per_segment, base_constraint, ...
			obstacle_point_filter_enabled, geometry_point_indices, ...
			geometry_state_mode);
		if isempty(hocbf_hard_time_mask)
			hocbf_hard_time_mask = candidate_hocbf_hard_mask;
		end
		line_report = struct('valid',true,'track_min_h',inf,'obstacle_min_h',inf);
		if line_filter_enabled
			line_report = evaluate_final_polyline_safety( ...
				reshape(candidate_path(end, source_rows, :),numel(source_rows),state_dim), ...
				data_transform,n_points_per_segment,base_constraint,filter_cfg);
		end
		valid = terminal_excess <= tolerance && ...
			min_hocbf_psi0 >= -tolerance && min_hocbf_psi1 >= -tolerance && ...
			min_track_point_h >= -track_point_tolerance && ...
			min_obstacle_point_h >= -obstacle_point_tolerance && line_report.valid;
		valid = valid && all(isfinite(candidate_path(:,source_rows,:)), 'all') && ...
			all(isfinite(parent_beta), 'all');
		if ~valid
			pending_queue = [pending_queue; parent_idx]; %#ok<AGROW>
			fprintf(['  %s rejected (attempt %d, seeds %s): ', ...
				'terminal excess=%+.3g, min psi0=%+.3g, min psi1=%+.3g, ', ...
				'min track-point h=%+.3g, min obstacle-point h=%+.3g.\n'], ...
				candidate_label, attempt_counts(parent_idx), ...
				seed_text(candidate_seeds(batch_idx, :)), terminal_excess, ...
				min_hocbf_psi0, min_hocbf_psi1, min_track_point_h, ...
				min_obstacle_point_h);
			if line_filter_enabled && ~line_report.valid
				fprintf(['    line check failed: track min h=%+.6g ', ...
					'(local segment %g, edge P%g-P%g, lambda %.6g); ', ...
					'obstacle min h=%+.6g (obstacle %g, local segment %g, ', ...
					'edge P%g-P%g, lambda %.6g).\n'], ...
					line_report.track_min_h,line_report.track_segment, ...
					line_report.track_edge,line_report.track_edge+1,line_report.track_lambda, ...
					line_report.obstacle_min_h,line_report.obstacle_index, ...
					line_report.obstacle_segment,line_report.obstacle_edge, ...
					line_report.obstacle_edge+1,line_report.obstacle_lambda);
			end
			continue;
		end
		destination_rows = (parent_idx - 1) * n_segments + (1:n_segments);
		accepted_x_init(destination_rows, :) = candidate_x_init(source_rows, :);
		accepted_path(:, destination_rows, :) = ...
			candidate_path(:, source_rows, :); %#ok<AGROW>
		accepted_uncertainty(:, destination_rows, :) = ...
			candidate_uncertainty(:, source_rows, :); %#ok<AGROW>
		accepted_seeds(parent_idx, :) = candidate_seeds(batch_idx, :);
		search_max_excess(parent_idx) = terminal_excess;
		final_max_excess(parent_idx) = terminal_excess;
		final_min_hocbf_psi0(parent_idx) = min_hocbf_psi0;
		final_min_hocbf_psi1(parent_idx) = min_hocbf_psi1;
		final_min_track_point_h(parent_idx) = min_track_point_h;
		final_min_obstacle_point_h(parent_idx) = min_obstacle_point_h;
		final_min_track_line_h(parent_idx) = line_report.track_min_h;
		final_min_obstacle_line_h(parent_idx) = line_report.obstacle_min_h;
		if segment_retry
			fprintf('  %s accepted (attempt %d, seed %s); kept for final assembly.\n', ...
				candidate_label, attempt_counts(parent_idx), ...
				seed_text(candidate_seeds(batch_idx,:)));
		else
			fprintf('  %s accepted (seeds %s): %d/%d complete.\n', ...
				candidate_label, seed_text(candidate_seeds(batch_idx, :)), ...
				nnz(all(isfinite(accepted_seeds), 2)), n_parents);
		end
	end
	% The displayed curve keeps P2:P5 from every segment after the first.
	% Its joining edge is therefore previous P5 -> current P2, which must
	% also pass even when the two nominally shared endpoints differ slightly.
	if line_filter_enabled && isempty(pending_queue) && ...
			isempty(exhausted_units) && full_segment_count > 1
		[join_bad_rows,join_track_h,join_obstacle_h] = final_polyline_join_checks( ...
			reshape(accepted_path(end,:,:),full_parent_count*full_segment_count,state_dim), ...
			data_transform,n_points_per_segment,full_segment_count,base_constraint,filter_cfg);
		if segment_retry
			final_min_track_line_h=min(final_min_track_line_h,join_track_h);
			final_min_obstacle_line_h=min(final_min_obstacle_line_h,join_obstacle_h);
			bad_units=join_bad_rows;
		else
			final_min_track_line_h=min(final_min_track_line_h, ...
				reshape(min(reshape(join_track_h,full_segment_count,[]),[],1),[],1));
			final_min_obstacle_line_h=min(final_min_obstacle_line_h, ...
				reshape(min(reshape(join_obstacle_h,full_segment_count,[]),[],1),[],1));
			bad_units=unique(ceil(join_bad_rows/full_segment_count),'stable');
		end
		for row=reshape(join_bad_rows,1,[])
			parent=ceil(row/full_segment_count);segment=mod(row-1,full_segment_count)+1;
			fprintf(['  trajectory %d joining edge segment %d P5 -> segment %d P2 ', ...
				'rejected: track h=%+.6g, obstacle h=%+.6g; retrying its owner.\n'], ...
				parent,segment-1,segment,join_track_h(row),join_obstacle_h(row));
		end
		accepted_seeds(bad_units,:)=nan;
		pending_queue=bad_units(:);
	end
end
diagnostics = merge_top_level_diagnostics(batch_diagnostics);
diagnostics.seed_filter_includes_rejected_attempts = true;
diagnostics.failed_sample_mask = false(full_parent_count*full_segment_count,1);
diagnostics.failed_sample_messages = repmat({''},full_parent_count*full_segment_count,1);

if segment_retry
	exhausted_parent_indices = floor((exhausted_units - 1) ./ ...
		full_segment_count) + 1;
	exhausted_segment_indices = mod(exhausted_units - 1, ...
		full_segment_count) + 1;
else
	exhausted_parent_indices = exhausted_units;
	exhausted_segment_indices = ones(size(exhausted_units));
end
exhaustion = struct( ...
	'occurred', ~isempty(exhausted_units), ...
	'unit_indices', exhausted_units, ...
	'parent_indices', exhausted_parent_indices, ...
	'segment_indices', exhausted_segment_indices, ...
	'max_attempts_per_unit', max_attempts);

acceptance = struct( ...
	'enabled', true, ...
	'criterion', ['round batches; each retry unit passes hard variance/HOCBF ', ...
		'and any explicitly enabled final geometry checks'], ...
	'tolerance', tolerance, ...
	'base_seed', base_seed, ...
	'seed_scope', char(seed_scope), ...
	'retry_scope', char(retry_scope), ...
	'max_attempts_per_trajectory', max_attempts, ...
	'parallel_trajectory_batch_size', batch_size, ...
	'accepted_seeds', accepted_seeds, ...
	'attempt_counts', attempt_counts, ...
	'search_max_excess', search_max_excess, ...
	'final_max_excess', final_max_excess, ...
	'final_min_margin', -final_max_excess, ...
	'final_min_hocbf_psi0', final_min_hocbf_psi0, ...
	'final_min_hocbf_psi1', final_min_hocbf_psi1, ...
	'final_min_track_point_h', final_min_track_point_h, ...
	'final_min_obstacle_point_h', final_min_obstacle_point_h, ...
	'final_min_track_line_h', final_min_track_line_h, ...
	'final_min_obstacle_line_h', final_min_obstacle_line_h, ...
	'final_track_line_filter_enabled', track_line_filter_enabled, ...
	'final_obstacle_line_filter_enabled', obstacle_line_filter_enabled, ...
	'final_track_line_tolerance', struct_field_default(filter_cfg,'final_track_line_tolerance',1e-8), ...
	'final_obstacle_line_tolerance', struct_field_default(filter_cfg,'final_obstacle_line_tolerance',1e-8), ...
	'final_track_point_filter_enabled', track_point_filter_enabled, ...
	'final_track_point_tolerance', track_point_tolerance, ...
	'final_track_point_include_margin', track_point_include_margin, ...
	'final_track_point_acceptance_margin', track_acceptance_margin, ...
	'final_track_point_margin_reference', track_margin_reference, ...
	'final_obstacle_point_filter_enabled', obstacle_point_filter_enabled, ...
	'final_obstacle_point_tolerance', obstacle_point_tolerance, ...
	'final_geometry_internal_points_only', geometry_internal_points_only, ...
	'final_geometry_state_mode', char(geometry_state_mode), ...
	'geometry_point_indices', geometry_point_indices, ...
	'beta_cap', beta_cap, ...
	'terminal_hard_time_mask', terminal_hard_time_mask, ...
	'hocbf_hard_time_mask', hocbf_hard_time_mask, ...
	'hard_time_end', max(times(terminal_hard_time_mask)), ...
	'fixed_hbar0', fixed_hbar0, ...
	'accepted_count', nnz(all(isfinite(accepted_seeds), 2)), ...
	'total_candidate_count', sum(attempt_counts));
acceptance.total_segment_candidate_count = sum(attempt_counts) * n_segments;
if segment_retry
	to_matrix = @(v) reshape(v,full_segment_count,full_parent_count)';
	accepted_seeds = to_matrix(accepted_seeds);
	acceptance.accepted_seeds = accepted_seeds;
	acceptance.segment_attempt_counts = to_matrix(attempt_counts);
	acceptance.attempt_counts = max(acceptance.segment_attempt_counts,[],2);
	max_fields = {'search_max_excess','final_max_excess'};
	min_fields = {'final_min_margin','final_min_hocbf_psi0', ...
		'final_min_hocbf_psi1','final_min_track_point_h','final_min_obstacle_point_h', ...
		'final_min_track_line_h','final_min_obstacle_line_h'};
	for field = [max_fields,min_fields]
		name = field{1};
		per_segment = to_matrix(acceptance.(name));
		acceptance.(['segment_',name]) = per_segment;
		if any(strcmp(name,max_fields))
			acceptance.(name) = max(per_segment,[],2);
		else
			acceptance.(name) = min(per_segment,[],2);
		end
	end
	acceptance.accepted_count = nnz(all(isfinite(accepted_seeds), 2));
	acceptance.accepted_segment_count = nnz(isfinite(accepted_seeds));
	acceptance.candidate_count_unit = 'segment';
	acceptance.criterion = ['independent segment retries; every segment must pass ', ...
		'hard variance/HOCBF and enabled final point/line checks'];
	complete_parent_count = nnz(all(isfinite(accepted_seeds), 2));
	fprintf('Segment-local filter complete: %d/%d full trajectories, %d/%d accepted segments, %d segment attempts.\n', ...
		complete_parent_count,full_parent_count,nnz(isfinite(accepted_seeds)), ...
		n_parents,sum(attempt_counts));
else
fprintf(['Parallel-batch variance seed filter complete: %d/%d accepted, ', ...
	'%d candidates tested, worst terminal excess %.3g, minimum HOCBF ', ...
	'psi0 %.3g, psi1 %.3g (tolerance %.3g).\n'], ...
	n_parents, n_parents, sum(attempt_counts), max(final_max_excess), ...
	min(final_min_hocbf_psi0), min(final_min_hocbf_psi1), tolerance);
end
end

function [times, path, diagnostics, x_init, seeds_out, acceptance, ...
	uncertainty, exhaustion] = replay_fixed_seed_rollouts( ...
	model_collection, constraint, fixed_seeds, full_parent_count, ...
	full_segment_count, internal_parent_count, internal_segment_count, ...
	state_dim, t0, t1, n_steps, ...
	refine_cfg, parallel_cfg, filter_cfg, seed_scope, retry_scope, ...
	segment_retry, fixed_hbar0)
% Replay an archived accepted seed layout exactly once. This mode is used
% for matched method comparisons: it never searches for replacement seeds.
fixed_seeds = double(fixed_seeds);
if seed_scope == "segment"
	expected_size = [full_parent_count, full_segment_count];
else
	expected_size = [full_parent_count, 1];
end
if ~isequal(size(fixed_seeds), expected_size)
	error('fixed_seed_matrix has size %s; expected %s.', ...
		mat2str(size(fixed_seeds)), mat2str(expected_size));
end
if any(~isfinite(fixed_seeds), 'all') || any(fixed_seeds < 0, 'all') || ...
		any(fixed_seeds > double(intmax('uint32')), 'all') || ...
		any(fixed_seeds ~= floor(fixed_seeds), 'all')
	error('fixed_seed_matrix must contain finite uint32-compatible integers.');
end

if segment_retry
	internal_seeds = reshape(fixed_seeds', [], 1);
else
	internal_seeds = fixed_seeds;
end
if size(internal_seeds, 1) ~= internal_parent_count || ...
		size(internal_seeds, 2) ~= internal_segment_count
	error('Internal fixed seed layout is inconsistent with rollout grouping.');
end
x_init = fixed_seed_initial_states(internal_seeds, state_dim, seed_scope);
replay_constraint = constraint;
replay_constraint.seed_filter_reject_failed_rollouts = true;
fprintf(['Fixed-seed replay: %d parent units, %d segment samples; ', ...
	'no seed search or replacement.\n'], internal_parent_count, ...
	internal_parent_count * internal_segment_count);
[times, path, diagnostics] = rk4_rollout(model_collection, x_init, ...
	t0, t1, n_steps, replay_constraint, refine_cfg, parallel_cfg);
failed = struct_field_default(diagnostics, 'failed_sample_mask', ...
	false(size(path, 2), 1));
failed = failed(:);
if any(failed) || any(~isfinite(path), 'all')
	nonfinite_rows = reshape(any(any(~isfinite(path), 1), 3), [], 1);
	failed_rows = find(failed | nonfinite_rows);
	error(['Archived fixed-seed replay failed for rollout rows %s. ', ...
		'Seeds were not replaced.'], mat2str(failed_rows(:)'));
end
uncertainty = evaluate_rollout_uncertainty(model_collection, times, path);
beta = aggregate_variance_values(uncertainty);
cap = terminal_variance_cap(times, constraint);
terminal_mask = terminal_variance_hard_mask(times, constraint);

unit_count = internal_parent_count;
attempt_counts = ones(unit_count, 1);
final_max_excess = nan(unit_count, 1);
final_min_psi0 = nan(unit_count, 1);
final_min_psi1 = nan(unit_count, 1);
for unit_idx = 1:unit_count
	rows = (unit_idx - 1) * internal_segment_count + ...
		(1:internal_segment_count);
	unit_beta = beta(:, rows);
	final_max_excess(unit_idx) = max( ...
		unit_beta(terminal_mask, :) - cap(terminal_mask), [], 'all');
	[final_min_psi0(unit_idx), final_min_psi1(unit_idx)] = ...
		integral_hocbf_margins(times, unit_beta, constraint);
end
nan_margin = nan(unit_count, 1);
acceptance = struct( ...
	'enabled', true, ...
	'criterion', 'archived fixed-seed replay without seed search or replacement', ...
	'tolerance', struct_field_default(filter_cfg, 'tolerance', 1e-8), ...
	'base_seed', struct_field_default(filter_cfg, 'base_seed', nan), ...
	'seed_scope', char(seed_scope), ...
	'retry_scope', char(retry_scope), ...
	'max_attempts_per_trajectory', 1, ...
	'parallel_trajectory_batch_size', unit_count, ...
	'accepted_seeds', internal_seeds, ...
	'attempt_counts', attempt_counts, ...
	'search_max_excess', final_max_excess, ...
	'final_max_excess', final_max_excess, ...
	'final_min_margin', -final_max_excess, ...
	'final_min_hocbf_psi0', final_min_psi0, ...
	'final_min_hocbf_psi1', final_min_psi1, ...
	'final_min_track_point_h', nan_margin, ...
	'final_min_obstacle_point_h', nan_margin, ...
	'final_min_track_line_h', nan_margin, ...
	'final_min_obstacle_line_h', nan_margin, ...
	'beta_cap', cap, ...
	'terminal_hard_time_mask', terminal_mask, ...
	'hocbf_hard_time_mask', false(size(times)), ...
	'hard_time_end', max(times(terminal_mask)), ...
	'fixed_hbar0', fixed_hbar0, ...
	'accepted_count', full_parent_count, ...
	'total_candidate_count', unit_count, ...
	'total_segment_candidate_count', unit_count * internal_segment_count, ...
	'fixed_seed_replay', true);
if segment_retry
	to_matrix = @(v) reshape(v, full_segment_count, full_parent_count)';
	seeds_out = to_matrix(internal_seeds);
	acceptance.accepted_seeds = seeds_out;
	acceptance.segment_attempt_counts = to_matrix(attempt_counts);
	acceptance.attempt_counts = max(acceptance.segment_attempt_counts, [], 2);
	fields = {'search_max_excess', 'final_max_excess', 'final_min_margin', ...
		'final_min_hocbf_psi0', 'final_min_hocbf_psi1', ...
		'final_min_track_point_h', 'final_min_obstacle_point_h', ...
		'final_min_track_line_h', 'final_min_obstacle_line_h'};
	for idx = 1:numel(fields)
		name = fields{idx};
		values = to_matrix(acceptance.(name));
		acceptance.(['segment_', name]) = values;
		if any(strcmp(name, {'search_max_excess', 'final_max_excess'}))
			acceptance.(name) = max(values, [], 2);
		else
			acceptance.(name) = min(values, [], 2);
		end
	end
	acceptance.accepted_segment_count = numel(seeds_out);
	acceptance.candidate_count_unit = 'segment';
else
	seeds_out = internal_seeds;
end
exhaustion = struct('occurred', false, 'unit_indices', zeros(0, 1), ...
	'parent_indices', zeros(0, 1), 'segment_indices', zeros(0, 1), ...
	'max_attempts_per_unit', 1);
diagnostics.seed_filter_includes_rejected_attempts = false;
fprintf('Fixed-seed replay complete: all %d archived seeds retained.\n', ...
	numel(seeds_out));
end

function x_init = fixed_seed_initial_states(seeds, state_dim, seed_scope)
n_parents = size(seeds, 1);
n_segments = size(seeds, 2);
x_init = zeros(n_parents * n_segments, state_dim);
for parent_idx = 1:n_parents
	rows = (parent_idx - 1) * n_segments + (1:n_segments);
	if seed_scope == "segment"
		for segment_idx = 1:n_segments
			stream = RandStream('mt19937ar', 'Seed', seeds(parent_idx, segment_idx));
			x_init(rows(segment_idx), :) = randn(stream, 1, state_dim);
		end
	else
		stream = RandStream('mt19937ar', 'Seed', seeds(parent_idx, 1));
		x_init(rows, :) = randn(stream, n_segments, state_dim);
	end
end
end

function [bad_rows,track_h,obstacle_h]=final_polyline_join_checks( ...
    states,transform,n_points,n_segments,constraint,filter)
physical=states.*transform.std(:)'+transform.mean(:)';
feature_dim=size(physical,2)/n_points;
if strcmpi(struct_field_default(filter,'final_geometry_state_mode','absolute'),'increment')
    physical=local_increment_rows_to_global(physical,feature_dim,n_points);
end
join_filter=filter;join_filter.final_geometry_state_mode='absolute';
identity=struct('mean',zeros(4,1),'std',ones(4,1));
track_h=inf(size(states,1),1);obstacle_h=track_h;bad_rows=zeros(0,1);
for row=1:size(states,1)
    if mod(row-1,n_segments)==0;continue;end
    previous_end=physical(row-1,(n_points-1)*feature_dim+(1:2));
    current_second=physical(row,feature_dim+(1:2));
    result=evaluate_final_polyline_safety([previous_end current_second], ...
        identity,2,constraint,join_filter);
    track_h(row)=result.track_min_h;obstacle_h(row)=result.obstacle_min_h;
    if ~result.valid;bad_rows(end+1,1)=row;end %#ok<AGROW>
end
end

function [x_init, seeds] = seeded_initial_states(base_seed, parent_indices, ...
	attempts, max_attempts, n_segments, state_dim, seed_scope, original_group_size)
n_candidates = numel(parent_indices);
x_init = zeros(n_candidates * n_segments, state_dim);
if seed_scope == "segment"
	seeds = zeros(n_candidates, n_segments);
else
	seeds = zeros(n_candidates, 1);
end
for candidate_idx = 1:n_candidates
	if original_group_size > 1
		original_parent = floor((double(parent_indices(candidate_idx))-1) / original_group_size);
		original_segment = mod(double(parent_indices(candidate_idx))-1,original_group_size);
		candidate_seed_row = double(base_seed) + ...
			(original_parent*double(max_attempts)+double(attempts(candidate_idx))-1) * ...
			original_group_size + original_segment;
	elseif seed_scope == "segment"
		attempt_offset = ((double(parent_indices(candidate_idx)) - 1) .* ...
			double(max_attempts) + double(attempts(candidate_idx)) - 1) .* ...
			double(n_segments);
		candidate_seed_row = double(base_seed) + attempt_offset + ...
			(0:(n_segments - 1));
	else
		candidate_seed_row = double(base_seed) + ...
			(double(parent_indices(candidate_idx)) - 1) * ...
			double(max_attempts) + double(attempts(candidate_idx)) - 1;
	end
	if any(candidate_seed_row < 0) || ...
			any(candidate_seed_row > double(intmax('uint32'))) || ...
			any(candidate_seed_row ~= floor(candidate_seed_row))
		error('Generated seed for trajectory %d is outside uint32 range.', ...
			parent_indices(candidate_idx));
	end
	seeds(candidate_idx, :) = candidate_seed_row;
	rows = (candidate_idx - 1) * n_segments + (1:n_segments);
	if seed_scope == "segment"
		for segment_idx = 1:n_segments
			stream = RandStream('mt19937ar', 'Seed', ...
				candidate_seed_row(segment_idx));
			x_init(rows(segment_idx), :) = randn(stream, 1, state_dim);
		end
	else
		stream = RandStream('mt19937ar', 'Seed', candidate_seed_row);
		x_init(rows, :) = randn(stream, n_segments, state_dim);
	end
end
end

function text_value = seed_text(seed_row)
seed_row = seed_row(:)';
if isscalar(seed_row)
	text_value = sprintf('%.0f', seed_row);
else
	text_value = sprintf('[%.0f..%.0f]', seed_row(1), seed_row(end));
end
end

function batch_constraint = subset_sample_targets( ...
	base_constraint, parent_indices, n_segments)
batch_constraint = base_constraint;
row_indices = zeros(numel(parent_indices) * n_segments, 1);
for idx = 1:numel(parent_indices)
	rows = (idx - 1) * n_segments + (1:n_segments);
	row_indices(rows) = (parent_indices(idx) - 1) * n_segments + (1:n_segments);
end
per_sample_fields = { ...
	'anchor_clf_targets', ...
	'track_boundary_reference_s_min_targets', ...
	'track_boundary_reference_s_max_targets'};
for field_idx = 1:numel(per_sample_fields)
	field_name = per_sample_fields{field_idx};
	if isfield(batch_constraint, field_name)
		values = batch_constraint.(field_name);
		batch_constraint.(field_name) = values(row_indices, :);
	end
end
% Preserve requested positions (zero means absent in this partial batch),
% so live plots retain original parent/segment grouping after accepted rows
% have been removed from the retry queue.
if isfield(batch_constraint,'live_trajectory_plot_sample_indices')
	requested = batch_constraint.live_trajectory_plot_sample_indices;
	[~,local_indices] = ismember(requested,row_indices);
	batch_constraint.live_trajectory_plot_sample_indices = reshape(local_indices,size(requested));
	batch_constraint.live_trajectory_original_sample_indices = row_indices;
	if ~any(local_indices)
		batch_constraint.live_trajectory_plot_enabled = false;
	end
end
end

function diag = remap_batch_diagnostics(diag, original_rows, attempts)
if ~isfield(diag,'hocbf'); return; end
H = diag.hocbf;
if isfield(H,'trace_sample_idx')
	local = H.trace_sample_idx;
	valid = isfinite(local) & local >= 1 & local <= numel(original_rows) & local == floor(local);
	H.trace_seed_attempt = nan(size(local));
	H.trace_seed_attempt(valid) = attempts(local(valid));
	H.trace_sample_idx(valid) = original_rows(local(valid));
end
if isfield(H,'min_obstacle_h_sample_idx')
	i=H.min_obstacle_h_sample_idx;
	if isscalar(i) && isfinite(i) && i>=1 && i<=numel(original_rows) && i==floor(i)
		H.min_obstacle_h_sample_idx=original_rows(i);
	end
end
diag.hocbf=H;
end

function beta_cap = terminal_variance_cap(times, constraint_cfg)
beta_final = constraint_cfg.terminal_variance_beta_final;
hbar0 = constraint_cfg.terminal_variance_ptzf_hbar0_fixed;
gamma = constraint_cfg.terminal_variance_ptzf_gamma;
terminal_time = struct_field_default(constraint_cfg, ...
	'terminal_variance_ptzf_terminal_time', ...
	struct_field_default(constraint_cfg, 'rollout_t_max', times(end)));
time_shift = struct_field_default(constraint_cfg, ...
	'terminal_variance_ptzf_time_shift', 0.0);
t_eff = (times(:) + time_shift) ./ terminal_time;
hbar = zeros(size(t_eff));
before_terminal = t_eff < 1.0;
remaining = 1.0 - t_eff(before_terminal);
hbar(before_terminal) = hbar0 .* exp(-gamma .* ...
	(t_eff(before_terminal) ./ remaining));
beta_cap = beta_final + hbar;
end

function hard_time_mask = terminal_variance_hard_mask(times, constraint_cfg)
ptcbf_end_time = struct_field_default(constraint_cfg, ...
	'terminal_variance_ptcbf_end_time', inf);
hard_time_mask = false(size(times(:)));
for time_idx = 1:numel(times)
	t_now = times(time_idx);
	ptcbf_active = struct_field_default(constraint_cfg, ...
		'ptcbf_enabled', false) && t_now < ptcbf_end_time;
	has_slack = slack_enabled_for_constraints( ...
		constraint_cfg, "terminal", t_now);
	hard_time_mask(time_idx) = ptcbf_active && ~has_slack;
end
if ~any(hard_time_mask)
	error(['Variance seed filtering found no rollout times at ', ...
		'which the terminal variance PTCBF is active without slack.']);
end
end

function [min_psi0, min_psi1, hard_time_mask] = ...
	integral_hocbf_margins(times, beta, constraint_cfg)
if ~struct_field_default(constraint_cfg, 'hocbf_enabled', false)
	min_psi0 = inf;
	min_psi1 = inf;
	hard_time_mask = false(size(times(:)));
	return;
end
hocbf_end_time = struct_field_default(constraint_cfg, ...
	'hocbf_filter_end_time', inf);
hard_time_mask = false(size(times(:)));
for time_idx = 1:numel(times)
	t_now = times(time_idx);
	has_slack = slack_enabled_for_constraints( ...
		constraint_cfg, "integral", t_now);
	hard_time_mask(time_idx) = t_now < hocbf_end_time && ~has_slack;
end
if ~any(hard_time_mask)
	min_psi0 = inf;
	min_psi1 = inf;
	return;
end

budget = constraint_cfg.integral_uncertainty_budget;
hbar = constraint_cfg.hocbf_relaxation_bound;
psi1_margin = constraint_cfg.psi1_margin;
min_psi0 = inf;
min_psi1 = inf;
for segment_idx = 1:size(beta, 2)
	cumulative_beta = cumtrapz(times(:), beta(:, segment_idx));
	psi0 = budget .* times(:) - cumulative_beta + hbar;
	alpha1 = (beta(1, segment_idx) - budget + psi1_margin) ./ hbar;
	psi1 = budget - beta(:, segment_idx) + alpha1 .* psi0;
	min_psi0 = min(min_psi0, min(psi0(hard_time_mask)));
	min_psi1 = min(min_psi1, min(psi1(hard_time_mask)));
end
end

function min_track_h = final_track_point_margin(standardized_segments, ...
	data_transform, n_segments, n_points_per_segment, constraint_cfg, enabled, ...
	point_indices, geometry_state_mode, acceptance_margin)
min_track_h = inf;
if ~enabled || ~struct_field_default(constraint_cfg, ...
		'track_boundary_enabled', false)
	return;
end
geometry = constraint_cfg.track_boundary_geometry;
if ~isfield(geometry, 'implicit_fields')
	error(['Final discrete track-point filtering requires ', ...
		'track_boundary_constraint_method=global_implicit_fields.']);
end
if size(standardized_segments, 1) == 1
	standardized_segments = reshape(standardized_segments, 1, []);
end
physical_segments = standardized_segments .* data_transform.std' + ...
	data_transform.mean';
feature_dim = size(physical_segments, 2) / n_points_per_segment;
if feature_dim ~= floor(feature_dim) || feature_dim < 2
	error('Cannot reconstruct final segment points for track-point filtering.');
end
if geometry_state_mode == "increment"
	physical_segments = local_increment_rows_to_global(physical_segments, ...
		feature_dim, n_points_per_segment);
end
% First-level acceptance may subtract the configured inward margin;
% levels without that filter option use the original implicit boundary.
% This changes acceptance only, never the margin passed to the control QP.
for segment_idx = 1:n_segments
	segment_points = reshape(physical_segments(segment_idx, :), ...
		feature_dim, [])';
	for point_idx = point_indices
		p = segment_points(point_idx, 1:2)';
		h_left = evaluate_track_implicit_field( ...
			geometry.implicit_fields, 1, p) - acceptance_margin;
		h_right = evaluate_track_implicit_field( ...
			geometry.implicit_fields, 2, p) - acceptance_margin;
		min_track_h = min([min_track_h, h_left, h_right]);
	end
end
end

function min_obstacle_h = final_obstacle_point_margin(standardized_segments, ...
	data_transform, n_segments, n_points_per_segment, constraint_cfg, enabled, ...
	point_indices, geometry_state_mode)
min_obstacle_h = inf;
if ~enabled || ~struct_field_default(constraint_cfg, 'obstacle_enabled', false)
	return;
end
obstacle = struct_field_default(constraint_cfg, ...
	'obstacle_physical_geometry', constraint_cfg.obstacle_geometry);
if size(standardized_segments, 1) == 1
	standardized_segments = reshape(standardized_segments, 1, []);
end
physical_segments = standardized_segments .* data_transform.std' + ...
	data_transform.mean';
feature_dim = size(physical_segments, 2) / n_points_per_segment;
if feature_dim ~= floor(feature_dim) || feature_dim < 2
	error('Cannot reconstruct final segment points for obstacle filtering.');
end
if geometry_state_mode == "increment"
	physical_segments = local_increment_rows_to_global(physical_segments, ...
		feature_dim, n_points_per_segment);
end
for segment_idx = 1:n_segments
	segment_points = reshape(physical_segments(segment_idx, :), ...
		feature_dim, [])';
	for point_idx = point_indices
		p = segment_points(point_idx, 1:2)';
		for obstacle_idx = 1:size(obstacle.centers, 2)
			h = obstacle_level_and_gradient(p, obstacle, obstacle_idx);
			min_obstacle_h = min(min_obstacle_h, h);
		end
	end
end
end

function merged = merge_top_level_diagnostics(diag_list)
diag_list = diag_list(~cellfun(@isempty, diag_list));
if isempty(diag_list)
	merged = struct();
	return;
end
merged = diag_list{1};
merged.max_cumulative_variance = max(cellfun( ...
	@(d) d.max_cumulative_variance, diag_list));
merged.rollout_elapsed_seconds = sum(cellfun( ...
	@(d) d.rollout_elapsed_seconds, diag_list));
merged.hocbf = merge_rollout_diagnostics(cellfun( ...
	@(d) d.hocbf, diag_list, 'UniformOutput', false));
end
