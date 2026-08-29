function beta_bound = seed_filter_initial_variance_bound( ...
	model_collection, filter_cfg, n_parents, n_segments, state_dim, t0)
%SEED_FILTER_INITIAL_VARIANCE_BOUND Deterministic common initial GP bound.
% Evaluate the first-attempt seed population used by the rejection sampler.
% This is much tighter than the GP prior variance bound for the first-level
% model and therefore keeps its prescribed-time envelope numerically useful.

base_seed = struct_field_required(filter_cfg, 'base_seed');
max_attempts = struct_field_default(filter_cfg, ...
	'max_attempts_per_trajectory', 100);
envelope_factor = struct_field_default(filter_cfg, ...
	'initial_beta_envelope_factor', 1.0);
if ~isscalar(envelope_factor) || ~isfinite(envelope_factor) || ...
		envelope_factor < 1.0
	error('initial_beta_envelope_factor must be a finite scalar >= 1.');
end

max_beta0 = 0.0;
n_outputs = numel(model_collection.model.output_models);
for parent_idx = 1:n_parents
	seed = double(base_seed) + ...
		(double(parent_idx) - 1) * double(max_attempts);
	stream = RandStream('mt19937ar', 'Seed', seed);
	states = randn(stream, n_segments, state_dim);
	for segment_idx = 1:n_segments
		x0 = states(segment_idx, :)';
		beta0 = 0.0;
		for output_idx = 1:n_outputs
			beta0 = beta0 + model_collection.model.output_models{output_idx}. ...
				predict_variance([t0; x0]);
		end
		max_beta0 = max(max_beta0, beta0);
	end
end
beta_bound = envelope_factor * max_beta0;
fprintf(['  Seed-filter initial beta envelope: observed max %.3f, ', ...
	'factor %.3g, fixed upper bound %.3f.\n'], ...
	max_beta0, envelope_factor, beta_bound);
end
