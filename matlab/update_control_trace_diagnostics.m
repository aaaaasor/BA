function trace = update_control_trace_diagnostics(trace, info, ...
		sample_idx, step_idx, stage_name)
stage_idx = find(strcmp(stage_name, {'k1', 'k2', 'k3', 'k4'}), 1);
if isempty(stage_idx)
	error('update_control_trace_diagnostics:UnknownStage', ...
		'Unknown RK4 stage %s.', stage_name);
end
row_idx = trace.n_trace_entries + 1;
if row_idx > trace.preallocated_trace_capacity
	error('update_control_trace_diagnostics:TraceCapacityExceeded', ...
		'Control trace capacity was exceeded.');
end
trace.n_trace_entries = row_idx;
trace.trace_sample_idx(row_idx) = sample_idx;
trace.trace_step_idx(row_idx) = step_idx;
trace.trace_stage_idx(row_idx) = stage_idx;
trace.trace_t(row_idx) = info.t;
trace.trace_mu(row_idx, :) = reshape(info.mu, 1, []);
trace.trace_v(row_idx, :) = reshape(info.v, 1, []);
trace.trace_u(row_idx, :) = reshape(info.u, 1, []);
trace.trace_u_ptclf_reference(row_idx, :) = ...
	reshape(info.u_ptclf_reference, 1, []);
trace.trace_u_after_ptcbf(row_idx, :) = ...
	reshape(info.u_after_ptcbf, 1, []);
trace.trace_u_ptcbf_correction(row_idx, :) = ...
	reshape(info.u_ptcbf_correction, 1, []);
trace.trace_u_hocbf_correction(row_idx, :) = ...
	reshape(info.u_hocbf_correction, 1, []);
end
