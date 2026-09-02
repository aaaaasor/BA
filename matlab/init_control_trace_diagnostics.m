function trace = init_control_trace_diagnostics(trace_capacity, state_dim, u_only)
% Preallocate only the fields needed by the third-level u figure.
if nargin < 3
	u_only = false;
end
trace.trace_sample_idx = zeros(trace_capacity, 1);
trace.trace_step_idx = zeros(trace_capacity, 1);
trace.trace_stage_idx = zeros(trace_capacity, 1);
trace.trace_t = zeros(trace_capacity, 1);
trace.trace_u = zeros(trace_capacity, state_dim);
if ~u_only
	trace.trace_mu = zeros(trace_capacity, state_dim);
	trace.trace_v = zeros(trace_capacity, state_dim);
	trace.trace_u_ptclf_reference = zeros(trace_capacity, state_dim);
	trace.trace_u_after_ptcbf = zeros(trace_capacity, state_dim);
	trace.trace_u_ptcbf_correction = zeros(trace_capacity, state_dim);
	trace.trace_u_hocbf_correction = zeros(trace_capacity, state_dim);
end
trace.control_trace_u_only = logical(u_only);
trace.n_trace_entries = 0;
trace.preallocated_trace_capacity = trace_capacity;
end
