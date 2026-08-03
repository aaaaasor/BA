function trace = finalize_control_trace_diagnostics(trace)
% Trim the lightweight preallocated control trace.
n = trace.n_trace_entries;
names = fieldnames(trace);
for idx = 1:numel(names)
	name = names{idx};
	if strncmp(name, 'trace_', 6)
		trace.(name) = trace.(name)(1:n, :);
	end
end
trace = rmfield(trace, ...
	{'n_trace_entries', 'preallocated_trace_capacity'});
end
