function hocbf_diag = finalize_hocbf_diagnostics(hocbf_diag)
% Trim preallocated trace storage and remove private bookkeeping fields.
if ~isfield(hocbf_diag, 'n_trace_entries') || ...
		~isfield(hocbf_diag, 'preallocated_trace_capacity')
	return;
end

n_trace_entries = hocbf_diag.n_trace_entries;
field_names = fieldnames(hocbf_diag);
for field_idx = 1:numel(field_names)
	field_name = field_names{field_idx};
	if ~strncmp(field_name, 'trace_', 6)
		continue;
	end
	trace_value = hocbf_diag.(field_name);
	if size(trace_value, 1) < n_trace_entries
		error('finalize_hocbf_diagnostics:ShortTrace', ...
			'Trace field %s has %d rows, but %d entries were recorded.', ...
			field_name, size(trace_value, 1), n_trace_entries);
	end
	hocbf_diag.(field_name) = trace_value(1:n_trace_entries, :);
end

hocbf_diag = rmfield(hocbf_diag, ...
	{'n_trace_entries', 'preallocated_trace_capacity'});
end
