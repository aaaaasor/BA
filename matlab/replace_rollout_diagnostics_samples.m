function combined = replace_rollout_diagnostics_samples(combined, local, global_rows)
%REPLACE_ROLLOUT_DIAGNOSTICS_SAMPLES Replace traces for rerun sample rows.
% Elapsed time remains cumulative so Time includes every rejected attempt.

global_rows = global_rows(:);
old_elapsed = struct_field_default(combined, 'rollout_elapsed_seconds', 0.0);
local_elapsed = struct_field_default(local, 'rollout_elapsed_seconds', 0.0);
if isfield(combined, 'hocbf') && isfield(local, 'hocbf')
	base_h = combined.hocbf;
	local_h = local.hocbf;
	if isfield(base_h, 'trace_sample_idx')
		keep = ~ismember(base_h.trace_sample_idx(:), global_rows);
		base_h = filter_trace_rows(base_h, keep);
	end
	if isfield(local_h, 'trace_sample_idx')
		idx = local_h.trace_sample_idx(:);
		valid = isfinite(idx) & idx >= 1 & idx <= numel(global_rows) & ...
			idx == floor(idx);
		mapped = idx;
		mapped(valid) = global_rows(idx(valid));
		local_h.trace_sample_idx = reshape(mapped, ...
			size(local_h.trace_sample_idx));
	end
	combined.hocbf = merge_rollout_diagnostics({base_h, local_h});
end
combined.rollout_elapsed_seconds = old_elapsed + local_elapsed;
combined.max_cumulative_variance = max( ...
	struct_field_default(combined, 'max_cumulative_variance', -inf), ...
	struct_field_default(local, 'max_cumulative_variance', -inf));
end

function value = filter_trace_rows(value, keep)
n = numel(keep);
names = fieldnames(value);
for idx = 1:numel(names)
	name = names{idx};
	if ~strncmp(name, 'trace_', 6)
		continue;
	end
	field_value = value.(name);
	if ~isempty(field_value) && size(field_value, 1) == n
		value.(name) = field_value(keep, :);
	end
end
end
