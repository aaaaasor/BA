% 把逐 sample 收集的 rollout 诊断结构体合并成一个，结果与串行时逐 sample
% 累加得到的结构体等价：trace_* 按 sample 顺序纵向拼接，min_*/max_* 取极值，
% 计数/耗时求和。对 hocbf 诊断和轻量 control trace 两种结构都适用。
function merged = merge_rollout_diagnostics(diag_list)
diag_list = diag_list(~cellfun(@isempty, diag_list));
if isempty(diag_list)
	merged = struct();
	return;
end
merged = diag_list{1};
if numel(diag_list) == 1
	return;
end

field_names = fieldnames(merged);
for field_idx = 1:numel(field_names)
	field_name = field_names{field_idx};
	values = cellfun(@(d) d.(field_name), diag_list, 'UniformOutput', false);
	if strncmp(field_name, 'trace_', 6)
		merged.(field_name) = vertcat(values{:});
		continue;
	end
	merged.(field_name) = merge_scalar_field(field_name, values);
end

% min_obstacle_h 的时间/sample/step 必须来自取到最小值的那条轨迹，
% 而不是各自独立取最小，所以单独按 argmin 覆盖。
companion_fields = {'min_obstacle_h_t', 'min_obstacle_h_sample_idx', ...
	'min_obstacle_h_step_idx'};
if isfield(merged, 'min_obstacle_h') && ...
		all(isfield(merged, companion_fields))
	h_values = cellfun(@(d) d.min_obstacle_h, diag_list);
	[~, argmin_idx] = min(h_values);
	for field_idx = 1:numel(companion_fields)
		merged.(companion_fields{field_idx}) = ...
			diag_list{argmin_idx}.(companion_fields{field_idx});
	end
end
end

function value = merge_scalar_field(field_name, values)
numeric_values = [values{:}];
if ~isnumeric(numeric_values) && ~islogical(numeric_values)
	value = values{1};
	return;
end
sum_fields = {'n_trace_entries', 'preallocated_trace_capacity'};
if any(strcmp(field_name, sum_fields)) || ...
		strncmp(field_name, 'total_', 6) || ...
		(numel(field_name) >= 6 && strcmp(field_name(end - 5:end), '_count'))
	value = sum(numeric_values);
elseif strcmp(field_name, 'obstacle_first_unsafe_t') || ...
		strncmp(field_name, 'min_', 4)
	value = min(numeric_values, [], 'omitnan');
elseif strncmp(field_name, 'max_', 4)
	value = max(numeric_values, [], 'omitnan');
else
	value = values{1};
end
if isempty(value)
	value = values{1};
end
end
