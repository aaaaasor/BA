% 合并逐 sample（并行时逐 worker）收集的 LoG-GP 调用统计，并打印与
% loggp_call_stats('report') 相同格式的汇总。stats_list 为 cell array。
function stats = merge_loggp_call_stats(stats_list, print_summary)
if nargin < 2
	print_summary = true;
end
stats_list = stats_list(~cellfun(@isempty, stats_list));
call_count = 0;
total_elapsed = 0.0;
total_n_local_gp = 0.0;
min_n_local_gp = inf;
max_n_local_gp = 0;
for idx = 1:numel(stats_list)
	stats_now = stats_list{idx};
	if stats_now.call_count == 0
		continue;
	end
	call_count = call_count + stats_now.call_count;
	total_elapsed = total_elapsed + stats_now.total_elapsed_seconds;
	total_n_local_gp = total_n_local_gp + ...
		stats_now.mean_n_local_gp * stats_now.call_count;
	min_n_local_gp = min(min_n_local_gp, stats_now.min_n_local_gp);
	max_n_local_gp = max(max_n_local_gp, stats_now.max_n_local_gp);
end
stats = struct('call_count', call_count, ...
	'mean_n_local_gp', total_n_local_gp / max(call_count, 1), ...
	'min_n_local_gp', min_n_local_gp, ...
	'max_n_local_gp', max_n_local_gp, ...
	'mean_elapsed_seconds', total_elapsed / max(call_count, 1), ...
	'total_elapsed_seconds', total_elapsed);
if ~print_summary
	return;
end
if call_count == 0
	fprintf('LoG-GP call stats: no calls recorded.\n');
else
	fprintf(['LoG-GP call stats: triggered %d times, local GP activated ', ...
		'per call mean/min/max = %.2f / %d / %d, elapsed per call ', ...
		'mean = %.6f s, total elapsed = %.3f s\n'], ...
		call_count, stats.mean_n_local_gp, min_n_local_gp, max_n_local_gp, ...
		stats.mean_elapsed_seconds, total_elapsed);
end
end
