% 统计 LoG-GP 在一次 rollout 里被触发了多少次、每次激活了多少个 local GP、
% 每次调用花了多久。action: 'reset'（rollout 开始前清零）、
% 'record'（predict_gp_stats.m 每次调用后记一笔）、'report'（打印汇总并返回结构体）。
function out = loggp_call_stats(action, varargin)
persistent call_count total_elapsed total_n_local_gp min_n_local_gp max_n_local_gp
if isempty(call_count)
	call_count = 0;
	total_elapsed = 0.0;
	total_n_local_gp = 0;
	min_n_local_gp = inf;
	max_n_local_gp = 0;
end
out = [];
switch action
	case 'reset'
		call_count = 0;
		total_elapsed = 0.0;
		total_n_local_gp = 0;
		min_n_local_gp = inf;
		max_n_local_gp = 0;
	case 'record'
		elapsed = varargin{1};
		n_local_gp = varargin{2};
		call_count = call_count + 1;
		total_elapsed = total_elapsed + elapsed;
		total_n_local_gp = total_n_local_gp + n_local_gp;
		min_n_local_gp = min(min_n_local_gp, n_local_gp);
		max_n_local_gp = max(max_n_local_gp, n_local_gp);
	case 'collect'
		% 与 'report' 相同，但不打印。rk4_rollout 逐 sample 收集（并行时
		% 每个 worker 有自己的 persistent），最后合并成一条汇总再打印。
		out = struct('call_count', call_count, ...
			'mean_n_local_gp', total_n_local_gp / max(call_count, 1), ...
			'min_n_local_gp', min_n_local_gp, ...
			'max_n_local_gp', max_n_local_gp, ...
			'mean_elapsed_seconds', total_elapsed / max(call_count, 1), ...
			'total_elapsed_seconds', total_elapsed);
	case 'report'
		if call_count == 0
			fprintf('LoG-GP call stats: no calls recorded.\n');
		else
			fprintf(['LoG-GP call stats: triggered %d times, local GP activated ', ...
				'per call mean/min/max = %.2f / %d / %d, elapsed per call ', ...
				'mean = %.6f s, total elapsed = %.3f s\n'], ...
				call_count, total_n_local_gp / call_count, min_n_local_gp, ...
				max_n_local_gp, total_elapsed / call_count, total_elapsed);
		end
		out = struct('call_count', call_count, ...
			'mean_n_local_gp', total_n_local_gp / max(call_count, 1), ...
			'min_n_local_gp', min_n_local_gp, ...
			'max_n_local_gp', max_n_local_gp, ...
			'mean_elapsed_seconds', total_elapsed / max(call_count, 1), ...
			'total_elapsed_seconds', total_elapsed);
	otherwise
		error('Unknown loggp_call_stats action: %s', action);
end
end
