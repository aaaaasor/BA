% 按配置准备并行池，返回本次 rollout 实际可用的 worker 数。
% 返回 0 表示串行执行（配置关闭、没有 Parallel Computing Toolbox、或只有一条
% 轨迹）。parallel_cfg 字段见 get_config.m 的 cfg.parallel。
function worker_count = ensure_parallel_pool(parallel_cfg, n_tasks)
if nargin < 2 || isempty(n_tasks)
	n_tasks = inf;
end
worker_count = 0;
if isempty(parallel_cfg) || ...
		~struct_field_default(parallel_cfg, 'enabled', false)
	return;
end
fallback_to_serial = struct_field_default(parallel_cfg, ...
	'fallback_to_serial', true);
if n_tasks <= 1
	% 只有一条轨迹时并行没有意义，直接串行（也省掉启动池的开销）。
	return;
end
if ~parallel_toolbox_available()
	if fallback_to_serial
		warning('ensure_parallel_pool:ToolboxUnavailable', ...
			['Parallel Computing Toolbox unavailable; ', ...
			'falling back to serial rollout.']);
		return;
	end
	error('ensure_parallel_pool:ToolboxUnavailable', ...
		['cfg.parallel.enabled is true but Parallel Computing Toolbox ', ...
		'is unavailable.']);
end

requested_workers = struct_field_default(parallel_cfg, 'num_workers', 0);
pool = gcp('nocreate');
try
	% 显式请求 worker 数时始终服从配置，不再按当前 rollout 的轨迹数截断。
	% 这样前一层即使只有 5 条轨迹，也会建立 8-worker 池供后续层复用。
	if ~isempty(pool) && requested_workers > 0 && ...
			pool.NumWorkers ~= requested_workers
		fprintf(['  Replacing the existing parallel pool with %d workers ', ...
			'(currently %d).\n'], requested_workers, pool.NumWorkers);
		delete(pool);
		pool = [];
	end
	if isempty(pool)
		if requested_workers > 0
			target_workers = requested_workers;
			% 保存的 profile 里 NumWorkers 可能小于这里要的核数（MATLAB 的
			% 'Processes' profile 默认值不一定等于物理核数），直接在内存里的
			% cluster 副本上调大，不动用户保存的 profile。
			cluster = parcluster();
			try
				if cluster.NumWorkers < target_workers
					cluster.NumWorkers = target_workers;
				end
			catch
				% Threads 等不支持改 NumWorkers 的 profile：按原样启动。
			end
			pool = parpool(cluster, target_workers);
		else
			pool = parpool();
		end
	end
catch pool_err
	if fallback_to_serial
		warning('ensure_parallel_pool:PoolStartFailed', ...
			'Could not prepare a parallel pool (%s); running serially.', ...
			pool_err.message);
		return;
	end
	rethrow(pool_err);
end
worker_count = pool.NumWorkers;
end

function available = parallel_toolbox_available()
available = exist('parpool', 'file') == 2 && ...
	exist('gcp', 'file') == 2 && ...
	license('test', 'Distrib_Computing_Toolbox') == 1;
end
