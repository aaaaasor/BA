% Roll out trajectory-space states with a fixed-step classical RK4 integrator.
% refine_cfg (可选): 结构体，字段 start_t/extra_steps，在 [start_t, t1] 区间内
% 额外插入 extra_steps 个更细的时间步（用于 PTZF 包络在 t->1 附近急剧收紧的
% 尾段），其余区间仍按 n_steps 均匀划分。不传则保持原来的均匀网格。
% parallel_cfg (可选): 见 get_config.m 的 cfg.parallel。默认(空或 enabled=false)
% 串行逐条 rollout；打开后每个 worker(核)一次算一条轨迹。两种模式的结果
% 完全一致——sample 之间本来就没有耦合，只有初始时刻的全局 hbar0 需要跨
% sample，而它在进入 sample 循环之前就已经算好。
function [times, path, diagnostics] = rk4_rollout(model_collection, ...
	x_init, t0, t1, n_steps, constraint_cfg, refine_cfg, parallel_cfg)

%% Allocate Path
if nargin >= 7 && ~isempty(refine_cfg) && ...
		struct_field_default(refine_cfg, 'extra_steps', 0) > 0
	refine_start_t = refine_cfg.start_t;
	extra_steps = refine_cfg.extra_steps;
	coarse_steps = max(round(n_steps * (refine_start_t - t0) / (t1 - t0)), 1);
	times_coarse = linspace(t0, refine_start_t, coarse_steps + 1)';
	times_fine = linspace(refine_start_t, t1, extra_steps + 1)';
	times = [times_coarse; times_fine(2:end)]; % 生成rollout所有时间点（尾段加密）
else
	times = linspace(t0, t1, n_steps + 1)'; % 生成rollout所有时间点
end
if nargin < 8
	parallel_cfg = [];
end
n_steps = numel(times) - 1;
dt_vec = diff(times); % 每一步的步长（非均匀网格时逐步不同）
n_samples = size(x_init, 1);
state_dim = size(x_init, 2);
path = zeros(n_steps + 1, n_samples, state_dim); % path(time_idx, sample_idx, state_idx)
constrained = ~isempty(constraint_cfg);
collect_diagnostics = constrained && struct_field_default( ...
	constraint_cfg, 'diagnostics', false);
collect_control_trace = constrained && struct_field_default( ...
	constraint_cfg, 'control_trace_enabled', false);
if constrained
	% rollout 实际只能跑到 t1(<1，PTZF公式在t=1处有奇点），把这个映射
	% 偏移量传给 terminal/CLF，让它们在 t=t1 时把 t_eff 当作精确的 1。
	constraint_cfg.ptzf_time_shift = max(1.0 - t1, 0.0);
end
diagnostics.max_cumulative_variance = 0.0;
diagnostics.rollout_elapsed_seconds = 0.0; % 初始化 roll-out 时间统计
loggp_call_stats('reset'); % 清零 LoG-GP 触发次数/local GP 激活数/耗时统计

%% Global Terminal Bound
rollout_timer = tic; % 从开始到现在总共多久
beta0_values = nan(n_samples, 1);
terminal_h0_global = nan;
if constrained
	for beta_sample_idx = 1:n_samples
		x0 = reshape(x_init(beta_sample_idx, :), [], 1);
		beta0_values(beta_sample_idx) = sum(arrayfun(@(i) ...
			model_collection.model.output_models{i}.predict_variance([times(1); x0]), ...
			1:numel(model_collection.model.output_models)));
	end
	if struct_field_default(constraint_cfg, ...
			'ptcbf_enabled', false)
		beta_final = constraint_cfg.terminal_variance_beta_final;
		terminal_margin = struct_field_default(constraint_cfg, ...
			'terminal_variance_ptzf_initial_margin', 1e-6);
		terminal_h0_global = max(max(beta0_values) - beta_final, 0.0) + ...
			terminal_margin;
		constraint_cfg.terminal_variance_ptzf_hbar0 = terminal_h0_global;
		fprintf(['  terminal global hbar0=%.3f from max beta0=%.3f, ', ...
			'beta_final=%.3f, margin=%.3g\n'], terminal_h0_global, ...
			max(beta0_values), beta_final, terminal_margin);
	end
end

%% RK4 Integration
% 每个 sample 是一条独立轨迹，因此并行模式下一个 worker 领一条。
ctx = struct( ...
	'times', times, ...
	'dt_vec', dt_vec, ...
	'n_steps', n_steps, ...
	'state_dim', state_dim, ...
	'constrained', constrained, ...
	'collect_diagnostics', collect_diagnostics, ...
	'collect_control_trace', collect_control_trace, ...
	'terminal_h0_global', terminal_h0_global);
worker_count = ensure_parallel_pool(parallel_cfg, n_samples);
sample_paths = cell(n_samples, 1);
sample_diags = cell(n_samples, 1);
sample_max_cumulative = zeros(n_samples, 1);
sample_loggp_stats = cell(n_samples, 1);
if worker_count > 0
	fprintf(['  Running %d trajectories in parallel on %d workers ', ...
		'(one trajectory per worker at a time)...\n'], ...
		n_samples, worker_count);
	parfor_opts = single_trajectory_parfor_options(worker_count);
	parfor (sample_idx = 1:n_samples, parfor_opts)
		[sample_paths{sample_idx}, sample_diags{sample_idx}, ...
			sample_max_cumulative(sample_idx), ...
			sample_loggp_stats{sample_idx}] = rollout_single_sample( ...
			model_collection, x_init, constraint_cfg, ctx, ...
			beta0_values(sample_idx), sample_idx);
	end
else
	batch_timer = tic; % 每隔10个sample打印一次进度时，用来显示这一批花了多少秒
	for sample_idx = 1:n_samples
		[sample_paths{sample_idx}, sample_diags{sample_idx}, ...
			sample_max_cumulative(sample_idx), ...
			sample_loggp_stats{sample_idx}] = rollout_single_sample( ...
			model_collection, x_init, constraint_cfg, ctx, ...
			beta0_values(sample_idx), sample_idx);
		if mod(sample_idx, 10) == 0 || sample_idx == n_samples
			batch_elapsed = toc(batch_timer);
			total_elapsed = toc(rollout_timer);
			fprintf(['  RK4 rolled out %d / %d samples ', ...
				'(batch %.1fs, total %.1fs)...\n'], ...
				sample_idx, n_samples, batch_elapsed, total_elapsed);
			batch_timer = tic;
		end
	end
end

%% Collect Per-Sample Results
for sample_idx = 1:n_samples
	path(:, sample_idx, :) = reshape(sample_paths{sample_idx}, ...
		n_steps + 1, 1, state_dim);
end
diagnostics.max_cumulative_variance = max(sample_max_cumulative);
diagnostics.hocbf = merge_rollout_diagnostics(sample_diags);
diagnostics.rollout_elapsed_seconds = toc(rollout_timer);
% 只有在开启约束，并且配置里要求输出诊断信息时，才打印 HOCBF 的诊断结果
if collect_diagnostics
	print_hocbf_diagnostics(diagnostics.hocbf, ...
		diagnostics.rollout_elapsed_seconds, ...
		struct_field_default(constraint_cfg, ...
		'closed_form_solver_enabled', false));
end
diagnostics.loggp_call_stats = merge_loggp_call_stats(sample_loggp_stats);
end

function parfor_opts = single_trajectory_parfor_options(worker_count)
% 固定 SubrangeSize=1，保证每个 worker 一次只领一条轨迹（默认的自动分块会
% 把多条轨迹打包给同一个 worker，轨迹之间耗时差异大时负载会不均）。
parfor_opts = worker_count;
try
	parfor_opts = parforOptions(gcp(), ...
		'RangePartitionMethod', 'fixed', 'SubrangeSize', 1);
catch
	% 老版本没有 parforOptions，退回按 worker 数限制的默认分块。
end
end

% 单条轨迹的完整 RK4 积分。所有 sample 共享的量通过 ctx 传入，per-sample 的
% 约束改动只作用在本地副本 sample_cfg 上，因此串行/并行结果一致。
function [sample_path, sample_diag, max_cumulative_variance, sample_stats] = ...
	rollout_single_sample(model_collection, x_init, constraint_cfg, ctx, ...
	beta0, sample_idx)

loggp_call_stats('reset'); % 本条轨迹的 LoG-GP 统计（并行时每个 worker 独立）
times = ctx.times;
dt_vec = ctx.dt_vec;
n_steps = ctx.n_steps;
state_dim = ctx.state_dim;
constrained = ctx.constrained;
collect_diagnostics = ctx.collect_diagnostics;
collect_control_trace = ctx.collect_control_trace;

sample_path = zeros(n_steps + 1, state_dim);
sample_path(1, :) = x_init(sample_idx, :); % 保存初始状态
if collect_diagnostics
	trace_capacity = 4 * n_steps;
	sample_diag = init_hocbf_diagnostics(trace_capacity, state_dim);
elseif collect_control_trace
	trace_capacity = 4 * n_steps;
	sample_diag = init_control_trace_diagnostics(trace_capacity, state_dim);
else
	sample_diag = init_hocbf_diagnostics();
end
max_cumulative_variance = 0.0;

sample_cfg = constraint_cfg;
cumulative_variance_now = 0.0; % 这条 sample 到当前时间为止累计的总方差积分
if constrained
	% Endpoint replacement needs the per-sample target even without PTCLF.
	if isfield(sample_cfg, 'anchor_clf_targets')
		sample_cfg.anchor_clf_target = ...
			sample_cfg.anchor_clf_targets(sample_idx, :)';
	end
	if struct_field_default(sample_cfg, 'ptclf_enabled', false) && ...
			isfield(sample_cfg, 'anchor_clf_target')
		% 包络式 PTCLF 需要按 sample 的初值 gbar0；SafeFlow 形式没有
		% 包络(收敛只靠 phi1 的发散性，与初值无关)，只打印 V0 供诊断。
		clf_is_envelope = ~strcmpi(struct_field_default(sample_cfg, ...
			'anchor_clf_form', 'envelope'), 'safeflow');
		clf_margin = struct_field_default(sample_cfg, ...
			'anchor_clf_ptzf_initial_margin', 1e-6);
		anchor_target = sample_cfg.anchor_clf_target(:);
		if isfield(sample_cfg, 'anchor_clf_matrix')
			anchor_error0 = sample_cfg.anchor_clf_matrix * ...
				x_init(sample_idx, :)' + ...
				sample_cfg.anchor_clf_offset(:) - anchor_target;
		else
			anchor_indices = sample_cfg.anchor_clf_indices(:)';
			anchor_error0 = x_init(sample_idx, anchor_indices)' - anchor_target;
		end
		clf_v0 = sum(anchor_error0 .^ 2);
		if ~clf_is_envelope
			fprintf('  sample %d: anchor CLF V0=%.6g\n', sample_idx, clf_v0);
		else
			sample_cfg.anchor_clf_ptzf_initial_bound = ...
				max(clf_v0, 0.0) + clf_margin;
			if struct_field_default(sample_cfg, ...
					'sequential_increment_qp_enabled', false) && ...
					isfield(sample_cfg, 'anchor_clf_matrix')
				if mod(numel(anchor_error0), 2) ~= 0
					error(['Sequential endpoint PTCLF requires equally ', ...
						'sized first/last endpoint errors.']);
				end
				endpoint_dim = numel(anchor_error0) / 2;
				first_error0 = anchor_error0(1:endpoint_dim);
				last_error0 = anchor_error0(endpoint_dim + (1:endpoint_dim));
				endpoint_margins = struct_field_default(sample_cfg, ...
					'anchor_clf_endpoint_ptzf_initial_margins', []);
				if isempty(endpoint_margins)
					endpoint_margins = 0.5 * clf_margin * ones(2, 1);
				else
					endpoint_margins = endpoint_margins(:);
					if numel(endpoint_margins) ~= 2
						error(['anchor_clf_endpoint_ptzf_initial_margins ', ...
							'must have two entries.']);
					end
				end
				endpoint_margin_ratios = struct_field_default( ...
					sample_cfg, ...
					'anchor_clf_endpoint_ptzf_initial_margin_ratios', []);
				if isempty(endpoint_margin_ratios)
					endpoint_margin_ratios = zeros(2, 1);
				else
					endpoint_margin_ratios = endpoint_margin_ratios(:);
					if numel(endpoint_margin_ratios) ~= 2 || ...
							any(endpoint_margin_ratios < 0)
						error(['anchor_clf_endpoint_ptzf_initial_margin', ...
							'_ratios must contain two nonnegative entries.']);
					end
				end
				sample_cfg.anchor_clf_endpoint_ptzf_initial_bounds = [ ...
					(1.0 + endpoint_margin_ratios(1)) * ...
						sum(first_error0 .^ 2) + endpoint_margins(1); ...
					(1.0 + endpoint_margin_ratios(2)) * ...
						sum(last_error0 .^ 2) + endpoint_margins(2)];
			end
			fprintf(['  sample %d: anchor CLF g0=%.6g, margin=%.3g, ', ...
				'gbar0=%.6g\n'], sample_idx, clf_v0, clf_margin, ...
				sample_cfg.anchor_clf_ptzf_initial_bound);
		end
	end
	% 根据初始方差动态计算 alpha1，保证 psi_1(0) >= psi1_margin
	h0 = sample_cfg.hocbf_relaxation_bound;
	B = sample_cfg.integral_uncertainty_budget;
	alpha1_computed = (beta0 - B) / h0 + sample_cfg.psi1_margin / h0;
	if struct_field_default(sample_cfg, 'ptcbf_enabled', false)
		terminal_margin = struct_field_default(sample_cfg, ...
			'terminal_variance_ptzf_initial_margin', 1e-6);
		terminal_h0 = ctx.terminal_h0_global;
		sample_cfg.terminal_variance_ptzf_hbar0 = terminal_h0;
		fprintf(['  sample %d: beta0=%.3f, B=%.3f, ', ...
			'alpha1=%.3f, terminal global hbar0=%.3f, terminal margin=%.3g\n'], ...
			sample_idx, beta0, B, alpha1_computed, ...
			terminal_h0, terminal_margin);
	else
		fprintf('  sample %d: beta0=%.3f, B=%.3f, alpha1=%.3f\n', sample_idx, beta0, B, alpha1_computed);
	end
	sample_cfg.hocbf_alpha1 = alpha1_computed;
end
% 端点 snap-and-hold（导师方案）: 到倒数 snap_flow_steps 步时把该段首尾点
% 的位置+斜率钉成上层生成点(anchor_clf_target)，此后这些维恒定不变。这几步
% 关掉 PTCLF（端点已被直接覆盖，不再需要 CLF 拉），但 HOCBF/PTCBF 仍作为
% 软约束保留（snap 区在 t≈0.99 > 切换时刻，二者本就带 slack），段内其余点
% 在方差软约束下继续 flow matching 贴合固定端点，抹平段边界折角。
% 仅对 index 型 CLF（第二层）生效；第三层是增量/矩阵型，端点非下标子集。
snap_flow_steps = struct_field_default(sample_cfg, ...
	'anchor_snap_flow_steps', 0);
% Snap is an independent endpoint-replacement experiment.  It must also
% work when PTCLF is disabled, so only the target/map availability gates it.
do_anchor_snap = constrained && snap_flow_steps > 0 && ...
	isfield(sample_cfg, 'anchor_clf_indices') && ...
	isfield(sample_cfg, 'anchor_clf_target');
snap_path_index = n_steps + 1 - snap_flow_steps;
snap_indices = [];
snap_target = [];
snap_constraint_cfg = sample_cfg;
if do_anchor_snap
	snap_indices = sample_cfg.anchor_clf_indices(:)';
	snap_target = sample_cfg.anchor_clf_target(:)';
	% 对照实验：只固定位置(x,y)，斜率(dx/ds,dy/ds)不覆盖，留给
	% flow/PTCBF/HOCBF 自己决定。anchor_clf_indices 按每个 anchor 点
	% [x, y, dx/ds, dy/ds] 4 维一组排列（见 build_anchor_clf_targets.m），
	% 因此每组的前 2 维是位置。
	position_only = struct_field_default(sample_cfg, ...
		'anchor_snap_position_only', false);
	if position_only
		anchor_feature_dim = 4;
		keep_mask = mod(0:(numel(snap_indices) - 1), anchor_feature_dim) < 2;
		snap_indices = snap_indices(keep_mask);
		snap_target = snap_target(keep_mask);
	end
	snap_target = reshape(snap_target, 1, []);
	% snap 区专用约束：关掉 PTCLF，保留 HOCBF/PTCBF（软）。
	snap_constraint_cfg = sample_cfg;
	snap_constraint_cfg.ptclf_enabled = false;
end
% 矩阵型 CLF（第三层增量表示）的 snap-and-hold：端点物理量是状态 z 的
% 线性组合 y=M*z+offset，不是某几个下标，不能直接赋值。改用最小范数
% 投影：把 z 沿着"改动量最小"的方向调整，使 M*z_new+offset 精确等于
% target，即解 M*delta=target-(M*z+offset) 的最小范数解
% delta=M'*(M*M')^{-1}*(...)，z_new=z+delta。段内其余(零空间)自由度
% 不受影响，继续由 flow/HOCBF/PTCBF 决定。
do_anchor_snap_matrix = constrained && snap_flow_steps > 0 && ...
	isfield(sample_cfg, 'anchor_clf_matrix') && ...
	isfield(sample_cfg, 'anchor_clf_offset') && ...
	(isfield(sample_cfg, 'anchor_clf_target') || ...
	 isfield(sample_cfg, 'anchor_clf_targets'));
snap_matrix = [];
snap_offset = [];
snap_target_mask = [];
snap_projector = [];
snap_constraint_cfg_matrix = sample_cfg;
if do_anchor_snap_matrix
	snap_matrix = sample_cfg.anchor_clf_matrix;
	snap_offset = sample_cfg.anchor_clf_offset(:);
	snap_target_mask = true(size(snap_offset));
	position_only = struct_field_default(sample_cfg, ...
		'anchor_snap_position_only', false);
	if position_only
		% 端点向量按 [首点(F); 末点(F)] 排列，每组前 2 维是位置
		% （见 build_increment_endpoint_clf_targets.m）。
		endpoint_dim = numel(snap_offset) / 2;
		pos_mask = false(numel(snap_offset), 1);
		pos_mask([1, 2, endpoint_dim + 1, endpoint_dim + 2]) = true;
		snap_matrix = snap_matrix(pos_mask, :);
		snap_offset = snap_offset(pos_mask);
		snap_target_mask = pos_mask;
	end
	% 投影算子只依赖 M（同一层所有 segment 共享），预先算好复用。
	snap_projector = snap_matrix' / (snap_matrix * snap_matrix');
	snap_constraint_cfg_matrix = sample_cfg;
	snap_constraint_cfg_matrix.ptclf_enabled = false;
	% The sequential tail QP fixes P1 through u1, solves the internal
	% obstacle rows through u2/u3/u4, then uses u5 to keep P5 fixed.
	snap_constraint_cfg_matrix.endpoint_hold_velocity_matrix = snap_matrix;
end
for step_idx = 1:n_steps % 对当前 sample 的每个时间步进行积分
	dt = dt_vec(step_idx); % 当前步的步长（非均匀网格时逐步不同）
	t_now = times(step_idx); % 当前时间
	x_now = reshape(sample_path(step_idx, :), [], 1); % 当前状态
	% Enter the snap region on the endpoint manifold before evaluating
	% flow matching. All following RK stages preserve M*x through the
	% hard velocity equality M*(mu+u)=0.
	if do_anchor_snap_matrix && step_idx >= snap_path_index
		snap_matrix_target = sample_cfg.anchor_clf_target(snap_target_mask);
		y_now = snap_matrix * x_now + snap_offset;
		x_now = x_now + snap_projector * (snap_matrix_target - y_now);
		sample_path(step_idx, :) = reshape(x_now, 1, []);
	end
	% snap 区（最后 snap_flow_steps 步）用无 PTCLF、仅 HOCBF/PTCBF 软约束。
	if do_anchor_snap && step_idx >= snap_path_index
		active_constraint_cfg = snap_constraint_cfg;
	elseif do_anchor_snap_matrix && step_idx >= snap_path_index
		active_constraint_cfg = snap_constraint_cfg_matrix;
	else
		active_constraint_cfg = sample_cfg;
	end
	%% k1
	if constrained
		[k1, hocbf_info] = constrained_velocity_field(model_collection, ...
			t_now, x_now, active_constraint_cfg, cumulative_variance_now, []);
		q1 = hocbf_info.sigma2; % 当前 RK4 子步 k1 位置的瞬时方差值
		if collect_diagnostics
			sample_diag = update_hocbf_diagnostics( ...
				sample_diag, hocbf_info, sample_idx, step_idx, 'k1');
		elseif collect_control_trace
			sample_diag = update_control_trace_diagnostics( ...
				sample_diag, hocbf_info, sample_idx, step_idx, 'k1');
		end
	else
		k1 = constrained_velocity_field(model_collection, t_now, x_now, [], 0.0, []);
		q1 = 0.0;
	end
	%% k2
	x_k2 = x_now + 0.5 * dt * k1;
	t_k2 = t_now + 0.5 * dt;
	if constrained
		[k2, hocbf_info] = constrained_velocity_field(model_collection, ...
			t_k2, x_k2, active_constraint_cfg, cumulative_variance_now + 0.5 * dt * q1, []);
		q2 = hocbf_info.sigma2;
		if collect_diagnostics
			sample_diag = update_hocbf_diagnostics( ...
				sample_diag, hocbf_info, sample_idx, step_idx, 'k2');
		elseif collect_control_trace
			sample_diag = update_control_trace_diagnostics( ...
				sample_diag, hocbf_info, sample_idx, step_idx, 'k2');
		end
	else
		k2 = constrained_velocity_field(model_collection, t_k2, x_k2, [], 0.0, []);
		q2 = 0.0;
	end
	%% k3
	x_k3 = x_now + 0.5 * dt * k2;
	t_k3 = t_now + 0.5 * dt;
	if constrained
		[k3, hocbf_info] = constrained_velocity_field(model_collection, ...
			t_k3, x_k3, active_constraint_cfg, cumulative_variance_now + 0.5 * dt * q2, []);
		q3 = hocbf_info.sigma2;
		if collect_diagnostics
			sample_diag = update_hocbf_diagnostics( ...
				sample_diag, hocbf_info, sample_idx, step_idx, 'k3');
		elseif collect_control_trace
			sample_diag = update_control_trace_diagnostics( ...
				sample_diag, hocbf_info, sample_idx, step_idx, 'k3');
		end
	else
		k3 = constrained_velocity_field(model_collection, t_k3, x_k3, [], 0.0, []);
		q3 = 0.0;
	end
	%% k4
	x_k4 = x_now + dt * k3;
	t_k4 = t_now + dt;
	if constrained
		[k4, hocbf_info] = constrained_velocity_field(model_collection, t_k4, ...
			x_k4, active_constraint_cfg, cumulative_variance_now + dt * q3, []);
		q4 = hocbf_info.sigma2;
		if collect_diagnostics
			sample_diag = update_hocbf_diagnostics( ...
				sample_diag, hocbf_info, sample_idx, step_idx, 'k4');
		elseif collect_control_trace
			sample_diag = update_control_trace_diagnostics( ...
				sample_diag, hocbf_info, sample_idx, step_idx, 'k4');
		end
	else
		k4 = constrained_velocity_field(model_collection, t_k4, x_k4, [], 0.0, []);
		q4 = 0.0;
	end

	%% RK4 update
	x_now = x_now + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
	if constrained
		cumulative_variance_now = cumulative_variance_now + ...
			(dt / 6.0) * (q1 + 2.0 * q2 + 2.0 * q3 + q4); % RK4数值积分的增量
	end
	max_cumulative_variance = max(max_cumulative_variance, ...
		cumulative_variance_now);
	sample_path(step_idx + 1, :) = reshape(x_now, 1, []);
	% snap-and-hold: 到达 snap_path_index 起把端点维钉成上层生成点，
	% 并在其后每一步重新钉住（保持固定），段内其余维继续 flow。
	if do_anchor_snap && (step_idx + 1) >= snap_path_index
		sample_path(step_idx + 1, snap_indices) = snap_target;
	end
	% 矩阵型 CLF 的 snap-and-hold: 最小范数投影，把 M*z+offset 精确
	% 拉回 target，零空间方向（段内其余自由度）不受影响。
	% Matrix snap is not repeated after RK4. The final operation is
	% constrained flow matching; its hard endpoint velocity equality keeps
	% P1/P5 fixed while obstacle PTCBF updates P2/P3/P4.
end
if collect_diagnostics
	sample_diag = finalize_hocbf_diagnostics(sample_diag);
elseif collect_control_trace
	sample_diag = finalize_control_trace_diagnostics(sample_diag);
end
sample_stats = loggp_call_stats('collect');
end
