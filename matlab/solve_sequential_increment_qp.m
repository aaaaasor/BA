function result = solve_sequential_increment_qp(grad_x, integral_bound, ...
	terminal_bound, endpoint_infos, hocbf_enabled, obstacle_info, ...
	grad_x_active, constraint_cfg, t, stats, terminal_info, ...
	integral_residual_without_u, terminal_residual_without_u)
% 第三层增量控制的两级级联：先用 P1/P5 PTCLF 求完整参考控制，再以该
% reference 为中心求内部点 obstacle PTCBF 的最小安全修正。局部级联完成
% 后，再用全维最小范数修正处理全局 HOCBF/terminal 行。

n_u = numel(stats.mu);
block_dim = struct_field_required(constraint_cfg, ...
	'increment_control_block_dim');
n_blocks = struct_field_required(constraint_cfg, ...
	'increment_control_block_count');
if block_dim * n_blocks ~= n_u
	error(['Sequential increment QP block shape %d x %d does not match ', ...
		'control dimension %d.'], block_dim, n_blocks, n_u);
end

A_all = zeros(0, n_u);
b_all = zeros(0, 1);
types_all = strings(0, 1);
owner_all = zeros(0, 1); % 0: global, m: solved with block m.

if hocbf_enabled && grad_x_active
	A_all(end + 1, :) = grad_x;
	b_all(end + 1, 1) = integral_bound;
	types_all(end + 1, 1) = "integral";
	owner_all(end + 1, 1) = 0;
end
if isfinite(terminal_bound) && grad_x_active
	A_all(end + 1, :) = grad_x;
	b_all(end + 1, 1) = terminal_bound;
	types_all(end + 1, 1) = "terminal";
	owner_all(end + 1, 1) = 0;
end
if ~isempty(endpoint_infos)
	first_info = endpoint_infos(1);
	if first_info.enabled && first_info.row_active
		A_all(end + 1, :) = first_info.grad;
		b_all(end + 1, 1) = first_info.bound;
		types_all(end + 1, 1) = "anchor_clf_first";
		owner_all(end + 1, 1) = 1;
	end
	last_info = endpoint_infos(2);
	if last_info.enabled && last_info.row_active
		A_all(end + 1, :) = last_info.grad;
		b_all(end + 1, 1) = last_info.bound;
		types_all(end + 1, 1) = "anchor_clf_last";
		owner_all(end + 1, 1) = n_blocks;
	end
end
if obstacle_info.enabled && ~isempty(obstacle_info.A)
	n_obstacle_rows = size(obstacle_info.A, 1);
	if ~isfield(obstacle_info, 'row_point_indices') || ...
			numel(obstacle_info.row_point_indices) ~= n_obstacle_rows
		error('Sequential increment QP requires one point index per obstacle row.');
	end
	A_all = [A_all; obstacle_info.A];
	b_all = [b_all; obstacle_info.bounds];
	types_all(end + (1:n_obstacle_rows), 1) = "obstacle";
	owner_all(end + (1:n_obstacle_rows), 1) = ...
		obstacle_info.row_point_indices(:);
end

ptclf_reference_enabled = struct_field_default(constraint_cfg, ...
	'sequential_ptclf_reference_enabled', false) && ~isempty(endpoint_infos);
ptclf_reference_rows = find(startsWith(types_all, "anchor_clf"));
if ptclf_reference_enabled && isempty(ptclf_reference_rows)
	ptclf_reference_enabled = false;
end

u = zeros(n_u, 1);
endpoint_hold_matrix = struct_field_default(constraint_cfg, ...
	'endpoint_hold_velocity_matrix', []);
endpoint_hold_active = ~isempty(endpoint_hold_matrix);
endpoint_hold_contribution = zeros(n_u, 1);
local_constraint_cfg = constraint_cfg;
if isfield(local_constraint_cfg, 'endpoint_hold_velocity_matrix')
	local_constraint_cfg = rmfield(local_constraint_cfg, ...
		'endpoint_hold_velocity_matrix');
end
if endpoint_hold_active
	if size(endpoint_hold_matrix, 2) ~= n_u
		error(['Endpoint hold matrix has %d columns but the control ', ...
			'dimension is %d.'], size(endpoint_hold_matrix, 2), n_u);
	end
	% P1 is represented directly by the first increment block. Cancel its
	% nominal velocity before solving the internal point controls.
	first_cols = 1:block_dim;
	u(first_cols) = -reshape(stats.mu(first_cols), [], 1);
	endpoint_hold_contribution(first_cols) = u(first_cols);
end
slack_all = zeros(size(b_all));
row_solved = false(size(b_all));
row_contributions = zeros(size(A_all));
effective_bounds = nan(size(b_all));
exitflag = 1;
iterations = 0;
solve_seconds = 0.0;
grad_tol = struct_field_default(constraint_cfg, 'grad_tol', 1e-6);

global_rows = find(owner_all == 0);

% 第一级 QP：先只用 P1/P5 的 PTCLF 行求参考控制 u_ptclf_reference。
% 这里一次求解完整增量控制 [u1;u2;u3;u4;u5]，目标是相对 GP 原始
% flow 的最小范数修正。由于 P5 是各增量的累计结果，其 PTCLF 梯度会
% 同时作用于多个控制块，因此 P5 修正会按照累计 Jacobian 自动分配，
% 而不是只交给最后一个控制 u5。
u_ptclf_reference = u;
if ptclf_reference_enabled
	stats_ptclf = stats;
	% ptclf_reference_rows 只包含 anchor_clf_first/anchor_clf_last，
	% 此次求解不包含障碍 PTCBF 行。P1/P5 两行一次性用完整增量控制求解。
	%
	% 首块加权(导师方案): 不拆成两级，而是在目标函数里给 block1(S0) 加权，
	% 让 QP 倾向于不动它。原因是目标 ||u||^2 在归一化空间、约束 A*u 在物理
	% 空间，而 S0 是绝对量(sigma=0.197)、各增量是差分量(sigma=0.0049)，
	% 换算差 40 倍 -> 最小范数按平方分配使 S0 性价比高 1600 倍，实测 P5 的
	% 修正 91%~100% 都压在 S0 上(表现为"整段平移"而非"调整段内形状"，且与
	% 按住 S0 的 P1 行对消)。w1=40 正好把这个差异补回代价侧。
	ptclf_control_weight = ones(n_u, 1);
	first_block_weight = struct_field_default(constraint_cfg, ...
		'first_block_control_weight', 1.0);
	ptclf_control_weight(1:block_dim) = first_block_weight;
	ptclf_constraint_cfg = local_constraint_cfg;
	ptclf_constraint_cfg.control_weight = ptclf_control_weight;
	ptclf_slack_enabled = slack_enabled_for_constraints( ...
		ptclf_constraint_cfg, types_all(ptclf_reference_rows), t);
	if any(ptclf_slack_enabled)
		[u_ptclf_reference, exitflag_ptclf, slack_ptclf, ~, ...
			iterations_ptclf, seconds_ptclf, contributions_ptclf] = ...
			solve_slack_qp(A_all(ptclf_reference_rows, :), ...
			b_all(ptclf_reference_rows), types_all(ptclf_reference_rows), ...
			ptclf_constraint_cfg, t, stats_ptclf, terminal_info, ...
			integral_residual_without_u, terminal_residual_without_u);
	else
		% The endpoint PTCLF has only two hard halfspaces. Solve its weighted
		% minimum-norm problem in closed form instead of exposing W^2 (1600
		% on block 1 for W=40) to quadprog.
		[u_ptclf_reference, exitflag_ptclf, slack_ptclf, ~, ...
			iterations_ptclf, seconds_ptclf, contributions_ptclf] = ...
			solve_weighted_min_norm_halfspaces( ...
			A_all(ptclf_reference_rows, :), ...
			b_all(ptclf_reference_rows), ptclf_control_weight);
	end
	u = u_ptclf_reference;
	slack_all(ptclf_reference_rows) = slack_ptclf;
	row_solved(ptclf_reference_rows) = true;
	effective_bounds(ptclf_reference_rows) = b_all(ptclf_reference_rows);
	row_contributions(ptclf_reference_rows, :) = contributions_ptclf;
	exitflag = min(exitflag, exitflag_ptclf);
	iterations = iterations + iterations_ptclf;
	solve_seconds = solve_seconds + seconds_ptclf;
end

% 保留每一级实际输出，诊断时直接比较三级控制，而不是用 KKT 贡献近似归因。
% 第二级 QP：将上一步的 u_ptclf_reference 作为 reference，只求满足
% 内部点障碍 PTCBF 所需的最小附加修正 Delta u。此处不再加入 P1/P5
% 的 PTCLF 行，因此安全修正可以改变 P1/P5 的实际收敛速度。
obstacle_rows = find(types_all == "obstacle");
if ~isempty(obstacle_rows)
	% Let the obstacle PTCBF minimally modify the complete PTCLF reference.
	% P2/P3/P4 depend on the shared S0 block, so freezing u1 forces the
	% low-leverage increment blocks to generate unnecessarily large controls.
	% In the final snap-and-hold region P1 is a hard velocity equality, so u1
	% must remain fixed there; u5 can then restore the remaining P5 velocity.
	if endpoint_hold_active
		filter_cols = block_dim + (1:((n_blocks - 1) * block_dim));
	else
		filter_cols = 1:n_u;
	end
	A_filter_all = A_all(obstacle_rows, filter_cols);
	b_filter_all = b_all(obstacle_rows) - ...
		A_all(obstacle_rows, :) * u_ptclf_reference;
	active_filter = sqrt(sum(A_filter_all .^ 2, 2)) >= grad_tol;
	% 梯度退化的障碍行没有一阶控制能力，视为已检查并跳过。
	row_solved(obstacle_rows(~active_filter)) = true;
	filter_rows = obstacle_rows(active_filter);
	A_filter = A_filter_all(active_filter, :);
	b_filter = b_filter_all(active_filter);
	if ~isempty(filter_rows)
		stats_filter = stats;
		stats_filter.mu = reshape(stats.mu(filter_cols), 1, []) + ...
			reshape(u_ptclf_reference(filter_cols), 1, []);
		% 首块加权(避障级)。与 PTCLF 级同样的道理: increment 表示下
		% point m 的位置是前 m 个 block 的累加，所以避障行在 block1 上的
		% 梯度是其它块的约 45 倍(实测 b1≈24~25 vs b2~b4≈0.53~0.57)。
		% 最小范数按 a^2/w^2 分配工作量，w=1 时 block1 的性价比是其它块的
		% 45^2≈2025 倍 -> 修正几乎全压在 u1 上，表现为"整段平移"。
		% 取 w≈45 可使各块工作量大致均衡；默认 1.0 保持原行为。
		% 注意: 这里的控制向量只覆盖 filter_cols，权重向量必须同长度；
		% endpoint_hold 激活时 filter_cols 本就不含 block1，无需加权。
		obstacle_constraint_cfg = local_constraint_cfg;
		obstacle_first_block_weight = struct_field_default(constraint_cfg, ...
			'obstacle_first_block_control_weight', 1.0);
		if obstacle_first_block_weight ~= 1.0 && ~endpoint_hold_active
			obstacle_control_weight = ones(numel(filter_cols), 1);
			obstacle_control_weight(1:block_dim) = obstacle_first_block_weight;
			obstacle_constraint_cfg.control_weight = obstacle_control_weight;
		end
		[u_filter, exitflag_filter, slack_filter, ~, ...
			iterations_filter, seconds_filter, contributions_filter] = ...
			solve_slack_qp(A_filter, b_filter, types_all(filter_rows), ...
			obstacle_constraint_cfg, t, stats_filter, terminal_info, ...
			integral_residual_without_u, terminal_residual_without_u);
		if numel(u_filter) ~= numel(filter_cols)
			error(['Internal PTCBF filter returned %d controls for %d ', ...
				'internal columns (filter size %s).'], numel(u_filter), ...
				numel(filter_cols), mat2str(size(u_filter)));
		end
		u(filter_cols) = reshape(u_ptclf_reference(filter_cols), [], 1) + ...
			u_filter(:);
		slack_all(filter_rows) = slack_filter;
		row_solved(filter_rows) = true;
		effective_bounds(filter_rows) = b_filter;
		row_contributions(filter_rows, filter_cols) = contributions_filter;
		exitflag = min(exitflag, exitflag_filter);
		iterations = iterations + iterations_filter;
		solve_seconds = solve_seconds + seconds_filter;
	end
end
u_after_ptcbf = u;

% 不再保留旧的 u1->u2->...->u5 逐块回退。出现未求解的局部行说明
% 当前配置与“PTCLF reference -> obstacle PTCBF”架构不一致，直接报错。
local_rows = find(owner_all ~= 0);
unsolved_local_rows = local_rows(~row_solved(local_rows));
if ~isempty(unsolved_local_rows)
	error(['Unsolved local QP rows after PTCLF-reference/PTCBF cascade: %s. ', ...
		'Enable sequential_ptclf_reference_enabled or remove the unsupported ', ...
		'local constraint rows.'], mat2str(unsolved_local_rows(:)'));
end

if endpoint_hold_active
	% Internal controls are now fixed. The final increment affects P5 but
	% not P2/P3/P4, so use only u5 to cancel the remaining endpoint velocity.
	last_cols = (n_blocks - 1) * block_dim + (1:block_dim);
	hold_residual = -endpoint_hold_matrix * (stats.mu(:) + u);
	last_map = endpoint_hold_matrix(:, last_cols);
	u_last = last_map \ hold_residual;
	u(last_cols) = u(last_cols) + u_last;
	endpoint_hold_contribution(last_cols) = ...
		endpoint_hold_contribution(last_cols) + u_last;
	remaining_hold_residual = endpoint_hold_matrix * (stats.mu(:) + u);
	if norm(remaining_hold_residual, inf) > 1e-8
		error(['Sequential endpoint hold could not preserve P1/P5: ', ...
			'max velocity residual %.6g.'], ...
			norm(remaining_hold_residual, inf));
	end
end

% HOCBF is a post-filter: minimally correct the complete local control
% vector after all PTCLF/PTCBF blocks have been solved. By design, this
% filter enforces only the global variance rows; it may temporarily alter
% the local PTCLF/PTCBF conditions, which take over after the filter ends.
if ~isempty(global_rows)
	b_filter = b_all(global_rows) - A_all(global_rows, :) * u;
	effective_bounds(global_rows) = b_filter;
	stats_filter = stats;
	if endpoint_hold_active
		stats_filter.mu = reshape(stats.mu(:) + u, size(stats.mu));
	end
	[u_filter, exitflag_filter, slack_filter, ~, iterations_filter, ...
		seconds_filter, contributions_filter, ...
		equality_filter_contribution] = solve_slack_qp( ...
		A_all(global_rows, :), b_filter, types_all(global_rows), ...
		constraint_cfg, t, stats_filter, terminal_info, ...
		integral_residual_without_u, terminal_residual_without_u);
	u = u + u_filter;
	endpoint_hold_contribution = endpoint_hold_contribution + ...
		equality_filter_contribution;
	slack_all(global_rows) = slack_filter;
	row_solved(global_rows) = true;
	row_contributions(global_rows, :) = contributions_filter;
	exitflag = min(exitflag, exitflag_filter);
	iterations = iterations + iterations_filter;
	solve_seconds = solve_seconds + seconds_filter;
end

raw_residuals = A_all * u - b_all;
relaxed_residuals = raw_residuals - slack_all;
result.u = u;
result.u_ptclf_reference = u_ptclf_reference;
result.u_after_ptcbf = u_after_ptcbf;
result.residuals_at_ptclf_reference = A_all * u_ptclf_reference - b_all;
result.residuals_after_ptcbf = A_all * u_after_ptcbf - b_all;
result.residuals_after_hocbf = raw_residuals;
result.exitflag = exitflag;
result.slack = slack_all;
result.raw_residuals = raw_residuals;
result.relaxed_residuals = relaxed_residuals;
result.iterations = iterations;
result.solve_seconds = solve_seconds;
result.row_contributions = row_contributions;
result.constraint_types = types_all;
result.row_solved = row_solved;
result.bounds = b_all;
result.effective_bounds = effective_bounds;
result.active_constraint_count = sum(abs(relaxed_residuals) <= 1e-7);
result.equality_contribution = endpoint_hold_contribution;
end
