function [u_col, exitflag, qp_slack, qp_residuals, qp_iterations, ...
	qp_solve_seconds, qp_u_row_contributions, ...
	qp_u_equality_contribution] = solve_slack_qp(A_qp, ...
	b_qp, constraint_types, constraint_cfg, t, stats, terminal_info, ...
	integral_residual_without_u, terminal_residual_without_u)
n_u = numel(stats.mu);
n_constraints = size(A_qp, 1);
slack_enabled = slack_enabled_for_constraints(constraint_cfg, constraint_types, t);
slack_rows = find(slack_enabled);

% Keep snapped endpoints fixed while flow matching and obstacle constraints
% act on the remaining degrees of freedom. The applied velocity is mu + u,
% so M(mu + u)=0 becomes the hard equality M*u=-M*mu.
endpoint_hold_matrix = struct_field_default(constraint_cfg, ...
	'endpoint_hold_velocity_matrix', []);
if ~isempty(endpoint_hold_matrix)
	if size(endpoint_hold_matrix, 2) ~= n_u
		error(['Endpoint hold matrix has %d columns but the control ', ...
			'dimension is %d.'], size(endpoint_hold_matrix, 2), n_u);
	end
	Aeq_u = endpoint_hold_matrix;
	beq = -Aeq_u * stats.mu(:);
	for row_idx = 1:size(Aeq_u, 1)
		row_norm = norm(Aeq_u(row_idx, :));
		if row_norm > 0
			Aeq_u(row_idx, :) = Aeq_u(row_idx, :) / row_norm;
			beq(row_idx) = beq(row_idx) / row_norm;
		end
	end
else
	Aeq_u = zeros(0, n_u);
	beq = zeros(0, 1);
end

% 行归一化（把整行和右端同除以行范数，半平面不变）：对所有行统一归一化，
% 不分硬约束/带 slack 的约束。
%   - 硬行: 否则第三层 CLF 行系数含增量 std(量级极小)，quadprog 会把
%     "系数极小 + 右端很负"的可行硬约束误判为不可行(exitflag=-2)。
%   - slack 行: 惩罚 w*delta^2 里 delta 的单位随 ‖A_i‖ 缩放，统一归一化后
%     delta 变成一致的"单位梯度"违反量，不同约束类型(integral/terminal/
%     anchor_clf/obstacle)之间、以及同类型内部(如障碍各点)的 slack 权重
%     才真正可比。
% ⚠️ 这会改变 integral/terminal/anchor_clf 现有 slack 权重(hocbf_slack_
% weight 等)的有效尺度——这些权重是在"不归一化"下精调出来的，grad_beta
% 的行范数常见到 10~18(‖A‖^2 达 100~300 量级)，归一化后同样的数值权重
% 代价骤降两个数量级，会明显偏软。切到全归一化后必须重新标定这些权重
% (w_new ≈ w_old * median(‖A_row‖)^2，用 print_hocbf_diagnostics 里新增
% 的 "grad_x row norm" / "anchor CLF grad row norm" 中位数换算)，否则
% 会复现"QP 在尾段大量吃 slack、correction norm 冲到 300+"的问题。
% 下限保护(第二道防线): row_norm 小于 grad_tol 时不归一化，直接跳过。
% 上游(anchor_clf_info.m / apply_hocbf_integral.m 的 grad_x_active)已经
% 会把梯度退化的行整行剔除，正常不会走到这里；这里只是防止漏网(比如
% obstacle 之外未来新增的约束类型忘了做退化检测)时，除以接近 0 的数
% 把 b_qp 放大成数值垃圾(实测见过 1e50 量级)，导致假性不可行。
row_norm_tol = struct_field_default(constraint_cfg, 'grad_tol', 1e-6);
row_scale = ones(n_constraints, 1);
for row_idx = 1:n_constraints
	row_norm = norm(A_qp(row_idx, :));
	if row_norm > row_norm_tol
		row_scale(row_idx) = row_norm;
		A_qp(row_idx, :) = A_qp(row_idx, :) / row_norm;
		b_qp(row_idx) = b_qp(row_idx) / row_norm;
	end
end
slack_weights = slack_weights_for_constraints( ...
	constraint_cfg, constraint_types(slack_rows));

% slack 权重在原始约束单位中定义。归一化后有
% delta_raw = ||A_i|| * delta_normalized，因此目标函数中的等价权重为
% w_i * ||A_i||^2。若遗漏该因子，梯度接近零的约束会把很小的原始
% 违反放大成巨大的归一化 slack，并诱发不必要的超大控制修正。
slack_objective_weights = slack_weights(:) .* row_scale(slack_rows) .^ 2;

% 每条约束一个 slack:
% A_i*u - delta_i <= b_i, delta_i >= 0。
% slack 关闭的约束保持 hard: A_i*u <= b_i。
A_slack = zeros(n_constraints, numel(slack_rows));
for slack_idx = 1:numel(slack_rows)
	A_slack(slack_rows(slack_idx), slack_idx) = -1.0;
end
A_solve = [A_qp, A_slack];
Aeq_solve = [Aeq_u, zeros(size(Aeq_u, 1), numel(slack_rows))];

if any(~isfinite(A_solve(:))) || any(~isfinite(b_qp(:)))
	error(['Constrained QP has non-finite entries at t=%.6g: ', ...
		'any(isnan/isinf(A))=%d, any(isnan/isinf(b))=%d, ', ...
		'active constraints=%s.'], t, ...
		any(~isfinite(A_solve(:))), any(~isfinite(b_qp(:))), ...
		strjoin(cellstr(constraint_types(:)'), ','));
end

% QP 目标:
% min 0.5*||W*u||^2 + 0.5*sum_i w_i*delta_i^2。
% HOCBF/PTCBF/PTCLF 可以使用不同的 slack weight。
%
% control_weight (可选, 默认全 1) 让不同控制块在目标里代价不同。
% 动机: 目标函数在归一化空间衡量代价(||u||^2)，而约束在物理空间衡量
% 效果(A*u)，两者的换算系数 sigma 逐块不同。第三层里 S0 是绝对量
% (sigma=0.197)、各增量是差分量(sigma=0.0049)，所以同样的 ||u|| 通过 S0
% 能产生 40 倍的物理效果，最小范数按平方分配 -> S0 的性价比是 1600 倍，
% QP 会把 91%~100% 的力气都压在 S0 上(实测)。给 S0 加权 w1 正好把这个
% 差异补回代价侧，使 QP 不再偏向它。
control_weight = struct_field_default(constraint_cfg, ...
	'control_weight', ones(n_u, 1));
control_weight = reshape(control_weight, [], 1);
if numel(control_weight) ~= n_u
	error(['control_weight has %d entries but the control dimension ', ...
		'is %d.'], numel(control_weight), n_u);
end
if any(control_weight <= 0) || any(~isfinite(control_weight))
	error('control_weight must be finite and strictly positive.');
end
n_slack = numel(slack_rows);
H = diag([control_weight .^ 2; slack_objective_weights]);
f = zeros(n_u + n_slack, 1);
lb = [-inf(n_u, 1); zeros(n_slack, 1)];

% The objective is nonnegative with its unique unconstrained minimum at
% u=0, slack=0. If every normalized upper bound is nonnegative, that point
% is already feasible and therefore globally optimal; avoid launching
% quadprog for the many inactive stages in the sequential cascade.
zero_equality_feasible = isempty(beq) || norm(beq, inf) <= 1e-10;
if all(b_qp >= 0.0) && zero_equality_feasible
	u_col = zeros(n_u, 1);
	exitflag = 1;
	qp_slack = zeros(n_constraints, 1);
	qp_residuals = -b_qp;
	qp_iterations = 0;
	qp_solve_seconds = 0.0;
	qp_u_row_contributions = zeros(n_constraints, n_u);
	qp_u_equality_contribution = zeros(n_u, 1);
	return;
end

% 闭式 active-set 枚举（含 slack 行）。该后端按层开关：第一、二层保持
% quadprog，第三层才优先使用小行数闭式解。
% 在 z=[u; delta] 里整个 slack QP 仍然是一个加权最小范数问题:
%   min 0.5*||diag(control_weight, sqrt(w))*z||^2
%   s.t. [A_qp, A_slack]*z <= b_qp,  [0, -I]*z <= 0,  [Aeq_u, 0]*z = beq
% 所以硬约束用的那套闭式枚举可以原样解软问题，且是精确解。
%
% 必须优先于 quadprog 的原因: interior-point-convex 在存在平行行时会在
% MaxIterations 处停滞返回 exitflag=0——integral 与 terminal 两行共用
% 同一个 grad_beta，行归一化后完全相同(实测 cosine=1.0000,
% cond(A_qp)=6e15)。此时原始最优解其实唯一且良态(诊断里 fval 恰等于
% 0.5*||u||^2、constrviolation=0)，卡住的只是对偶最优性判据，因为重复行
% 的乘子不唯一，KKT 矩阵秩亏。闭式枚举用 pinv 解 Gram，天然取重复行乘子
% 的最小范数解，不受这种退化影响。
closed_form_max_rows = 12;
closed_form_rows = n_constraints + n_slack;
closed_form_enabled = struct_field_default(constraint_cfg, ...
	'closed_form_solver_enabled', false);
if closed_form_enabled && closed_form_rows <= closed_form_max_rows
	A_closed = [A_solve; [zeros(n_slack, n_u), -eye(n_slack)]];
	b_closed = [b_qp(:); zeros(n_slack, 1)];
	z_weight = [control_weight; sqrt(slack_objective_weights(:))];
	try
		[z_closed, exitflag, ~, closed_residuals, qp_iterations, ...
			qp_solve_seconds, closed_row_contributions, ...
			closed_equality_contribution] = ...
			solve_weighted_min_norm_halfspaces( ...
				A_closed, b_closed, z_weight, Aeq_solve, beq);
		u_col = z_closed(1:n_u);
		qp_slack = zeros(n_constraints, 1);
		% slack 属于归一化后的行，换回原始约束单位再交给诊断。
		qp_slack(slack_rows) = ...
			row_scale(slack_rows) .* z_closed(n_u + 1:end);
		% 末尾 n_slack 行是 delta>=0 的界，不属于外部约束，丢掉。
		qp_residuals = closed_residuals(1:n_constraints);
		qp_u_row_contributions = ...
			closed_row_contributions(1:n_constraints, 1:n_u);
		qp_u_equality_contribution = closed_equality_contribution(1:n_u);
		return;
	catch closed_err
		% 硬约束真不可行时才会走到这里。交给 quadprog 复现，让下面的
		% failure dump 打出完整结构，不要在这里把问题吞掉。
		warning('solve_slack_qp:closedFormFailed', ...
			['Closed-form active-set solve failed at t=%.6g (%s); ', ...
			'falling back to quadprog.'], t, closed_err.message);
	end
end

if ~(exist('quadprog', 'file') == 2 || exist('quadprog', 'builtin') == 5)
	if closed_form_enabled
		error(['Constrained rollout requires quadprog when more than %d ', ...
			'closed-form rows are active.'], closed_form_max_rows);
	else
		error('This rollout level is configured to use quadprog, but quadprog is unavailable.');
	end
end

persistent quadprog_options active_quadprog_options phase1_linprog_options
if isempty(quadprog_options)
	quadprog_options = optimoptions('quadprog', 'Display', 'off', ...
		'MaxIterations', 2000, 'ConstraintTolerance', 1e-6, ...
		'OptimalityTolerance', 1e-6, 'StepTolerance', 1e-8);
end
qp_timer = tic;
[z_col, qp_fval, exitflag, qp_output, qp_lambda] = quadprog( ...
	H, f, A_solve, b_qp, Aeq_solve, beq, lb, [], [], quadprog_options);
qp_solve_seconds = toc(qp_timer);
qp_iterations = qp_output.iterations;

% A corridor contributes exact opposite row pairs, while integral and
% terminal contribute duplicate rows.  Interior-point can incorrectly
% report infeasibility on this highly degenerate geometry even though all
% non-safety rows have slack.  If that happens, first find a point that
% satisfies only the genuinely hard rows, construct the soft slacks
% explicitly, and start active-set from this certified feasible point.
if exitflag <= 0 && ...
		(exist('linprog', 'file') == 2 || exist('linprog', 'builtin') == 5)
	hard_rows = find(~slack_enabled);
	try
		if isempty(phase1_linprog_options)
			phase1_linprog_options = optimoptions('linprog', 'Display', 'off');
		end
		phase_timer = tic;
		[u_feasible, ~, phase_exitflag] = linprog(zeros(n_u, 1), ...
			A_qp(hard_rows, :), b_qp(hard_rows), Aeq_u, beq, ...
			[], [], phase1_linprog_options);
		qp_solve_seconds = qp_solve_seconds + toc(phase_timer);
		if phase_exitflag > 0 && ~isempty(u_feasible) && ...
				all(isfinite(u_feasible))
			z_feasible = [u_feasible; zeros(n_slack, 1)];
			if n_slack > 0
				z_feasible(n_u + (1:n_slack)) = max( ...
					A_qp(slack_rows, :) * u_feasible - ...
					b_qp(slack_rows), 0.0) + 1e-10;
			end
			if isempty(active_quadprog_options)
				active_quadprog_options = optimoptions('quadprog', ...
					'Display', 'off', 'Algorithm', 'active-set', ...
					'MaxIterations', 1000, 'ConstraintTolerance', 1e-7, ...
					'OptimalityTolerance', 1e-7, 'StepTolerance', 1e-10);
			end
			active_timer = tic;
			[z_phase_active, fval_phase_active, ...
				exitflag_phase_active, output_phase_active, ...
				lambda_phase_active] = quadprog(H, f, A_solve, b_qp, ...
				Aeq_solve, beq, lb, [], z_feasible, ...
				active_quadprog_options);
			qp_solve_seconds = qp_solve_seconds + toc(active_timer);
			phase_active_feasible = ~isempty(z_phase_active) && ...
				all(isfinite(z_phase_active)) && ...
				max([A_solve * z_phase_active - b_qp; 0.0]) <= 1e-6 && ...
				max([lb - z_phase_active; 0.0]) <= 1e-9;
			if phase_active_feasible && ~isempty(Aeq_solve)
				phase_active_feasible = norm( ...
					Aeq_solve * z_phase_active - beq, inf) <= 1e-6;
			end
			if phase_active_feasible && exitflag_phase_active > 0
				z_col = z_phase_active;
				qp_fval = fval_phase_active;
				exitflag = exitflag_phase_active;
				qp_output = output_phase_active;
				qp_lambda = lambda_phase_active;
				qp_iterations = qp_iterations + ...
					output_phase_active.iterations;
			end
		end
	catch phase_err
		warning('solve_slack_qp:phaseOneRetryFailed', ...
			'Hard-row feasibility/active-set retry failed at t=%.6g: %s', ...
			t, phase_err.message);
	end
end

% Interior-point-convex can exhaust its iteration budget when two soft
% rows are parallel (the integral and terminal variance rows share the
% same grad_beta) while nearby PTCLF/obstacle rows are hard. In that case
% quadprog can already have a strictly feasible, small solution but still
% return exitflag=0 because the dual optimality test has stalled. Refine
% that feasible point with active-set, which is less sensitive to the
% duplicated half-space geometry.
if exitflag <= 0 && ~isempty(z_col) && all(isfinite(z_col))
	primary_z = z_col;
	primary_fval = qp_fval;
	primary_output = qp_output;
	primary_lambda = qp_lambda;
	primary_ineq_violation = max([A_solve * primary_z - b_qp; 0.0]);
	primary_eq_violation = 0.0;
	if ~isempty(Aeq_solve)
		primary_eq_violation = norm(Aeq_solve * primary_z - beq, inf);
	end
	primary_lb_violation = max([lb - primary_z; 0.0]);
	primary_feasible = primary_ineq_violation <= 1e-6 && ...
		primary_eq_violation <= 1e-6 && primary_lb_violation <= 1e-9;

	if primary_feasible
		try
			if isempty(active_quadprog_options)
				active_quadprog_options = optimoptions('quadprog', ...
					'Display', 'off', 'Algorithm', 'active-set', ...
					'MaxIterations', 500, 'ConstraintTolerance', 1e-7, ...
					'OptimalityTolerance', 1e-7, 'StepTolerance', 1e-10);
			end
			active_timer = tic;
			[z_active, fval_active, exitflag_active, output_active, ...
				lambda_active] = quadprog(H, f, A_solve, b_qp, ...
				Aeq_solve, beq, lb, [], primary_z, active_quadprog_options);
			qp_solve_seconds = qp_solve_seconds + toc(active_timer);
			active_feasible = ~isempty(z_active) && ...
				all(isfinite(z_active));
			if active_feasible
				active_feasible = max([A_solve * z_active - b_qp; 0.0]) ...
					<= 1e-6 && max([lb - z_active; 0.0]) <= 1e-9;
				if active_feasible && ~isempty(Aeq_solve)
					active_feasible = norm( ...
						Aeq_solve * z_active - beq, inf) <= 1e-6;
				end
			end
			if active_feasible && exitflag_active > 0
				z_col = z_active;
				qp_fval = fval_active;
				exitflag = exitflag_active;
				qp_output = output_active;
				qp_lambda = lambda_active;
				qp_iterations = qp_iterations + output_active.iterations;
			elseif active_feasible && fval_active <= primary_fval
				z_col = z_active;
				qp_fval = fval_active;
				qp_output = output_active;
				qp_lambda = lambda_active;
				qp_iterations = qp_iterations + output_active.iterations;
			end
		catch active_err
			warning('solve_slack_qp:activeSetRetryFailed', ...
				'Active-set retry failed at t=%.6g: %s', t, ...
				active_err.message);
		end

		% A feasible iterate with a modest first-order residual is safe to
		% apply. Do not abort a valid RK rollout only because the dual
		% stopping test reached MaxIterations.
		candidate_ineq_violation = max([A_solve * z_col - b_qp; 0.0]);
		candidate_eq_violation = 0.0;
		if ~isempty(Aeq_solve)
			candidate_eq_violation = norm(Aeq_solve * z_col - beq, inf);
		end
		candidate_lb_violation = max([lb - z_col; 0.0]);
		candidate_first_order = inf;
		if isfield(qp_output, 'firstorderopt')
			candidate_first_order = qp_output.firstorderopt;
		end
		if exitflag <= 0 && candidate_ineq_violation <= 1e-6 && ...
				candidate_eq_violation <= 1e-6 && ...
				candidate_lb_violation <= 1e-9 && ...
				candidate_first_order <= 1.0 && isfinite(qp_fval)
			exitflag = 1;
			warning('solve_slack_qp:acceptedFeasibleIterate', ...
				['Accepted feasible quadprog iterate at t=%.6g after the ', ...
				'optimality iteration limit (firstorderopt=%.3g).'], ...
				t, candidate_first_order);
		end

		% Restore the original feasible candidate if an unsuccessful retry
		% replaced it with anything worse.
		if exitflag <= 0 && (~all(isfinite(z_col)) || ...
				max([A_solve * z_col - b_qp; 0.0]) > 1e-6)
			z_col = primary_z;
			qp_fval = primary_fval;
			qp_output = primary_output;
			qp_lambda = primary_lambda;
		end
	end
end
% 失败回退：quadprog 偶尔在"最优解在可行域深内部、约束全松弛"的实例上
% 停滞(exitflag<=0)，返回一个可行但远非最优的垃圾解(见诊断:firstorderopt
% 巨大、solution norm 大)。若此时所有约束的 bound 都 >= 0，则 u=0(delta=0)
% 就同时可行且全局最优(目标=0，即这一步根本不需要任何控制修正)。这种情况
% 直接回退到 u=0，既躲开垃圾解(注入大 u 会制造发散)，又给出正确答案。
% 只有当某个约束 bound < 0(真需要非零 u)时才 dump+报错，不掩盖真问题。
qp_feas_tol = 1e-6;
if (exitflag <= 0 || isempty(z_col)) && ...
		all(b_qp >= -qp_feas_tol) && zero_equality_feasible
	z_col = zeros(n_u + n_slack, 1);
	qp_lambda.ineqlin = zeros(n_constraints, 1);
	exitflag = 1;   % u=0 可证明最优，标记为已解
	warning('solve_slack_qp:fallbackZeroU', ...
		['QP 在 t=%.4g 停滞(quadprog exitflag<=0)，但所有约束 bound>=0，', ...
		'u=0 可行且最优，回退到 u=0。'], t);
end
if exitflag <= 0 || isempty(z_col)
	row_norms = sqrt(sum(A_qp .^ 2, 2));
	% QP 失败时 dump 完整结构，直接看是不是行近似平行/条件数爆炸导致
	% quadprog 卡死(exitflag=0)，而不是继续靠 b_qp 数值猜测。
	fprintf('\n===== QP FAILURE DUMP at t=%.6g =====\n', t);
	fprintf('constraint_types: %s\n', ...
		strjoin(cellstr(constraint_types(:)'), ','));
	fprintf('A_qp (normalized rows, one row per constraint):\n');
	disp(A_qp);
	fprintf('b_qp (normalized): %s\n', mat2str(b_qp(:)', 5));
	fprintf('row norms (pre-normalization): %s\n', mat2str(row_scale(:)', 5));
	% 行两两夹角余弦：接近 ±1 说明两行近似平行(病态来源)。
	n_rows_dump = size(A_qp, 1);
	if n_rows_dump >= 2
		cos_mat = A_qp * A_qp';   % 归一化后行范数≈1，直接近似余弦
		fprintf('row pairwise dot (normalized ~ cosine):\n');
		disp(cos_mat);
	end
	fprintf('cond(A_qp)=%.4g, cond(A_solve)=%.4g\n', ...
		cond(A_qp), cond(A_solve));
	% Hessian 尺度诊断：u 部分对角=1，slack 部分=slack_objective_weights。
	% 若最小特征值接近 0(半正定)或条件数很大(u 与 slack 尺度悬殊)，内点法
	% 可能在退化/尺度失衡的目标上反复徘徊到迭代耗尽。
	H_sym = 0.5 * (H + H');
	H_eig = eig(H_sym);
	fprintf('diag(H): %s\n', mat2str(diag(H_sym)', 5));
	fprintf('H eig min/max: %.6g / %.6g,  cond(H): %.6g\n', ...
		min(H_eig), max(H_eig), cond(H_sym));
	fprintf('slack_objective_weights: %s\n', ...
		mat2str(slack_objective_weights(:)', 5));
	% 线性项 f：目标应是纯二次 0.5*u'u + 0.5*w*delta^2，f 必须全 0。
	% 若 f 非零，最优点位置会偏移，firstorderopt 大可能指向未预期的方向。
	fprintf('f (linear term): %s\n', mat2str(f(:)', 12));
	fprintf('norm(f): %.12g,  all finite f: %d\n', ...
		norm(f), all(isfinite(f)));
	% 控制上下界：确认 u 无界(lb=-inf, ub 空)，排除"anchor 行在限幅下不可达"。
	fprintf('lb: %s\n', mat2str(lb(:)', 5));
	fprintf('ub: (empty -> +inf on all vars)\n');
	% 逐行"在无界 u 下能达到的最小 A_i*u"：无界时对硬行恒为 -inf(必可达)。
	% 若将来加了 ub/lb 限幅，这里能直接看出哪行不可达。
	% 分清是哪一项 KKT 分量卡着：firstorderopt(一阶最优性度量)对应
	% OptimalityTolerance，constrviolation(约束违反)对应 ConstraintTolerance。
	% 情况一(firstorderopt 略大、constrviolation≈0): 只是最优性判据过严，
	%   放宽 OptimalityTolerance 有望解决。
	% 情况二(firstorderopt 很大, 如 0.1): 根本没接近 KKT 点(停滞/循环)，
	%   放宽容差无意义。
	% 情况三(constrviolation 明显 > ConstraintTolerance): 是可行性没满足，
	%   放宽 OptimalityTolerance 无意义，要查 A_solve/slack/变量排列。
	fprintf('algorithm: %s\n', qp_output.algorithm);
	if isfield(qp_output, 'firstorderopt')
		fprintf('firstorderopt: %.12g\n', qp_output.firstorderopt);
	end
	if isfield(qp_output, 'constrviolation')
		fprintf('constrviolation: %.12g\n', qp_output.constrviolation);
	end
	fprintf('final objective (fval): %.12g\n', qp_fval);
	if ~isempty(z_col) && all(isfinite(z_col))
		raw_violation = max(A_solve * z_col - b_qp);
		fprintf('independent max(Ax-b): %.12g\n', raw_violation);
		fprintf('solution norm: %.12g\n', norm(z_col));
	end
	fprintf('quadprog exitflag=%d, iterations=%d, message: %s\n', ...
		exitflag, qp_iterations, qp_output.message);
	fprintf('===== END QP FAILURE DUMP =====\n\n');
	error(['Constrained QP failed at t=%.6g: active constraints=%s, beta=%.6g, ', ...
		'beta_cap=%.6g, h_T=%.6g, integral residual=%.6g, ', ...
		'terminal residual=%.6g, grad norm=%.6g, exitflag=%d, iterations=%d, ', ...
		'row norms=%s, b_qp=%s, slack enabled=%s.'], ...
		t, strjoin(cellstr(constraint_types(:)'), ','), ...
		stats.sigma2, terminal_info.beta_cap, terminal_info.h, ...
		integral_residual_without_u, terminal_residual_without_u, ...
		norm(stats.sigma2_grad_x), exitflag, qp_iterations, ...
		mat2str(row_norms', 4), mat2str(b_qp(:)', 4), ...
		mat2str(double(slack_enabled(:)')));
end
u_col = z_col(1:n_u);
% KKT stationarity on the unconstrained u variables is
% W^2*u + A_qp'*lambda + Aeq_u'*nu = 0, hence u = -W^-2*(...). The inverse
% weight must be applied here, otherwise the per-row contributions no longer
% reconstruct u once control_weight differs from 1.
inv_weight_sq = 1 ./ (control_weight .^ 2);
qp_u_row_contributions = zeros(n_constraints, n_u);
if isstruct(qp_lambda) && isfield(qp_lambda, 'ineqlin') && ...
		numel(qp_lambda.ineqlin) == n_constraints
	qp_u_row_contributions = -bsxfun(@times, ...
		qp_lambda.ineqlin(:), bsxfun(@times, A_qp, inv_weight_sq'));
end
qp_u_equality_contribution = zeros(n_u, 1);
if isstruct(qp_lambda) && isfield(qp_lambda, 'eqlin') && ...
		numel(qp_lambda.eqlin) == size(Aeq_u, 1)
	qp_u_equality_contribution = ...
		-inv_weight_sq .* (Aeq_u' * qp_lambda.eqlin(:));
end
qp_slack = zeros(n_constraints, 1);
% QP slack belongs to the normalized row. Convert it back to the raw
% constraint unit so diagnostics can compare raw residual - raw slack.
qp_slack(slack_rows) = ...
	row_scale(slack_rows) .* z_col(n_u + 1:end);

% qp_residuals 是带 slack 后的残差:
% A_i*u - delta_i - b_i，应当 <= 0。
qp_residuals = A_solve * z_col - b_qp;
end
