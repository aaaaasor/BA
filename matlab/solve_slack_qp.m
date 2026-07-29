function [u_col, exitflag, qp_slack, qp_residuals, qp_iterations, ...
	qp_solve_seconds, qp_u_row_contributions, ...
	qp_u_equality_contribution] = solve_slack_qp(A_qp, ...
	b_qp, constraint_types, constraint_cfg, t, stats, terminal_info, ...
	integral_residual_without_u, terminal_residual_without_u)
if ~(exist('quadprog', 'file') == 2 || exist('quadprog', 'builtin') == 5)
	error('Constrained rollout QP requires quadprog from Optimization Toolbox.');
end
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

options = optimoptions('quadprog', 'Display', 'off', ...
	'MaxIterations', 2000, 'ConstraintTolerance', 1e-6, ...
	'OptimalityTolerance', 1e-6, 'StepTolerance', 1e-8);
qp_timer = tic;
[z_col, qp_fval, exitflag, qp_output, qp_lambda] = quadprog( ...
	H, f, A_solve, b_qp, Aeq_solve, beq, lb, [], [], options);
qp_solve_seconds = toc(qp_timer);
qp_iterations = qp_output.iterations;
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
