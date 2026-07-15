function [u_col, exitflag, qp_slack, qp_residuals, qp_iterations, qp_solve_seconds] = solve_slack_qp(A_qp, ...
	b_qp, constraint_types, constraint_cfg, t, stats, terminal_info, ...
	integral_residual_without_u, terminal_residual_without_u)
if ~(exist('quadprog', 'file') == 2 || exist('quadprog', 'builtin') == 5)
	error('Constrained rollout QP requires quadprog from Optimization Toolbox.');
end
n_u = numel(stats.mu);
n_constraints = size(A_qp, 1);
slack_enabled = slack_enabled_for_constraints(constraint_cfg, constraint_types, t);
slack_rows = find(slack_enabled);

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
for row_idx = 1:n_constraints
	row_norm = norm(A_qp(row_idx, :));
	if row_norm > row_norm_tol
		A_qp(row_idx, :) = A_qp(row_idx, :) / row_norm;
		b_qp(row_idx) = b_qp(row_idx) / row_norm;
	end
end
slack_weights = slack_weights_for_constraints( ...
	constraint_cfg, constraint_types(slack_rows));

% 每条约束一个 slack:
% A_i*u - delta_i <= b_i, delta_i >= 0。
% slack 关闭的约束保持 hard: A_i*u <= b_i。
A_slack = zeros(n_constraints, numel(slack_rows));
for slack_idx = 1:numel(slack_rows)
	A_slack(slack_rows(slack_idx), slack_idx) = -1.0;
end
A_solve = [A_qp, A_slack];

if any(~isfinite(A_solve(:))) || any(~isfinite(b_qp(:)))
	error(['Constrained QP has non-finite entries at t=%.6g: ', ...
		'any(isnan/isinf(A))=%d, any(isnan/isinf(b))=%d, ', ...
		'active constraints=%s.'], t, ...
		any(~isfinite(A_solve(:))), any(~isfinite(b_qp(:))), ...
		strjoin(cellstr(constraint_types(:)'), ','));
end

% QP 目标:
% min 0.5*||u||^2 + 0.5*sum_i w_i*delta_i^2。
% HOCBF/PTCBF/PTCLF 可以使用不同的 slack weight。
H = diag([ones(n_u, 1); slack_weights(:)]);
f = zeros(n_u + numel(slack_rows), 1);
lb = [-inf(n_u, 1); zeros(numel(slack_rows), 1)];

options = optimoptions('quadprog', 'Display', 'off', ...
	'MaxIterations', 200, 'ConstraintTolerance', 1e-6, ...
	'OptimalityTolerance', 1e-6, 'StepTolerance', 1e-10);
qp_timer = tic;
[z_col, ~, exitflag, qp_output] = quadprog(H, f, A_solve, b_qp, [], [], lb, [], [], options);
qp_solve_seconds = toc(qp_timer);
qp_iterations = qp_output.iterations;
if exitflag <= 0 || isempty(z_col)
	row_norms = sqrt(sum(A_qp .^ 2, 2));
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
qp_slack = zeros(n_constraints, 1);
qp_slack(slack_rows) = z_col(n_u + 1:end);

% qp_residuals 是带 slack 后的残差:
% A_i*u - delta_i - b_i，应当 <= 0。
qp_residuals = A_solve * z_col - b_qp;
end
