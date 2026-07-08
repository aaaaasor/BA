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
	error(['Constrained QP infeasible at t=%.6g: active constraints=%s, beta=%.6g, ', ...
		'beta_cap=%.6g, h_T=%.6g, integral residual=%.6g, ', ...
		'terminal residual=%.6g, grad norm=%.6g, exitflag=%d, iterations=%d.'], ...
		t, strjoin(cellstr(constraint_types(:)'), ','), ...
		stats.sigma2, terminal_info.beta_cap, terminal_info.h, ...
		integral_residual_without_u, terminal_residual_without_u, ...
		norm(stats.sigma2_grad_x), exitflag, qp_iterations);
end
u_col = z_col(1:n_u);
qp_slack = zeros(n_constraints, 1);
qp_slack(slack_rows) = z_col(n_u + 1:end);

% qp_residuals 是带 slack 后的残差:
% A_i*u - delta_i - b_i，应当 <= 0。
qp_residuals = A_solve * z_col - b_qp;
end
