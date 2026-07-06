function [v, diagnostics] = apply_hocbf_integral(stats, constraint_cfg, ...
    t, cumulative_variance)
mu = stats.mu;
grad_x = stats.sigma2_grad_x;
budget = constraint_cfg.integral_uncertainty_budget;
alpha1 = constraint_cfg.hocbf_alpha1;
alpha2 = constraint_cfg.hocbf_alpha2;
hocbf_enabled = struct_field_default(constraint_cfg, 'hocbf_enabled', true);

% beta_drift 是不加控制修正 u 时的方差变化率:
% beta_dot_nom = beta_t + grad_beta' * mu.
% 加控制后 beta_dot = beta_drift + grad_beta' * u.

beta_drift = stats.sigma2_t + sum(grad_x .* mu', 2); % 如果不加控制修正，只按照 GP 原始速度 mu 走，当前方差会怎么变化
average_variance = cumulative_variance / max(t, eps);

% Integral HOCBF 控制累计方差:
% integral_0^t beta(tau) dtau <= B*t + h_bar(t).
% b = B*t - cumulative_variance, 因此要求 psi0 = b + h_bar >= 0。
b = budget * t - cumulative_variance;
b_dot = budget - stats.sigma2;

% 当前 HOCBF relaxation 是常数 h_bar，不随时间收紧。
% 因此 h_bar_dot = 0, h_bar_ddot = 0。
h_bar = constraint_cfg.hocbf_relaxation_bound .* ones(size(t));
h_bar_dot = zeros(size(t));
h_bar_ddot = zeros(size(t));

% 二阶 HOCBF:
% psi1 = psi0_dot + alpha1*psi0,
% psi2 = psi1_dot + alpha2*psi1 >= 0.
% 展开后得到 QP 线性约束 grad_beta' * u + rho <= 0。
psi0 = b + h_bar;
psi0_dot = b_dot + h_bar_dot;
psi1 = psi0_dot + alpha1 * psi0;
hocbf_drift = -beta_drift + h_bar_ddot + alpha1 * psi0_dot + alpha2 * psi1;
rho = -hocbf_drift;
rho_beta_t = stats.sigma2_t;
rho_grad_beta_mu = beta_drift - stats.sigma2_t;
rho_minus_h_bar_ddot = -h_bar_ddot;
rho_minus_alpha1_psi0_dot = -alpha1 * psi0_dot;
rho_minus_alpha2_psi1 = -alpha2 * psi1;

grad_norm_sq = sum(grad_x .^ 2, 2);
grad_tol = constraint_cfg.grad_tol;
u = zeros(size(mu'));
terminal_info = terminal_variance_ptcbf(stats, constraint_cfg, ...
	t, beta_drift);
terminal_enabled = terminal_info.enabled;
clf_info = anchor_clf_info(stats, constraint_cfg, t);
clf_enabled = clf_info.enabled;

% HOCBF 的 QP 上界: grad_beta' * u <= integral_bound。
% residual_without_u 是 u=0 时的违反量，主要用于诊断。
integral_bound = -rho;
integral_residual_without_u = rho;
terminal_residual_without_u = terminal_info.residual_without_u;
clf_residual_without_u = clf_info.residual_without_u;
qp_exitflag = nan;
qp_active_constraint_count = 0;
qp_slack = zeros(0, 1);
hocbf_slack = 0.0;
terminal_slack = 0.0;
clf_slack = 0.0;

% 所有启用的 HOCBF/PTCBF/PTCLF 约束都统一进入 QP。
% 若三个约束都关闭，A_qp 为空，u 保持 0，即直接使用 GP 原速度。
[A_qp, b_qp, constraint_types] = active_qp_constraints(grad_x, ...
	integral_bound, terminal_info.bound, clf_info, hocbf_enabled);
if ~isempty(A_qp)
	[u_col, qp_exitflag, qp_slack, qp_residuals] = solve_slack_qp(A_qp, b_qp, ...
		constraint_types, ...
		constraint_cfg, t, stats, terminal_info, integral_residual_without_u, ...
		terminal_residual_without_u);
	u = reshape(u_col, 1, []);
	qp_active_constraint_count = sum(abs(qp_residuals) <= 1e-7);
	hocbf_rows = constraint_types == "integral";
	terminal_rows = constraint_types == "terminal";
	clf_rows = constraint_types == "anchor_clf";
	if any(hocbf_rows)
		hocbf_slack = max(qp_slack(hocbf_rows));
	end
	if any(terminal_rows)
		terminal_slack = max(qp_slack(terminal_rows));
	end
	if any(clf_rows)
		clf_slack = max(qp_slack(clf_rows));
	end
end
constraint_residual = sum(grad_x .* u, 2) + rho;
terminal_constraint_residual = sum(grad_x .* u, 2) - terminal_info.bound;
clf_raw_residual = clf_info.grad * u' - clf_info.bound;

% raw residual 是原始约束违反量；relaxed residual = raw residual - slack。
% 带 slack 的 QP 约束满足时 relaxed residual 应该 <= 0。
clf_constraint_residual = clf_raw_residual - clf_slack;
psi2 = -constraint_residual;
v = mu + u';
diagnostics.active = norm(u) > 0.0;
diagnostics.control_enabled = true;
diagnostics.hocbf_enabled = hocbf_enabled;
diagnostics.u = u';
diagnostics.correction_norm = norm(u);
diagnostics.max_abs_u = max(abs(u(:)));
diagnostics.grad_norm = norm(stats.sigma2_grad_x);
diagnostics.t = t;
diagnostics.hocbf_alpha1 = alpha1;
diagnostics.cumulative_variance = cumulative_variance;
diagnostics.time_average_variance = average_variance;
diagnostics.barrier_b = b;
diagnostics.relaxed_barrier_b = psi0;
diagnostics.psi0 = psi0;
diagnostics.psi0_dot = psi0_dot;
diagnostics.b_dot = b_dot;
diagnostics.h_bar = h_bar;
diagnostics.h_bar_dot = h_bar_dot;
diagnostics.alpha1_psi0 = alpha1 * psi0;
diagnostics.psi1 = psi1;
diagnostics.psi2 = psi2;
diagnostics.psi1_dot = psi2 - alpha2 * psi1;
diagnostics.rho = rho;
diagnostics.rho_beta_t = rho_beta_t;
diagnostics.rho_grad_beta_mu = rho_grad_beta_mu;
diagnostics.rho_minus_h_bar_ddot = rho_minus_h_bar_ddot;
diagnostics.rho_minus_alpha1_psi0_dot = rho_minus_alpha1_psi0_dot;
diagnostics.rho_minus_alpha2_psi1 = rho_minus_alpha2_psi1;
diagnostics.hocbf_constraint_residual = constraint_residual;
diagnostics.hocbf_relaxed_constraint_residual = constraint_residual - hocbf_slack;
diagnostics.hocbf_slack = hocbf_slack;
diagnostics.integral_bound = integral_bound;
diagnostics.ptcbf_enabled = terminal_enabled;
diagnostics.terminal_ptzf_initial_bound = terminal_info.ptzf_initial_bound;
diagnostics.terminal_ptzf_bound = terminal_info.ptzf_bound;
diagnostics.terminal_ptzf_bound_dot = terminal_info.ptzf_bound_dot;
diagnostics.terminal_inequality_h = terminal_info.inequality_h;
diagnostics.terminal_beta_cap = terminal_info.beta_cap;
diagnostics.terminal_beta_cap_dot = terminal_info.beta_cap_dot;
diagnostics.terminal_h = terminal_info.h;
diagnostics.terminal_alpha_h = terminal_info.alpha_h;
diagnostics.terminal_bound = terminal_info.bound;
diagnostics.terminal_constraint_residual = terminal_constraint_residual;
diagnostics.terminal_relaxed_constraint_residual = ...
	terminal_constraint_residual - terminal_slack;
diagnostics.terminal_slack = terminal_slack;
diagnostics.terminal_active = terminal_enabled && ...
	terminal_residual_without_u > 0.0;
diagnostics.ptclf_enabled = clf_enabled;
diagnostics.anchor_clf_v = clf_info.v;
diagnostics.anchor_clf_bound = clf_info.bound;
diagnostics.anchor_clf_residual = clf_constraint_residual;
diagnostics.anchor_clf_raw_residual = clf_raw_residual;
diagnostics.anchor_clf_residual_without_u = clf_residual_without_u;
diagnostics.anchor_clf_slack = clf_slack;
diagnostics.qp_slack = max([hocbf_slack, terminal_slack, clf_slack]);
diagnostics.qp_slack_vector = qp_slack;
diagnostics.anchor_clf_ptzf_bound = clf_info.ptzf_bound;
diagnostics.anchor_clf_ptzf_bound_dot = clf_info.ptzf_bound_dot;
diagnostics.anchor_clf_ptzf_cg = clf_info.ptzf_cg;
diagnostics.anchor_clf_cpt = clf_info.cpt;
diagnostics.qp_exitflag = qp_exitflag;
diagnostics.qp_active_constraint_count = qp_active_constraint_count;
diagnostics.sigma2 = stats.sigma2;
diagnostics.min_grad_norm_sq = min(grad_norm_sq);
end
