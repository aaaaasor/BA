function [v, diagnostics] = apply_hocbf_integral(stats, constraint_cfg, ...
    t, cumulative_variance)
mu = stats.mu;
grad_x = stats.sigma2_grad_x;
budget = constraint_cfg.integral_uncertainty_budget;
alpha1 = constraint_cfg.hocbf_alpha1;
alpha2 = constraint_cfg.hocbf_alpha2;

beta_drift = stats.sigma2_t + sum(grad_x .* mu', 2); % 如果不加控制修正，只按照 GP 原始速度 mu 走，当前方差会怎么变化
average_variance = cumulative_variance / max(t, eps);
b = budget * t - cumulative_variance;
b_dot = budget - stats.sigma2;
[h_bar, h_bar_dot, h_bar_ddot] = hocbf_relaxation_bound(t, constraint_cfg);
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
terminal_info = local_terminal_variance_ptcbf(stats, constraint_cfg, ...
	t, beta_drift);
terminal_enabled = terminal_info.enabled;
integral_bound = -rho;
integral_residual_without_u = rho;
terminal_residual_without_u = terminal_info.residual_without_u;
qp_exitflag = nan;
qp_active_constraint_count = 0;

if terminal_enabled
	if grad_norm_sq <= grad_tol && ...
			(integral_residual_without_u > 0.0 || ...
			terminal_residual_without_u > 0.0)
		error(['Terminal/integral PTCBF infeasible at t=%.6g: ', ...
			'beta=%.6g, beta_cap=%.6g, h_T=%.6g, ', ...
			'integral residual=%.6g, terminal residual=%.6g, ', ...
			'grad norm=%.6g.'], t, stats.sigma2, ...
			terminal_info.beta_cap, terminal_info.h, ...
			integral_residual_without_u, terminal_residual_without_u, ...
			sqrt(grad_norm_sq));
	end
	[A_qp, b_qp] = local_active_qp_constraints(grad_x, ...
		integral_bound, terminal_info.bound, grad_tol);
	if ~isempty(A_qp)
		[u_col, qp_exitflag] = local_solve_qp(A_qp, b_qp, ...
			t, stats, terminal_info, integral_residual_without_u, ...
			terminal_residual_without_u);
		u = reshape(u_col, 1, []);
		qp_residuals = A_qp * u_col - b_qp;
		qp_active_constraint_count = sum(abs(qp_residuals) <= 1e-7);
	end
else
	active = (rho > 0.0) & (grad_norm_sq > grad_tol);
	if any(active)
		denom = grad_norm_sq(active);
		u(active, :) = grad_x(active, :) .* (-rho(active) ./ denom);
	end
end
constraint_residual = sum(grad_x .* u, 2) + rho;
terminal_constraint_residual = sum(grad_x .* u, 2) - terminal_info.bound;
psi2 = -constraint_residual;
v = mu + u';
diagnostics.active = norm(u) > 0.0;
diagnostics.control_enabled = true;
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
diagnostics.integral_bound = integral_bound;
diagnostics.terminal_variance_enabled = terminal_enabled;
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
diagnostics.terminal_active = terminal_enabled && ...
	terminal_residual_without_u > 0.0;
diagnostics.qp_exitflag = qp_exitflag;
diagnostics.qp_active_constraint_count = qp_active_constraint_count;
diagnostics.sigma2 = stats.sigma2;
diagnostics.min_grad_norm_sq = min(grad_norm_sq);
end

function terminal_info = local_terminal_variance_ptcbf(stats, ...
	constraint_cfg, t, beta_drift)
terminal_info.enabled = struct_field_default(constraint_cfg, ...
	'terminal_variance_enabled', false);
terminal_info.ptzf_initial_bound = nan;
terminal_info.ptzf_bound = nan;
terminal_info.ptzf_bound_dot = nan;
terminal_info.inequality_h = nan;
terminal_info.beta_cap = nan;
terminal_info.beta_cap_dot = nan;
terminal_info.h = nan;
terminal_info.alpha_h = nan;
terminal_info.bound = inf;
terminal_info.residual_without_u = -inf;
if ~terminal_info.enabled
	return;
end

beta_final = constraint_cfg.terminal_variance_beta_final;
if ~isfield(constraint_cfg, 'terminal_variance_ptzf_hbar0')
	error(['Terminal variance PTCBF initial bound must be set per ', ...
		'sample by rk4_rollout.']);
end
hbar0 = constraint_cfg.terminal_variance_ptzf_hbar0;
gamma = constraint_cfg.terminal_variance_ptzf_gamma;
alpha_terminal = constraint_cfg.terminal_variance_alpha;
remaining_tau = max(1.0 - t, eps);
shape = t ./ remaining_tau;
shape_dot = remaining_tau .^ (-1.0) + t .* remaining_tau .^ (-2.0);
ptzf_bound = hbar0 .* exp(-gamma .* shape);
ptzf_bound_dot = -gamma .* shape_dot .* ptzf_bound;
terminal_inequality_h = stats.sigma2 - beta_final;
beta_cap = beta_final + ptzf_bound;
beta_cap_dot = ptzf_bound_dot;
h_terminal = ptzf_bound - terminal_inequality_h;
alpha_h = alpha_terminal .* h_terminal;
bound = -beta_drift + alpha_h + ptzf_bound_dot;

terminal_info.ptzf_bound = ptzf_bound;
terminal_info.ptzf_initial_bound = hbar0;
terminal_info.ptzf_bound_dot = ptzf_bound_dot;
terminal_info.inequality_h = terminal_inequality_h;
terminal_info.beta_cap = beta_cap;
terminal_info.beta_cap_dot = beta_cap_dot;
terminal_info.h = h_terminal;
terminal_info.alpha_h = alpha_h;
terminal_info.bound = bound;
terminal_info.residual_without_u = -bound;
end

function [A_qp, b_qp] = local_active_qp_constraints(grad_x, ...
	integral_bound, terminal_bound, grad_tol)
A_qp = [];
b_qp = [];
if sum(grad_x .^ 2, 2) <= grad_tol
	return;
end
A_qp = grad_x;
b_qp = min(integral_bound, terminal_bound);
end

function [u_col, exitflag] = local_solve_qp(A_qp, b_qp, t, stats, ...
	terminal_info, integral_residual_without_u, terminal_residual_without_u)
if ~(exist('quadprog', 'file') == 2 || exist('quadprog', 'builtin') == 5)
	error('Terminal variance PTCBF requires quadprog from Optimization Toolbox.');
end
n_u = size(A_qp, 2);
H = eye(n_u);
f = zeros(n_u, 1);
options = optimoptions('quadprog', 'Display', 'off');
[u_col, ~, exitflag] = quadprog(H, f, A_qp, b_qp, [], [], [], [], [], options);
if exitflag <= 0 || isempty(u_col)
	error(['PTCBF QP infeasible at t=%.6g: beta=%.6g, ', ...
		'beta_cap=%.6g, h_T=%.6g, integral residual=%.6g, ', ...
		'terminal residual=%.6g, grad norm=%.6g, exitflag=%d.'], ...
		t, stats.sigma2, terminal_info.beta_cap, terminal_info.h, ...
		integral_residual_without_u, terminal_residual_without_u, ...
		norm(stats.sigma2_grad_x), exitflag);
end
end
