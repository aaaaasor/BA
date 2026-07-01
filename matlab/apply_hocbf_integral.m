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
[h_bar, h_bar_dot, h_bar_ddot] = ptzf_bound(t, constraint_cfg);
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
control_activation_time = struct_field_default(constraint_cfg, ...
	'control_activation_time', 0.0);
control_enabled = t >= control_activation_time;
active = control_enabled & (rho > 0.0) & (grad_norm_sq > grad_tol);
u = zeros(size(mu'));
if any(active)
	denom = grad_norm_sq(active);
	u(active, :) = grad_x(active, :) .* (-rho(active) ./ denom);
end
constraint_residual = sum(grad_x .* u, 2) + rho;
psi2 = -constraint_residual;
v = mu + u';
diagnostics.active = any(active);
diagnostics.control_enabled = control_enabled;
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
diagnostics.sigma2 = stats.sigma2;
diagnostics.min_grad_norm_sq = min(grad_norm_sq);
end
