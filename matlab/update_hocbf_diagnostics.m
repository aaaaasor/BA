function hocbf_diag = update_hocbf_diagnostics(hocbf_diag, info, ...
	sample_idx, step_idx, stage_name)
switch stage_name
	case 'k1'
		stage_idx = 1;
	case 'k2'
		stage_idx = 2;
	case 'k3'
		stage_idx = 3;
	case 'k4'
		stage_idx = 4;
	otherwise
		stage_idx = nan;
end
hocbf_diag.trace_sample_idx(end + 1, 1) = sample_idx;
hocbf_diag.trace_step_idx(end + 1, 1) = step_idx;
hocbf_diag.trace_stage_idx(end + 1, 1) = stage_idx;
hocbf_diag.trace_t(end + 1, 1) = info.t;
hocbf_diag.trace_psi0(end + 1, 1) = info.psi0;
hocbf_diag.trace_psi1(end + 1, 1) = info.psi1;
hocbf_diag.trace_psi2(end + 1, 1) = info.psi2;
hocbf_diag.trace_psi1_dot(end + 1, 1) = info.psi1_dot;
hocbf_diag.trace_psi0_dot(end + 1, 1) = info.psi0_dot;
hocbf_diag.trace_b_dot(end + 1, 1) = info.b_dot;
hocbf_diag.trace_h_bar_dot(end + 1, 1) = info.h_bar_dot;
hocbf_diag.trace_alpha1_psi0(end + 1, 1) = info.alpha1_psi0;
hocbf_diag.trace_rho(end + 1, 1) = info.rho;
hocbf_diag.trace_rho_components(end + 1, :) = [ ...
	info.rho_beta_t, info.rho_grad_beta_mu, ...
	info.rho_minus_h_bar_ddot, info.rho_minus_alpha1_psi0_dot, ...
	info.rho_minus_alpha2_psi1];
u_now = reshape(info.u, 1, []);
hocbf_diag.trace_u(end + 1, 1:numel(u_now)) = u_now;
hocbf_diag.trace_correction_norm(end + 1, 1) = info.correction_norm;
hocbf_diag.trace_max_abs_u(end + 1, 1) = info.max_abs_u;
hocbf_diag.trace_barrier_b(end + 1, 1) = info.barrier_b;
hocbf_diag.trace_relaxed_barrier_b(end + 1, 1) = info.relaxed_barrier_b;
hocbf_diag.trace_h_bar(end + 1, 1) = info.h_bar;
hocbf_diag.trace_sigma2(end + 1, 1) = info.sigma2;
hocbf_diag.trace_cumulative_variance(end + 1, 1) = info.cumulative_variance;
hocbf_diag.trace_time_average_variance(end + 1, 1) = info.time_average_variance;
hocbf_diag.max_correction_norm = max(hocbf_diag.max_correction_norm, info.correction_norm);
hocbf_diag.max_grad_norm = max(hocbf_diag.max_grad_norm, info.grad_norm);
hocbf_diag.max_sigma2 = max(hocbf_diag.max_sigma2, info.sigma2);
hocbf_diag.min_barrier_b = min(hocbf_diag.min_barrier_b, info.barrier_b);
hocbf_diag.min_relaxed_barrier_b = min(hocbf_diag.min_relaxed_barrier_b, info.relaxed_barrier_b);
if isfinite(info.psi1) && info.psi1 < hocbf_diag.min_psi1
	hocbf_diag.min_psi1 = info.psi1;
	hocbf_diag.min_psi1_sample_idx = sample_idx;
	hocbf_diag.min_psi1_step_idx = step_idx;
	hocbf_diag.min_psi1_stage = stage_name;
	hocbf_diag.min_psi1_t = info.t;
	hocbf_diag.min_psi1_sigma2 = info.sigma2;
	hocbf_diag.min_psi1_cumulative_variance = info.cumulative_variance;
	hocbf_diag.min_psi1_time_average_variance = info.time_average_variance;
	hocbf_diag.min_psi1_barrier_b = info.barrier_b;
	hocbf_diag.min_psi1_psi0 = info.psi0;
	hocbf_diag.min_psi1_psi0_dot = info.psi0_dot;
	hocbf_diag.min_psi1_h_bar = info.h_bar;
	hocbf_diag.min_psi1_h_bar_dot = info.h_bar_dot;
	hocbf_diag.min_psi1_alpha_psi0 = info.hocbf_alpha1 * info.psi0;
end
hocbf_diag.min_psi2 = min(hocbf_diag.min_psi2, info.psi2);
hocbf_diag.max_hocbf_constraint_residual = max(hocbf_diag.max_hocbf_constraint_residual, info.hocbf_constraint_residual);
end
