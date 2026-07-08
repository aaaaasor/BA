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
hocbf_diag.trace_hocbf_enabled(end + 1, 1) = info.hocbf_enabled;
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
hocbf_diag.trace_integral_bound(end + 1, 1) = info.integral_bound;
hocbf_diag.trace_hocbf_constraint_residual(end + 1, 1) = ...
	info.hocbf_constraint_residual;
hocbf_diag.trace_hocbf_relaxed_constraint_residual(end + 1, 1) = ...
	info.hocbf_relaxed_constraint_residual;
hocbf_diag.trace_hocbf_slack(end + 1, 1) = info.hocbf_slack;
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
hocbf_diag.trace_terminal_ptzf_initial_bound(end + 1, 1) = ...
	info.terminal_ptzf_initial_bound;
hocbf_diag.trace_ptcbf_enabled(end + 1, 1) = info.ptcbf_enabled;
hocbf_diag.trace_terminal_ptzf_bound(end + 1, 1) = info.terminal_ptzf_bound;
hocbf_diag.trace_terminal_ptzf_bound_dot(end + 1, 1) = ...
	info.terminal_ptzf_bound_dot;
hocbf_diag.trace_terminal_inequality_h(end + 1, 1) = ...
	info.terminal_inequality_h;
hocbf_diag.trace_terminal_beta_cap(end + 1, 1) = info.terminal_beta_cap;
hocbf_diag.trace_terminal_beta_cap_dot(end + 1, 1) = info.terminal_beta_cap_dot;
hocbf_diag.trace_terminal_h(end + 1, 1) = info.terminal_h;
hocbf_diag.trace_terminal_bound(end + 1, 1) = info.terminal_bound;
hocbf_diag.trace_terminal_constraint_residual(end + 1, 1) = ...
	info.terminal_constraint_residual;
hocbf_diag.trace_terminal_relaxed_constraint_residual(end + 1, 1) = ...
	info.terminal_relaxed_constraint_residual;
hocbf_diag.trace_terminal_slack(end + 1, 1) = info.terminal_slack;
hocbf_diag.trace_anchor_clf_v(end + 1, 1) = info.anchor_clf_v;
hocbf_diag.trace_anchor_clf_bound(end + 1, 1) = info.anchor_clf_bound;
hocbf_diag.trace_anchor_clf_residual(end + 1, 1) = ...
	info.anchor_clf_residual;
hocbf_diag.trace_anchor_clf_raw_residual(end + 1, 1) = ...
	info.anchor_clf_raw_residual;
hocbf_diag.trace_anchor_clf_slack(end + 1, 1) = ...
	info.anchor_clf_slack;
hocbf_diag.trace_qp_slack(end + 1, 1) = info.qp_slack;
hocbf_diag.trace_anchor_clf_ptzf_bound(end + 1, 1) = ...
	info.anchor_clf_ptzf_bound;
hocbf_diag.trace_anchor_clf_ptzf_bound_dot(end + 1, 1) = ...
	info.anchor_clf_ptzf_bound_dot;
hocbf_diag.trace_anchor_clf_ptzf_cg(end + 1, 1) = ...
	info.anchor_clf_ptzf_cg;
hocbf_diag.trace_anchor_clf_cpt(end + 1, 1) = info.anchor_clf_cpt;
hocbf_diag.trace_qp_exitflag(end + 1, 1) = info.qp_exitflag;
hocbf_diag.trace_qp_iterations(end + 1, 1) = info.qp_iterations;
hocbf_diag.trace_qp_active_constraint_count(end + 1, 1) = ...
	info.qp_active_constraint_count;
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
hocbf_diag.max_hocbf_relaxed_constraint_residual = max( ...
	hocbf_diag.max_hocbf_relaxed_constraint_residual, ...
	info.hocbf_relaxed_constraint_residual);
if isfinite(info.terminal_h) && info.terminal_h < hocbf_diag.min_terminal_h
	hocbf_diag.min_terminal_h = info.terminal_h;
	hocbf_diag.min_terminal_h_sample_idx = sample_idx;
	hocbf_diag.min_terminal_h_step_idx = step_idx;
	hocbf_diag.min_terminal_h_stage = stage_name;
	hocbf_diag.min_terminal_h_t = info.t;
	hocbf_diag.min_terminal_h_sigma2 = info.sigma2;
	hocbf_diag.min_terminal_h_inequality_h = info.terminal_inequality_h;
	hocbf_diag.min_terminal_h_ptzf_bound = info.terminal_ptzf_bound;
	hocbf_diag.min_terminal_h_beta_cap = info.terminal_beta_cap;
	hocbf_diag.min_terminal_h_residual = info.terminal_constraint_residual;
end
hocbf_diag.max_terminal_constraint_residual = max( ...
	hocbf_diag.max_terminal_constraint_residual, ...
	info.terminal_constraint_residual);
hocbf_diag.max_terminal_relaxed_constraint_residual = max( ...
	hocbf_diag.max_terminal_relaxed_constraint_residual, ...
	info.terminal_relaxed_constraint_residual);
if isfinite(info.qp_exitflag)
	hocbf_diag.min_qp_exitflag = min(hocbf_diag.min_qp_exitflag, ...
		info.qp_exitflag);
end
if isfinite(info.qp_iterations)
	hocbf_diag.max_qp_iterations = max(hocbf_diag.max_qp_iterations, ...
		info.qp_iterations);
end
if isfield(info, 'qp_solve_seconds') && isfinite(info.qp_solve_seconds)
	hocbf_diag.total_qp_solve_seconds = hocbf_diag.total_qp_solve_seconds + ...
		info.qp_solve_seconds;
end
hocbf_diag.max_qp_active_constraint_count = max( ...
	hocbf_diag.max_qp_active_constraint_count, ...
	info.qp_active_constraint_count);
end
