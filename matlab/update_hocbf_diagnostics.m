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
if isfield(info, 'hocbf_filter_active')
	hocbf_diag.trace_hocbf_filter_active(end + 1, 1) = ...
		info.hocbf_filter_active;
else
	hocbf_diag.trace_hocbf_filter_active(end + 1, 1) = info.hocbf_enabled;
end
if isfield(info, 'obstacle_filter_active')
	hocbf_diag.trace_obstacle_filter_active(end + 1, 1) = ...
		info.obstacle_filter_active;
else
	hocbf_diag.trace_obstacle_filter_active(end + 1, 1) = false;
end
hocbf_diag.trace_hocbf_row_active(end + 1, 1) = ...
	info.hocbf_row_active;
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
x_now = reshape(info.x, 1, []);
mu_now = reshape(info.mu, 1, []);
v_now = reshape(info.v, 1, []);
hocbf_diag.trace_x(end + 1, 1:numel(x_now)) = x_now;
hocbf_diag.trace_mu(end + 1, 1:numel(mu_now)) = mu_now;
hocbf_diag.trace_v(end + 1, 1:numel(v_now)) = v_now;
u_now = reshape(info.u, 1, []);
hocbf_diag.trace_u(end + 1, 1:numel(u_now)) = u_now;
u_ptclf_now = reshape(info.u_ptclf_reference, 1, []);
u_ptcbf_now = reshape(info.u_after_ptcbf, 1, []);
du_ptcbf_now = reshape(info.u_ptcbf_correction, 1, []);
du_hocbf_now = reshape(info.u_hocbf_correction, 1, []);
hocbf_diag.trace_u_ptclf_reference(end + 1, 1:numel(u_now)) = u_ptclf_now;
hocbf_diag.trace_u_after_ptcbf(end + 1, 1:numel(u_now)) = u_ptcbf_now;
hocbf_diag.trace_u_ptcbf_correction(end + 1, 1:numel(u_now)) = du_ptcbf_now;
hocbf_diag.trace_u_hocbf_correction(end + 1, 1:numel(u_now)) = du_hocbf_now;
hocbf_diag.trace_stage_ptclf_residual_at_reference(end + 1, 1) = ...
	info.stage_ptclf_residual_at_reference;
hocbf_diag.trace_stage_ptclf_residual_after_ptcbf(end + 1, 1) = ...
	info.stage_ptclf_residual_after_ptcbf;
hocbf_diag.trace_stage_ptclf_residual_after_hocbf(end + 1, 1) = ...
	info.stage_ptclf_residual_after_hocbf;
hocbf_diag.trace_stage_obstacle_residual_after_ptcbf(end + 1, 1) = ...
	info.stage_obstacle_residual_after_ptcbf;
hocbf_diag.trace_stage_obstacle_residual_after_hocbf(end + 1, 1) = ...
	info.stage_obstacle_residual_after_hocbf;
hocbf_diag.trace_stage_hocbf_residual_after_ptcbf(end + 1, 1) = ...
	info.stage_hocbf_residual_after_ptcbf;
hocbf_diag.trace_stage_hocbf_residual_after_hocbf(end + 1, 1) = ...
	info.stage_hocbf_residual_after_hocbf;
u_hocbf_now = reshape(info.u_contribution_hocbf, 1, []);
u_terminal_now = reshape(info.u_contribution_terminal, 1, []);
u_anchor_now = reshape(info.u_contribution_anchor_clf, 1, []);
u_obstacle_now = reshape(info.u_contribution_obstacle, 1, []);
hocbf_diag.trace_u_contribution_hocbf(end + 1, 1:numel(u_now)) = ...
	u_hocbf_now;
hocbf_diag.trace_u_contribution_terminal(end + 1, 1:numel(u_now)) = ...
	u_terminal_now;
hocbf_diag.trace_u_contribution_anchor_clf(end + 1, 1:numel(u_now)) = ...
	u_anchor_now;
hocbf_diag.trace_u_contribution_obstacle(end + 1, 1:numel(u_now)) = ...
	u_obstacle_now;
hocbf_diag.trace_u_contribution_hocbf_norm(end + 1, 1) = ...
	info.u_contribution_hocbf_norm;
hocbf_diag.trace_u_contribution_terminal_norm(end + 1, 1) = ...
	info.u_contribution_terminal_norm;
hocbf_diag.trace_u_contribution_anchor_clf_norm(end + 1, 1) = ...
	info.u_contribution_anchor_clf_norm;
hocbf_diag.trace_u_contribution_obstacle_norm(end + 1, 1) = ...
	info.u_contribution_obstacle_norm;
hocbf_diag.trace_kkt_u_reconstruction_error(end + 1, 1) = ...
	info.kkt_u_reconstruction_error;
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
hocbf_diag.trace_anchor_clf_first_v(end + 1, 1) = info.anchor_clf_first_v;
hocbf_diag.trace_anchor_clf_first_ptzf_bound(end + 1, 1) = ...
	info.anchor_clf_first_ptzf_bound;
hocbf_diag.trace_anchor_clf_first_bound(end + 1, 1) = ...
	info.anchor_clf_first_bound;
hocbf_diag.trace_anchor_clf_first_effective_bound(end + 1, 1) = ...
	info.anchor_clf_first_effective_bound;
hocbf_diag.trace_anchor_clf_first_raw_residual(end + 1, 1) = ...
	info.anchor_clf_first_raw_residual;
hocbf_diag.trace_anchor_clf_first_residual(end + 1, 1) = ...
	info.anchor_clf_first_residual;
hocbf_diag.trace_anchor_clf_first_slack(end + 1, 1) = ...
	info.anchor_clf_first_slack;
hocbf_diag.trace_anchor_clf_first_grad_norm(end + 1, 1) = ...
	info.anchor_clf_first_grad_norm;
hocbf_diag.trace_anchor_clf_first_owner_grad_norm(end + 1, 1) = ...
	info.anchor_clf_first_owner_grad_norm;
hocbf_diag.trace_anchor_clf_first_row_active(end + 1, 1) = ...
	info.anchor_clf_first_row_active;
hocbf_diag.trace_anchor_clf_first_row_present(end + 1, 1) = ...
	info.anchor_clf_first_row_present;
hocbf_diag.trace_anchor_clf_first_u_contribution_norm(end + 1, 1) = ...
	info.anchor_clf_first_u_contribution_norm;
hocbf_diag.trace_anchor_clf_last_v(end + 1, 1) = info.anchor_clf_last_v;
hocbf_diag.trace_anchor_clf_last_ptzf_bound(end + 1, 1) = ...
	info.anchor_clf_last_ptzf_bound;
hocbf_diag.trace_anchor_clf_last_bound(end + 1, 1) = ...
	info.anchor_clf_last_bound;
hocbf_diag.trace_anchor_clf_last_effective_bound(end + 1, 1) = ...
	info.anchor_clf_last_effective_bound;
hocbf_diag.trace_anchor_clf_last_raw_residual(end + 1, 1) = ...
	info.anchor_clf_last_raw_residual;
hocbf_diag.trace_anchor_clf_last_residual(end + 1, 1) = ...
	info.anchor_clf_last_residual;
hocbf_diag.trace_anchor_clf_last_slack(end + 1, 1) = ...
	info.anchor_clf_last_slack;
hocbf_diag.trace_anchor_clf_last_grad_norm(end + 1, 1) = ...
	info.anchor_clf_last_grad_norm;
hocbf_diag.trace_anchor_clf_last_owner_grad_norm(end + 1, 1) = ...
	info.anchor_clf_last_owner_grad_norm;
hocbf_diag.trace_anchor_clf_last_row_active(end + 1, 1) = ...
	info.anchor_clf_last_row_active;
hocbf_diag.trace_anchor_clf_last_row_present(end + 1, 1) = ...
	info.anchor_clf_last_row_present;
hocbf_diag.trace_anchor_clf_last_u_contribution_norm(end + 1, 1) = ...
	info.anchor_clf_last_u_contribution_norm;
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
% 行范数追踪(供 slack 权重重新标定)。
hocbf_diag.trace_grad_norm(end + 1, 1) = info.grad_norm;
if isfield(info, 'anchor_clf_grad_norm')
	hocbf_diag.trace_anchor_clf_grad_norm(end + 1, 1) = info.anchor_clf_grad_norm;
end
% 障碍 CBF 诊断汇总(fix 5)。info.obstacle_* 由 apply_hocbf_integral 恒设置。
if isfield(info, 'obstacle_enabled')
	hocbf_diag.trace_obstacle_h_min(end + 1, 1) = info.obstacle_h_min;
	hocbf_diag.trace_obstacle_slack(end + 1, 1) = info.obstacle_slack;
	hocbf_diag.trace_obstacle_n_rows(end + 1, 1) = info.obstacle_n_rows;
	hocbf_diag.trace_obstacle_max_residual_without_u(end + 1, 1) = ...
		info.obstacle_max_residual_without_u;
	hocbf_diag.trace_obstacle_candidate_count(end + 1, 1) = ...
		info.obstacle_candidate_count;
	hocbf_diag.trace_obstacle_skipped_count(end + 1, 1) = ...
		info.obstacle_skipped_count;
	hocbf_diag.trace_obstacle_unsafe_count(end + 1, 1) = ...
		info.obstacle_unsafe_count;
	hocbf_diag.trace_obstacle_unsafe_skipped_count(end + 1, 1) = ...
		info.obstacle_unsafe_skipped_count;
	hocbf_diag.trace_obstacle_min_point_idx(end + 1, 1) = ...
		info.obstacle_min_point_idx;
	hocbf_diag.trace_obstacle_min_obstacle_idx(end + 1, 1) = ...
		info.obstacle_min_obstacle_idx;
	hocbf_diag.trace_obstacle_min_grad_true_norm(end + 1, 1) = ...
		info.obstacle_min_grad_true_norm;
	hocbf_diag.trace_obstacle_min_grad_ctrl_norm(end + 1, 1) = ...
		info.obstacle_min_grad_ctrl_norm;
	hocbf_diag.trace_obstacle_min_row_active(end + 1, 1) = ...
		info.obstacle_min_row_active;
	hocbf_diag.trace_obstacle_min_phi(end + 1, 1) = ...
		info.obstacle_min_phi;
	hocbf_diag.trace_obstacle_min_drift(end + 1, 1) = ...
		info.obstacle_min_drift;
	hocbf_diag.trace_obstacle_min_qp_control(end + 1, 1) = ...
		info.obstacle_min_qp_control;
	hocbf_diag.trace_obstacle_min_true_control(end + 1, 1) = ...
		info.obstacle_min_true_control;
	hocbf_diag.trace_obstacle_min_qp_hdot(end + 1, 1) = ...
		info.obstacle_min_qp_hdot;
	hocbf_diag.trace_obstacle_min_true_hdot(end + 1, 1) = ...
		info.obstacle_min_true_hdot;
	hocbf_diag.trace_obstacle_min_qp_residual(end + 1, 1) = ...
		info.obstacle_min_qp_residual;
	hocbf_diag.trace_obstacle_min_true_residual(end + 1, 1) = ...
		info.obstacle_min_true_residual;
	hocbf_diag.trace_obstacle_min_row_slack(end + 1, 1) = ...
		info.obstacle_min_row_slack;
	if info.obstacle_enabled
		hocbf_diag.obstacle_eval_count = hocbf_diag.obstacle_eval_count + 1;
		hocbf_diag.obstacle_candidate_count = ...
			hocbf_diag.obstacle_candidate_count + ...
			info.obstacle_candidate_count;
		hocbf_diag.obstacle_skipped_count = ...
			hocbf_diag.obstacle_skipped_count + ...
			info.obstacle_skipped_count;
		hocbf_diag.obstacle_unsafe_count = ...
			hocbf_diag.obstacle_unsafe_count + ...
			info.obstacle_unsafe_count;
		hocbf_diag.obstacle_unsafe_skipped_count = ...
			hocbf_diag.obstacle_unsafe_skipped_count + ...
			info.obstacle_unsafe_skipped_count;
		if isfinite(info.obstacle_h_min)
			if info.obstacle_h_min < hocbf_diag.min_obstacle_h
				hocbf_diag.min_obstacle_h = info.obstacle_h_min;
				hocbf_diag.min_obstacle_h_t = info.t;
				hocbf_diag.min_obstacle_h_sample_idx = sample_idx;
				hocbf_diag.min_obstacle_h_step_idx = step_idx;
			end
			if info.obstacle_h_min < 0 && ...
					~isfinite(hocbf_diag.obstacle_first_unsafe_t)
				hocbf_diag.obstacle_first_unsafe_t = info.t;
			end
		end
		hocbf_diag.max_obstacle_slack = max( ...
			hocbf_diag.max_obstacle_slack, info.obstacle_slack);
		if info.obstacle_max_residual_without_u > 0
			hocbf_diag.obstacle_active_count = ...
				hocbf_diag.obstacle_active_count + 1;
		end
	end
end
end
