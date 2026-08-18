function [v, diagnostics] = apply_hocbf_integral(stats, constraint_cfg, ...
    t, cumulative_variance)
mu = stats.mu;
grad_x = stats.sigma2_grad_x;
budget = constraint_cfg.integral_uncertainty_budget;
alpha1 = constraint_cfg.hocbf_alpha1;
alpha2 = constraint_cfg.hocbf_alpha2;
hocbf_enabled = struct_field_default(constraint_cfg, 'hocbf_enabled', true);
hocbf_filter_end_time = struct_field_default(constraint_cfg, ...
	'hocbf_filter_end_time', inf);
hocbf_filter_active = hocbf_enabled && t < hocbf_filter_end_time;

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
terminal_info = terminal_variance_ptcbf(stats, constraint_cfg, ...
	t, beta_drift);
terminal_enabled = terminal_info.enabled;
clf_info = anchor_clf_info(stats, constraint_cfg, t);
clf_enabled = clf_info.enabled;
obstacle_info = obstacle_cbf_info(stats, constraint_cfg, t);
boundary_info = track_boundary_cbf_info(stats, constraint_cfg, t);
[obstacle_info, boundary_info] = joint_safety_softmin_info( ...
	obstacle_info, boundary_info, stats, constraint_cfg, t);
obstacle_enabled = obstacle_info.enabled;
sequential_increment_qp = struct_field_default(constraint_cfg, ...
	'sequential_increment_qp_enabled', false);
endpoint_clf_infos = struct([]);
if sequential_increment_qp && clf_enabled
	endpoint_clf_infos = anchor_clf_endpoint_infos(stats, constraint_cfg, t);
end

% HOCBF 的 QP 上界: grad_beta' * u <= integral_bound。
% residual_without_u 是 u=0 时的违反量，主要用于诊断。
integral_bound = -rho;
integral_residual_without_u = rho;
terminal_residual_without_u = terminal_info.residual_without_u;
clf_residual_without_u = clf_info.residual_without_u;
qp_exitflag = nan;
qp_iterations = nan;
qp_solve_seconds = 0.0;
qp_active_constraint_count = 0;
qp_slack = zeros(0, 1);
hocbf_slack = 0.0;
terminal_slack = 0.0;
clf_slack = 0.0;
obstacle_slack = 0.0;
n_u = numel(mu);

% 梯度退化检测(HOCBF/terminal 共用同一个 grad_x): grad_x 趋近 0 时(如
% 恰好落在 LoG-GP 局部专家拼接边界)，u 对这两行没有一阶影响力，跳过
% (导师方案: grad 太小时让 u=0)。
grad_x_active = norm(grad_x) >= grad_tol;

% 所有启用的 HOCBF/PTCBF/PTCLF/障碍 CBF 约束都统一进入 QP。
% 若约束都关闭，A_qp 为空，u 保持 0，即直接使用 GP 原速度。
if sequential_increment_qp
	sequential_result = solve_sequential_increment_qp(grad_x, ...
		integral_bound, terminal_info.bound, endpoint_clf_infos, ...
		hocbf_filter_active, obstacle_info, boundary_info, grad_x_active, ...
		constraint_cfg, t, ...
		stats, terminal_info, integral_residual_without_u, ...
		terminal_residual_without_u);
	u_col = sequential_result.u;
	u_ptclf_reference_col = sequential_result.u_ptclf_reference;
	u_after_ptcbf_col = sequential_result.u_after_ptcbf;
	stage_residuals_ptclf = sequential_result.residuals_at_ptclf_reference;
	stage_residuals_ptcbf = sequential_result.residuals_after_ptcbf;
	stage_residuals_hocbf = sequential_result.residuals_after_hocbf;
	qp_exitflag = sequential_result.exitflag;
	qp_slack = sequential_result.slack;
	qp_residuals = sequential_result.relaxed_residuals;
	qp_iterations = sequential_result.iterations;
	qp_solve_seconds = sequential_result.solve_seconds;
	qp_u_row_contributions = sequential_result.row_contributions;
	qp_u_equality_contribution = sequential_result.equality_contribution;
	constraint_types = sequential_result.constraint_types;
	qp_active_constraint_count = ...
		sequential_result.active_constraint_count;
else
	[A_qp, b_qp, constraint_types] = active_qp_constraints(grad_x, ...
		integral_bound, terminal_info.bound, clf_info, hocbf_filter_active, ...
		obstacle_info, grad_x_active, boundary_info);
	has_control_equalities = ~isempty(struct_field_default( ...
		constraint_cfg, 'endpoint_hold_velocity_matrix', []));
	if ~isempty(A_qp) || has_control_equalities
		[u_col, qp_exitflag, qp_slack, qp_residuals, qp_iterations, ...
			qp_solve_seconds, qp_u_row_contributions, ...
			qp_u_equality_contribution] = solve_slack_qp( ...
			A_qp, b_qp, constraint_types, constraint_cfg, t, stats, ...
			terminal_info, integral_residual_without_u, ...
			terminal_residual_without_u);
	else
		u_col = zeros(n_u, 1);
		qp_u_row_contributions = zeros(0, n_u);
		qp_u_equality_contribution = zeros(n_u, 1);
		constraint_types = strings(0, 1);
	end
	u_ptclf_reference_col = u_col;
	u_after_ptcbf_col = u_col;
	stage_residuals_ptclf = nan(size(constraint_types));
	stage_residuals_ptcbf = nan(size(constraint_types));
	stage_residuals_hocbf = nan(size(constraint_types));
end
u = reshape(u_col, 1, []);
v = mu + u';
if ~struct_field_default(constraint_cfg, 'diagnostics', false)
	if struct_field_default(constraint_cfg, ...
			'control_trace_enabled', false)
		diagnostics = struct( ...
			'sigma2', stats.sigma2, ...
			't', t, ...
			'mu', reshape(mu, 1, []), ...
			'v', reshape(v, 1, []), ...
			'u', u, ...
			'u_ptclf_reference', reshape(u_ptclf_reference_col, 1, []), ...
			'u_after_ptcbf', reshape(u_after_ptcbf_col, 1, []), ...
			'u_ptcbf_correction', reshape( ...
				u_after_ptcbf_col - u_ptclf_reference_col, 1, []), ...
			'u_hocbf_correction', reshape( ...
				u_col - u_after_ptcbf_col, 1, []));
		return;
	end
	% RK4 only needs sigma2 to integrate cumulative variance when trace
	% diagnostics are disabled.  Everything below this point derives report
	% fields from the already-computed control and cannot affect v.
	diagnostics = struct('sigma2', stats.sigma2);
	return;
end
if ~sequential_increment_qp && exist('qp_residuals', 'var')
	qp_active_constraint_count = sum(abs(qp_residuals) <= 1e-7);
end
hocbf_rows = constraint_types == "integral";
terminal_rows = constraint_types == "terminal";
clf_rows = startsWith(constraint_types, "anchor_clf");
obstacle_rows = constraint_types == "obstacle";
clf_first_rows = constraint_types == "anchor_clf_first";
clf_last_rows = constraint_types == "anchor_clf_last";
hocbf_row_active = any(hocbf_rows);
u_contribution_hocbf = sum( ...
	qp_u_row_contributions(hocbf_rows, :), 1)';
u_contribution_terminal = sum( ...
	qp_u_row_contributions(terminal_rows, :), 1)';
u_contribution_anchor_clf = sum( ...
	qp_u_row_contributions(clf_rows, :), 1)';
u_contribution_anchor_clf_first = sum( ...
	qp_u_row_contributions(clf_first_rows, :), 1)';
u_contribution_anchor_clf_last = sum( ...
	qp_u_row_contributions(clf_last_rows, :), 1)';
u_contribution_obstacle = sum( ...
	qp_u_row_contributions(obstacle_rows, :), 1)';
kkt_u_reconstruction_error = norm(u_col - sum( ...
	qp_u_row_contributions, 1)' - qp_u_equality_contribution);
if any(hocbf_rows)
	hocbf_slack = max(qp_slack(hocbf_rows));
end
if any(terminal_rows)
	terminal_slack = max(qp_slack(terminal_rows));
end
if any(clf_rows)
	if sequential_increment_qp
		clf_slack = sum(qp_slack(clf_rows));
	else
		clf_slack = max(qp_slack(clf_rows));
	end
end
if any(obstacle_rows)
	obstacle_slack = max(qp_slack(obstacle_rows));
end
% Compare the derivative represented in the QP with the derivative through
% the true point map. Skipped candidates retain NaN row slack.
n_obstacle_candidates = numel(obstacle_info.candidate_h_values);
obstacle_candidate_qp_control = nan(n_obstacle_candidates, 1);
obstacle_candidate_true_control = nan(n_obstacle_candidates, 1);
obstacle_candidate_qp_hdot = nan(n_obstacle_candidates, 1);
obstacle_candidate_true_hdot = nan(n_obstacle_candidates, 1);
obstacle_candidate_qp_residual = nan(n_obstacle_candidates, 1);
obstacle_candidate_true_residual = nan(n_obstacle_candidates, 1);
obstacle_candidate_slack = nan(n_obstacle_candidates, 1);
if n_obstacle_candidates > 0
	obstacle_candidate_qp_control = ...
		obstacle_info.candidate_grad_ctrl * u';
	obstacle_candidate_true_control = ...
		obstacle_info.candidate_grad_true * u';
	obstacle_candidate_qp_hdot = obstacle_info.candidate_drift + ...
		obstacle_candidate_qp_control;
	obstacle_candidate_true_hdot = obstacle_info.candidate_drift + ...
		obstacle_candidate_true_control;
	obstacle_candidate_qp_residual = -(obstacle_candidate_qp_hdot + ...
		obstacle_info.candidate_phi .* obstacle_info.candidate_h_values);
	obstacle_candidate_true_residual = -(obstacle_candidate_true_hdot + ...
		obstacle_info.candidate_phi .* obstacle_info.candidate_h_values);
	active_candidates = obstacle_info.active_candidate_indices;
	if ~isempty(active_candidates)
		obstacle_candidate_slack(active_candidates) = 0.0;
		if exist('obstacle_rows', 'var') && any(obstacle_rows)
			obstacle_candidate_slack(active_candidates) = ...
				qp_slack(obstacle_rows);
		end
	end
end
constraint_residual = sum(grad_x .* u, 2) + rho;
terminal_constraint_residual = sum(grad_x .* u, 2) - terminal_info.bound;
clf_raw_residual = clf_info.grad * u' - clf_info.bound;

% The sequential QP uses separate P1/P5 rows. Keep their diagnostics
% separate as well; the aggregate anchor CLF above is retained for
% compatibility with first/second-level plots.
endpoint_diag = repmat(struct( ...
	'v', nan, 'ptzf_bound', nan, 'qp_bound', nan, 'effective_bound', nan, ...
	'raw_residual', nan, 'relaxed_residual', nan, 'slack', nan, ...
	'grad_norm', nan, 'owner_grad_norm', nan, ...
	'row_active', false, 'row_present', false, ...
	'u_contribution_norm', 0.0), 2, 1);
if numel(endpoint_clf_infos) == 2
	endpoint_types = ["anchor_clf_first"; "anchor_clf_last"];
	for endpoint_idx = 1:2
		info_now = endpoint_clf_infos(endpoint_idx);
		endpoint_diag(endpoint_idx).v = info_now.v;
		endpoint_diag(endpoint_idx).ptzf_bound = info_now.ptzf_bound;
		endpoint_diag(endpoint_idx).qp_bound = info_now.bound;
		endpoint_diag(endpoint_idx).grad_norm = norm(info_now.grad);
		block_dim = struct_field_required(constraint_cfg, ...
			'increment_control_block_dim');
		block_count = struct_field_required(constraint_cfg, ...
			'increment_control_block_count');
		owner_block = 1 + (endpoint_idx - 1) * (block_count - 1);
		owner_cols = (owner_block - 1) * block_dim + (1:block_dim);
		endpoint_diag(endpoint_idx).owner_grad_norm = norm( ...
			info_now.grad(owner_cols));
		endpoint_diag(endpoint_idx).row_active = info_now.row_active;
		row_now = find(constraint_types == endpoint_types(endpoint_idx), 1);
		if ~isempty(row_now)
			endpoint_diag(endpoint_idx).row_present = true;
			endpoint_diag(endpoint_idx).effective_bound = ...
				sequential_result.effective_bounds(row_now);
			endpoint_diag(endpoint_idx).raw_residual = ...
				info_now.grad * u' - info_now.bound;
			endpoint_diag(endpoint_idx).slack = qp_slack(row_now);
			endpoint_diag(endpoint_idx).relaxed_residual = ...
				endpoint_diag(endpoint_idx).raw_residual - qp_slack(row_now);
			endpoint_diag(endpoint_idx).u_contribution_norm = norm( ...
				qp_u_row_contributions(row_now, :));
		end
	end
end

% raw residual 是原始约束违反量；relaxed residual = raw residual - slack。
% 带 slack 的 QP 约束满足时 relaxed residual 应该 <= 0。
clf_constraint_residual = clf_raw_residual - clf_slack;
psi2 = -constraint_residual;
diagnostics.active = norm(u) > 0.0;
diagnostics.control_enabled = true;
diagnostics.hocbf_enabled = hocbf_enabled;
diagnostics.hocbf_filter_active = hocbf_filter_active;
diagnostics.hocbf_row_active = hocbf_row_active;
diagnostics.x = stats.x(:)';
diagnostics.mu = mu;
diagnostics.v = v;
diagnostics.u = u';
diagnostics.u_ptclf_reference = reshape(u_ptclf_reference_col, 1, []);
diagnostics.u_after_ptcbf = reshape(u_after_ptcbf_col, 1, []);
diagnostics.u_ptcbf_correction = reshape( ...
	u_after_ptcbf_col - u_ptclf_reference_col, 1, []);
diagnostics.u_hocbf_correction = reshape( ...
	u_col - u_after_ptcbf_col, 1, []);
diagnostics.stage_ptclf_residual_at_reference = max_type_residual( ...
	stage_residuals_ptclf, constraint_types, "anchor_clf");
diagnostics.stage_ptclf_residual_after_ptcbf = max_type_residual( ...
	stage_residuals_ptcbf, constraint_types, "anchor_clf");
diagnostics.stage_ptclf_residual_after_hocbf = max_type_residual( ...
	stage_residuals_hocbf, constraint_types, "anchor_clf");
diagnostics.stage_obstacle_residual_after_ptcbf = max_type_residual( ...
	stage_residuals_ptcbf, constraint_types, "obstacle");
diagnostics.stage_obstacle_residual_after_hocbf = max_type_residual( ...
	stage_residuals_hocbf, constraint_types, "obstacle");
diagnostics.stage_hocbf_residual_after_ptcbf = max_type_residual( ...
	stage_residuals_ptcbf, constraint_types, "integral");
diagnostics.stage_hocbf_residual_after_hocbf = max_type_residual( ...
	stage_residuals_hocbf, constraint_types, "integral");
diagnostics.correction_norm = norm(u);
diagnostics.max_abs_u = max(abs(u(:)));
diagnostics.u_contribution_hocbf = u_contribution_hocbf;
diagnostics.u_contribution_terminal = u_contribution_terminal;
diagnostics.u_contribution_anchor_clf = u_contribution_anchor_clf;
diagnostics.u_contribution_anchor_clf_first = ...
	u_contribution_anchor_clf_first;
diagnostics.u_contribution_anchor_clf_last = ...
	u_contribution_anchor_clf_last;
diagnostics.u_contribution_obstacle = u_contribution_obstacle;
diagnostics.u_contribution_hocbf_norm = norm(u_contribution_hocbf);
diagnostics.u_contribution_terminal_norm = norm(u_contribution_terminal);
diagnostics.u_contribution_anchor_clf_norm = norm( ...
	u_contribution_anchor_clf);
diagnostics.u_contribution_obstacle_norm = norm(u_contribution_obstacle);
diagnostics.kkt_u_reconstruction_error = kkt_u_reconstruction_error;
diagnostics.grad_norm = norm(stats.sigma2_grad_x);
% HOCBF/terminal 共用同一个 grad_x 行，anchor_clf 另有自己的行；两者的
% 归一化前范数 r 用于把 slack 权重从"不归一化"标定换算到"全归一化"标定
% (w_new = w_old * r^2)，见 report_constraint_row_norms.m。
diagnostics.anchor_clf_grad_norm = norm(clf_info.grad);
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
diagnostics.anchor_clf_first_v = endpoint_diag(1).v;
diagnostics.anchor_clf_first_ptzf_bound = endpoint_diag(1).ptzf_bound;
diagnostics.anchor_clf_first_bound = endpoint_diag(1).qp_bound;
diagnostics.anchor_clf_first_effective_bound = ...
	endpoint_diag(1).effective_bound;
diagnostics.anchor_clf_first_raw_residual = endpoint_diag(1).raw_residual;
diagnostics.anchor_clf_first_residual = endpoint_diag(1).relaxed_residual;
diagnostics.anchor_clf_first_slack = endpoint_diag(1).slack;
diagnostics.anchor_clf_first_grad_norm = endpoint_diag(1).grad_norm;
diagnostics.anchor_clf_first_owner_grad_norm = ...
	endpoint_diag(1).owner_grad_norm;
diagnostics.anchor_clf_first_row_active = endpoint_diag(1).row_active;
diagnostics.anchor_clf_first_row_present = endpoint_diag(1).row_present;
diagnostics.anchor_clf_first_u_contribution_norm = ...
	endpoint_diag(1).u_contribution_norm;
diagnostics.anchor_clf_last_v = endpoint_diag(2).v;
diagnostics.anchor_clf_last_ptzf_bound = endpoint_diag(2).ptzf_bound;
diagnostics.anchor_clf_last_bound = endpoint_diag(2).qp_bound;
diagnostics.anchor_clf_last_effective_bound = ...
	endpoint_diag(2).effective_bound;
diagnostics.anchor_clf_last_raw_residual = endpoint_diag(2).raw_residual;
diagnostics.anchor_clf_last_residual = endpoint_diag(2).relaxed_residual;
diagnostics.anchor_clf_last_slack = endpoint_diag(2).slack;
diagnostics.anchor_clf_last_grad_norm = endpoint_diag(2).grad_norm;
diagnostics.anchor_clf_last_owner_grad_norm = ...
	endpoint_diag(2).owner_grad_norm;
diagnostics.anchor_clf_last_row_active = endpoint_diag(2).row_active;
diagnostics.anchor_clf_last_row_present = endpoint_diag(2).row_present;
diagnostics.anchor_clf_last_u_contribution_norm = ...
	endpoint_diag(2).u_contribution_norm;
diagnostics.obstacle_enabled = obstacle_enabled;
diagnostics.obstacle_h_min = obstacle_info.h_min;
diagnostics.obstacle_n_rows = obstacle_info.n_rows;
diagnostics.obstacle_slack = obstacle_slack;
diagnostics.obstacle_max_residual_without_u = obstacle_info.max_residual_without_u;
diagnostics.obstacle_candidate_count = n_obstacle_candidates;
diagnostics.obstacle_skipped_count = sum( ...
	~obstacle_info.candidate_row_active);
diagnostics.obstacle_unsafe_count = sum( ...
	obstacle_info.candidate_h_values < 0);
diagnostics.obstacle_unsafe_skipped_count = sum( ...
	obstacle_info.candidate_h_values < 0 & ...
	~obstacle_info.candidate_row_active);
diagnostics.obstacle_min_point_idx = nan;
diagnostics.obstacle_min_obstacle_idx = nan;
diagnostics.obstacle_min_grad_true_norm = nan;
diagnostics.obstacle_min_grad_ctrl_norm = nan;
diagnostics.obstacle_min_row_active = false;
diagnostics.obstacle_min_phi = nan;
diagnostics.obstacle_min_drift = nan;
diagnostics.obstacle_min_qp_control = nan;
diagnostics.obstacle_min_true_control = nan;
diagnostics.obstacle_min_qp_hdot = nan;
diagnostics.obstacle_min_true_hdot = nan;
diagnostics.obstacle_min_qp_residual = nan;
diagnostics.obstacle_min_true_residual = nan;
diagnostics.obstacle_min_row_slack = nan;
if n_obstacle_candidates > 0
	[~, min_obstacle_candidate_idx] = min( ...
		obstacle_info.candidate_h_values);
	diagnostics.obstacle_min_point_idx = ...
		obstacle_info.candidate_point_indices(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_obstacle_idx = ...
		obstacle_info.candidate_obstacle_indices(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_grad_true_norm = ...
		obstacle_info.candidate_grad_true_norm(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_grad_ctrl_norm = ...
		obstacle_info.candidate_grad_ctrl_norm(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_row_active = ...
		obstacle_info.candidate_row_active(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_phi = ...
		obstacle_info.candidate_phi(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_drift = ...
		obstacle_info.candidate_drift(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_qp_control = ...
		obstacle_candidate_qp_control(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_true_control = ...
		obstacle_candidate_true_control(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_qp_hdot = ...
		obstacle_candidate_qp_hdot(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_true_hdot = ...
		obstacle_candidate_true_hdot(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_qp_residual = ...
		obstacle_candidate_qp_residual(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_true_residual = ...
		obstacle_candidate_true_residual(min_obstacle_candidate_idx);
	diagnostics.obstacle_min_row_slack = ...
		obstacle_candidate_slack(min_obstacle_candidate_idx);
end
diagnostics.obstacle_active = obstacle_enabled && ...
	obstacle_info.max_residual_without_u > 0.0;
% This flag records whether the obstacle PTCBF stage is enabled at this
% time, independently of whether it happens to need a nonzero correction.
diagnostics.obstacle_filter_active = obstacle_enabled;
diagnostics.qp_slack = max([hocbf_slack, terminal_slack, clf_slack, obstacle_slack]);
diagnostics.qp_slack_vector = qp_slack;
diagnostics.anchor_clf_ptzf_bound = clf_info.ptzf_bound;
diagnostics.anchor_clf_ptzf_bound_dot = clf_info.ptzf_bound_dot;
diagnostics.anchor_clf_ptzf_cg = clf_info.ptzf_cg;
diagnostics.anchor_clf_cpt = clf_info.cpt;
diagnostics.qp_exitflag = qp_exitflag;
diagnostics.qp_iterations = qp_iterations;
diagnostics.qp_solve_seconds = qp_solve_seconds;
diagnostics.qp_active_constraint_count = qp_active_constraint_count;
diagnostics.sigma2 = stats.sigma2;
diagnostics.min_grad_norm_sq = min(grad_norm_sq);
end

function value = max_type_residual(residuals, constraint_types, type_name)
if type_name == "anchor_clf"
	rows = startsWith(constraint_types, type_name);
else
	rows = constraint_types == type_name;
end
values = residuals(rows);
values = values(isfinite(values));
if isempty(values)
	value = nan;
else
	value = max(values);
end
end
