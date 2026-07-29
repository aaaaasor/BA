function terminal_info = terminal_variance_ptcbf(stats, constraint_cfg, ...
	t, beta_drift)
ptcbf_enabled = struct_field_default(constraint_cfg, ...
	'ptcbf_enabled', false);
ptcbf_end_time = struct_field_default(constraint_cfg, ...
	'terminal_variance_ptcbf_end_time', inf);
terminal_info.enabled = ptcbf_enabled && t < ptcbf_end_time;
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

% Terminal PTCBF 控制末端预测方差:
% 目标是 beta(t) <= beta_final。
% 用 prescribed-time envelope hbar_T(t) 允许前期较松、末端收紧。
beta_final = constraint_cfg.terminal_variance_beta_final;
if ~isfield(constraint_cfg, 'terminal_variance_ptzf_hbar0')
	error(['Terminal variance PTCBF initial bound must be set per ', ...
		'sample by rk4_rollout.']);
end
hbar0 = constraint_cfg.terminal_variance_ptzf_hbar0;
gamma = constraint_cfg.terminal_variance_ptzf_gamma;
alpha_terminal = constraint_cfg.terminal_variance_alpha;

% blow-up time transformation: shape = t_eff/(1-t_eff)。
% t_eff -> 1 时 shape -> inf，因此 hbar_T -> 0。
% rollout 实际只能跑到 rollout_t_max (<1)，用 ptzf_time_shift 把这个
% 实际终点映射到 t_eff=1，使 hbar_T 在 rollout 真正结束时精确收敛到 0，
% 而不是渐近地趋近于 0。
time_shift = struct_field_default(constraint_cfg, 'ptzf_time_shift', 0.0);
t_eff = t + time_shift;
remaining_tau = max(1.0 - t_eff, eps);
shape = t_eff ./ remaining_tau;
shape_dot = remaining_tau .^ (-1.0) + t_eff .* remaining_tau .^ (-2.0);
ptzf_bound = hbar0 .* exp(-gamma .* shape);
ptzf_bound_dot = -gamma .* shape_dot .* ptzf_bound;

% h_terminal = hbar_T - (beta - beta_final)。
% PTCBF 条件: h_dot + alpha_terminal*h >= 0。
% 展开得到 QP 线性约束:
% grad_beta' * u <= -beta_drift + alpha_terminal*h + hbar_T_dot。
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
