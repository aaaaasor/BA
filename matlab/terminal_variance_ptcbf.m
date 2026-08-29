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
terminal_info.per_dimension = false;
terminal_info.rows_A = zeros(0, 0);
terminal_info.rows_bound = zeros(0, 1);
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
time_varying_alpha = struct_field_default(constraint_cfg, ...
	'terminal_variance_time_varying_alpha', false);
alpha_power = struct_field_default(constraint_cfg, ...
	'terminal_variance_time_varying_alpha_power', 2.0);

% Map physical time to an independent theoretical PTZF clock.  Constraint
% activation/slack times remain physical and can be tuned separately.
terminal_time = struct_field_default(constraint_cfg, ...
	'terminal_variance_ptzf_terminal_time', ...
	struct_field_default(constraint_cfg, 'rollout_t_max', 1.0));
time_shift = struct_field_default(constraint_cfg, ...
	'terminal_variance_ptzf_time_shift', 0.0);
if ~(isscalar(terminal_time) && isfinite(terminal_time) && terminal_time > 0)
	error('terminal_variance_ptzf_terminal_time must be finite and positive.');
end
if ~(isscalar(time_shift) && isfinite(time_shift) && time_shift >= 0)
	error('terminal_variance_ptzf_time_shift must be finite and nonnegative.');
end
% Only the variance PTZF clock is shifted.  Obstacle/boundary PTCBFs are
% assembled elsewhere from the original physical time t.
t_eff = (t + time_shift) ./ terminal_time;
if t_eff >= 1.0
	% Continuous terminal extension: hbar(1)=0 and hbar_dot(1)=0.
	ptzf_bound = 0.0;
	ptzf_bound_dot = 0.0;
	% The prescribed-time gain is singular only on the open interval.  At
	% the exact terminal RK4 stage retain the finite endpoint condition;
	% the preceding interior stages have already applied the blow-up gain.
	alpha_multiplier = 1.0;
else
	remaining_tau = 1.0 - t_eff;
	shape = t_eff ./ remaining_tau;
	% d/dt [tau/(1-tau)] = (1/T)/(1-tau)^2.
	shape_dot = (1.0 ./ terminal_time) .* remaining_tau .^ (-2.0);
	ptzf_bound = hbar0 .* exp(-gamma .* shape);
	ptzf_bound_dot = -gamma .* shape_dot .* ptzf_bound;
	alpha_multiplier = remaining_tau .^ (-alpha_power);
end
if ~time_varying_alpha
	alpha_multiplier = 1.0;
end
alpha_gain = alpha_terminal .* alpha_multiplier;

% h_terminal = hbar_T - (beta - beta_final)。
% PTCBF 条件: h_dot + alpha_terminal*h >= 0。
% 展开得到 QP 线性约束:
% grad_beta' * u <= -beta_drift + alpha_terminal*h + hbar_T_dot。
per_dimension = struct_field_default(constraint_cfg, ...
	'terminal_variance_per_dimension', false) && ...
	isfield(stats, 'sigma2_grad_x_components');
if per_dimension
	% One PTCBF row per GP output.  Summing sigma_i^2 into a single row
	% lets the per-output gradients partially cancel (measured: the summed
	% gradient is ~1.6x shorter than the sum of the individual norms near
	% t=1), which costs the QP control authority exactly where the terminal
	% constraint is hardest to satisfy.
	sigma2_i = stats.sigma2_components(:);
	n_out = numel(sigma2_i);
	beta_final_i = beta_final_per_dimension(constraint_cfg, beta_final, n_out);
	hbar0_i = hbar0(:);
	if isscalar(hbar0_i)
		hbar0_i = hbar0_i .* (beta_final_i ./ sum(beta_final_i));
	end
	grad_i = stats.sigma2_grad_x_components;
	drift_i = stats.sigma2_t_components(:) + grad_i * stats.mu(:);
	ptzf_i = hbar0_i .* (ptzf_bound ./ max(hbar0, eps));
	ptzf_dot_i = hbar0_i .* (ptzf_bound_dot ./ max(hbar0, eps));
	h_i = ptzf_i - (sigma2_i - beta_final_i);
	terminal_info.per_dimension = true;
	terminal_info.rows_A = grad_i;
	terminal_info.rows_bound = -drift_i + alpha_gain .* h_i + ptzf_dot_i;
	terminal_info.beta_final_components = beta_final_i;
	terminal_info.h_components = h_i;
end
terminal_inequality_h = stats.sigma2 - beta_final;
beta_cap = beta_final + ptzf_bound;
beta_cap_dot = ptzf_bound_dot;
h_terminal = ptzf_bound - terminal_inequality_h;
alpha_h = alpha_gain .* h_terminal;
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

function beta_final_i = beta_final_per_dimension(constraint_cfg, ...
	beta_final_total, n_out)
% Split the scalar budget across outputs.  Default: proportional to each
% output's prior variance, so dimensions the GP is inherently less certain
% about are not given an unreachably tight share.
beta_final_i = struct_field_default(constraint_cfg, ...
	'terminal_variance_beta_final_components', []);
if ~isempty(beta_final_i)
	beta_final_i = beta_final_i(:);
	if numel(beta_final_i) ~= n_out
		error(['terminal_variance_beta_final_components must have one ', ...
			'entry per GP output.']);
	end
	return;
end
signal_std = struct_field_default(constraint_cfg, 'signal_std_vec', []);
if numel(signal_std) == n_out
	w = signal_std(:) .^ 2;
	beta_final_i = beta_final_total .* w ./ sum(w);
else
	beta_final_i = repmat(beta_final_total ./ n_out, n_out, 1);
end
end
