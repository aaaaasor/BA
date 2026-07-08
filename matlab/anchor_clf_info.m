function clf_info = anchor_clf_info(stats, constraint_cfg, t)
clf_info.enabled = struct_field_default(constraint_cfg, ...
	'ptclf_enabled', false);
clf_info.v = nan;
clf_info.grad = zeros(1, numel(stats.mu));
clf_info.bound = inf;
clf_info.residual_without_u = -inf;
clf_info.ptzf_bound = nan;
clf_info.ptzf_bound_dot = nan;
clf_info.ptzf_cg = nan;
clf_info.cpt = nan;
if ~clf_info.enabled
	return;
end
if ~isfield(constraint_cfg, 'anchor_clf_target') || ...
		~isfield(constraint_cfg, 'anchor_clf_indices')
	error('Anchor CLF requires anchor_clf_target and anchor_clf_indices.');
end
indices = constraint_cfg.anchor_clf_indices(:)';
target = constraint_cfg.anchor_clf_target(:);
if numel(target) ~= numel(indices)
	error('Anchor CLF target size must match anchor_clf_indices.');
end

% Anchor PTCLF 只作用在指定的 anchor 坐标 indices 上。
% e = x_anchor - x_ref, g(x) = e' * e。
% 参考 anchor 固定，因此 g_dot = 2*e'*(mu_anchor + u_anchor)。
x_now = stats.x(:);
mu = stats.mu(:);
e = x_now(indices) - target;
clf_info.v = sum(e .^ 2);
clf_info.grad(indices) = 2.0 * e';
clf_drift = 2.0 * sum(e .* mu(indices));

% UniConFlow-style PTCLF:
% g_dot <= gamma(gbar(t) - g) + gbar_dot(t)。
% 这里取线性 class-K: gamma(r) = cpt * r。
% gbar(t) = gbar0 * exp(-c_g * t/(1-t)^3) 是 blow-up time transformation
% 的三次分母版本：分母比原始版本(1-t)收得更快，t=T 处的奇点更"硬"，
% 临近终点时对瞬时反应速度的要求会比一次方分母版本更极端。
% 若 anchor_clf_ptzf_enabled = false，退化为 gbar==0 的普通 CLF：
% g_dot <= -cpt*g，即标准指数收敛条件，不再需要 cg/margin。
ptzf_enabled = struct_field_default(constraint_cfg, ...
	'anchor_clf_ptzf_enabled', true);
cpt = struct_field_default(constraint_cfg, 'anchor_clf_cpt', 1.0);
if ptzf_enabled
	ptzf_cg = struct_field_default(constraint_cfg, ...
		'anchor_clf_ptzf_cg', 0.1);
	gbar0 = struct_field_default(constraint_cfg, ...
		'anchor_clf_ptzf_initial_bound', max(clf_info.v, 0.0));
	% rollout 实际只能跑到 rollout_t_max (<1)，用 ptzf_time_shift 把这个
	% 实际终点映射到 t_eff=1，使 gbar 在 rollout 真正结束时精确收敛到 0。
	time_shift = struct_field_default(constraint_cfg, 'ptzf_time_shift', 0.0);
	t_eff = t + time_shift;
	remaining_tau = max(1.0 - t_eff, eps);
	% shape = t_eff / (1-t_eff)^3，分母为三次方的 blow-up 函数。
	shape = t_eff .* remaining_tau .^ (-3.0);
	shape_dot = remaining_tau .^ (-3.0) + 3.0 .* t_eff .* remaining_tau .^ (-4.0);
	ptzf_bound = gbar0 .* exp(-ptzf_cg .* shape);
	ptzf_bound_dot = -ptzf_cg .* shape_dot .* ptzf_bound;
else
	ptzf_cg = 0.0;
	ptzf_bound = 0.0;
	ptzf_bound_dot = 0.0;
end

% QP 中 PTCLF 行:
% grad_g' * u <= cpt*(gbar - g) + gbar_dot - 2*e'*mu_anchor。
clf_info.bound = cpt * (ptzf_bound - clf_info.v) + ...
	ptzf_bound_dot - clf_drift;
clf_info.residual_without_u = -clf_info.bound;
clf_info.ptzf_bound = ptzf_bound;
clf_info.ptzf_bound_dot = ptzf_bound_dot;
clf_info.ptzf_cg = ptzf_cg;
clf_info.cpt = cpt;
end
