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
% row_active=false 时该行从 QP 中剔除(梯度退化，如已收敛到目标 e≈0)，
% 而不是让 u 去处理一个没有一阶影响力的行——留着只会在归一化时被除大，
% 或在硬约束下导致假性不可行。
clf_info.row_active = true;
if ~clf_info.enabled
	return;
end
use_linear_map = isfield(constraint_cfg, 'anchor_clf_matrix');
if ~isfield(constraint_cfg, 'anchor_clf_target') || ...
		(~use_linear_map && ~isfield(constraint_cfg, 'anchor_clf_indices'))
	error(['Anchor CLF requires anchor_clf_target and either ', ...
		'anchor_clf_indices or anchor_clf_matrix.']);
end
target = constraint_cfg.anchor_clf_target(:);
x_now = stats.x(:);
mu = stats.mu(:);
if use_linear_map
	% 线性映射形式（第三层增量表示）: y = M*x + offset 是物理端点，
	% e = y - target, g = e'*e, grad_x g = 2*M'*e,
	% g_dot = 2*e'*M*(mu + u)。
	M = constraint_cfg.anchor_clf_matrix;
	map_offset = constraint_cfg.anchor_clf_offset(:);
	e = M * x_now + map_offset - target;
	clf_info.v = sum(e .^ 2);
	clf_info.grad = 2.0 * (M' * e)';
	clf_drift = 2.0 * sum(e .* (M * mu));
else
	indices = constraint_cfg.anchor_clf_indices(:)';
	if numel(target) ~= numel(indices)
		error('Anchor CLF target size must match anchor_clf_indices.');
	end
	% Anchor PTCLF 只作用在指定的 anchor 坐标 indices 上。
	% e = x_anchor - x_ref, g(x) = e' * e。
	% 参考 anchor 固定，因此 g_dot = 2*e'*(mu_anchor + u_anchor)。
	e = x_now(indices) - target;
	clf_info.v = sum(e .^ 2);
	clf_info.grad(indices) = 2.0 * e';
	clf_drift = 2.0 * sum(e .* mu(indices));
end

% 梯度退化检测: 段已收敛到目标(e≈0)时，grad 趋近于 0 但不精确为 0。
% u 对这种行没有一阶影响力，直接跳过(导师方案: grad 太小时让 u=0)。
grad_tol = struct_field_default(constraint_cfg, 'grad_tol', 1e-6);
if norm(clf_info.grad) < grad_tol
	clf_info.row_active = false;
end

% UniConFlow-style PTCLF:
% g_dot <= gamma(gbar(t) - g) + gbar_dot(t)。
% 这里取线性 class-K: gamma(r) = cpt * r。
% gbar(t) = gbar0 * exp(-c_g * t/(1-t))，公式本身不变（一次方分母）。
% shape_dot 用精确求导结果 (1-t)^-2（平方分母，跟导师文档一致）。
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
	% shape = t_eff / (1-t_eff)，gbar(t) 公式本身不变。
	shape = t_eff ./ remaining_tau;
	shape_dot = remaining_tau .^ (-2.0);
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
