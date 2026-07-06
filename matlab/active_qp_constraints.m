function [A_qp, b_qp, constraint_types] = active_qp_constraints(grad_x, ...
	integral_bound, terminal_bound, clf_info, hocbf_enabled)
A_qp = [];
b_qp = [];
constraint_types = strings(0, 1);

% 第 1 行: integral HOCBF。
% grad_beta' * u <= integral_bound。
if hocbf_enabled
	A_qp = [A_qp; grad_x];
	b_qp = [b_qp; integral_bound];
	constraint_types(end + 1, 1) = "integral";
end

% 第 2 行: terminal PTCBF。如果 terminal 未启用，bound 为 inf，则不加入。
% terminal PTCBF 和 HOCBF 使用同一个 grad_beta，但右端 bound 不同。
if isfinite(terminal_bound)
	A_qp = [A_qp; grad_x];
	b_qp = [b_qp; terminal_bound];
	constraint_types(end + 1, 1) = "terminal";
end

% 第 3 行: anchor PTCLF。
% grad_g' * u <= anchor_clf_bound，其中 grad_g 只在 anchor 坐标上非零。
if clf_info.enabled
	A_qp = [A_qp; clf_info.grad];
	b_qp = [b_qp; clf_info.bound];
	constraint_types(end + 1, 1) = "anchor_clf";
end
end
