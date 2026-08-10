function [A_qp, b_qp, constraint_types] = active_qp_constraints(grad_x, ...
	integral_bound, terminal_bound, clf_info, hocbf_enabled, obstacle_info, ...
	grad_x_active, boundary_info)
A_qp = [];
b_qp = [];
constraint_types = strings(0, 1);
if nargin < 7
	grad_x_active = true;
end

% 第 1 行: integral HOCBF。
% grad_beta' * u <= integral_bound。grad_x_active=false(梯度退化，如恰好
% 落在 LoG-GP 局部专家拼接边界)时跳过——u 对这行没有一阶影响力，硬留在
% QP 里可能导致假性不可行(导师方案: grad 太小时让 u=0)。
if hocbf_enabled && grad_x_active
	A_qp = [A_qp; grad_x];
	b_qp = [b_qp; integral_bound];
	constraint_types(end + 1, 1) = "integral";
end

% 第 2 行: terminal PTCBF。如果 terminal 未启用，bound 为 inf，则不加入。
% terminal PTCBF 和 HOCBF 使用同一个 grad_beta，但右端 bound 不同。
if isfinite(terminal_bound) && grad_x_active
	A_qp = [A_qp; grad_x];
	b_qp = [b_qp; terminal_bound];
	constraint_types(end + 1, 1) = "terminal";
end

% 第 3 行: anchor PTCLF。
% grad_g' * u <= anchor_clf_bound，其中 grad_g 只在 anchor 坐标上非零。
% row_active=false(梯度退化，如已收敛到目标 e≈0)时跳过，不加入 QP。
if clf_info.enabled && struct_field_default(clf_info, 'row_active', true)
	A_qp = [A_qp; clf_info.grad];
	b_qp = [b_qp; clf_info.bound];
	constraint_types(end + 1, 1) = "anchor_clf";
end

% 第 4 类: 障碍 CBF（可多行，每个作用点每个障碍一行）。
% A_obs * u <= bounds_obs，A/bounds 已在 obstacle_cbf_info 里算好。
if nargin >= 6 && ~isempty(obstacle_info) && obstacle_info.enabled && ...
		~isempty(obstacle_info.A)
	n_obstacle_rows = size(obstacle_info.A, 1);
	A_qp = [A_qp; obstacle_info.A];
	b_qp = [b_qp; obstacle_info.bounds];
	constraint_types(end + (1:n_obstacle_rows), 1) = "obstacle";
end

% Track left/right boundary CBF. Each selected trajectory point contributes
% one row for the left spline and one for the right spline.
if nargin >= 8 && ~isempty(boundary_info) && boundary_info.enabled && ...
		~isempty(boundary_info.A)
	n_boundary_rows = size(boundary_info.A, 1);
	A_qp = [A_qp; boundary_info.A];
	b_qp = [b_qp; boundary_info.bounds];
	constraint_types(end + (1:n_boundary_rows), 1) = "boundary";
end
end
