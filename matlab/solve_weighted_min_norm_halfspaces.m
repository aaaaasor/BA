function [u, exitflag, slack, residuals, iterations, solve_seconds, ...
	row_contributions] = solve_weighted_min_norm_halfspaces(A, b, ...
	control_weight)
% Solve min 0.5*||W*u||^2 subject to A*u <= b by active-set enumeration.
% This is intended for the two hard endpoint PTCLF rows. With q=W*u, the
% Hessian is the identity and every candidate has a closed-form solution.

solve_timer = tic;
n_rows = size(A, 1);
n_u = size(A, 2);
b = b(:);
control_weight = control_weight(:);
if numel(b) ~= n_rows || numel(control_weight) ~= n_u
	error('Weighted minimum-norm halfspace dimensions are inconsistent.');
end
if any(control_weight <= 0) || any(~isfinite(control_weight))
	error('control_weight must be finite and strictly positive.');
end
if n_rows > 12
	error(['Closed-form active-set enumeration is limited to 12 rows; ', ...
		'received %d.'], n_rows);
end

% Row scaling preserves each halfspace and improves the small Gram solves.
row_scale = sqrt(sum(A .^ 2, 2));
row_scale(row_scale == 0) = 1;
A_scaled = A ./ row_scale;
b_scaled = b ./ row_scale;
inv_weight = 1 ./ control_weight;
C = A_scaled .* inv_weight';

feas_tol = 1e-9 * max(1, norm(b_scaled, inf));
dual_tol = 1e-9;
best_objective = inf;
best_q = [];
best_lambda = zeros(n_rows, 1);
iterations = 0;

% The unconstrained minimizer is the first candidate.
q_zero = zeros(n_u, 1);
if all(C * q_zero <= b_scaled + feas_tol)
	best_q = q_zero;
	best_objective = 0;
end

% At the optimum, a subset of the endpoint rows is active. For each
% subset I, q=-C_I'*lambda and lambda=-(C_I*C_I')^dagger*b_I.
for mask = 1:(2 ^ n_rows - 1)
	iterations = iterations + 1;
	active = find(bitget(mask, 1:n_rows));
	C_active = C(active, :);
	b_active = b_scaled(active);
	gram = C_active * C_active';
	lambda_active = -pinv(gram) * b_active;
	q_candidate = -C_active' * lambda_active;
	if any(lambda_active < -dual_tol)
		continue;
	end
	if norm(C_active * q_candidate - b_active, inf) > 10 * feas_tol
		continue;
	end
	if any(C * q_candidate > b_scaled + 10 * feas_tol)
		continue;
	end
	objective = 0.5 * (q_candidate' * q_candidate);
	if objective < best_objective
		best_objective = objective;
		best_q = q_candidate;
		best_lambda = zeros(n_rows, 1);
		best_lambda(active) = lambda_active;
	end
end

if isempty(best_q)
	error('Hard endpoint PTCLF halfspaces are infeasible.');
end

u = inv_weight .* best_q;
exitflag = 1;
slack = zeros(n_rows, 1);
residuals = A * u - b;
row_contributions = zeros(n_rows, n_u);
for row_idx = 1:n_rows
	q_part = -best_lambda(row_idx) * C(row_idx, :)';
	row_contributions(row_idx, :) = (inv_weight .* q_part)';
end
solve_seconds = toc(solve_timer);
end
