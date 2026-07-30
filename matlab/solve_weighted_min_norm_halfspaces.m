function [u, exitflag, slack, residuals, iterations, solve_seconds, ...
	row_contributions, equality_contribution] = ...
	solve_weighted_min_norm_halfspaces(A, b, control_weight, Aeq, beq)
% Solve
%   min 0.5*||W*u||^2  s.t. A*u <= b, Aeq*u = beq
% by enumerating the small inequality active set.  For each active set the
% KKT system has a closed-form minimum-norm solution in q=W*u.  This avoids
% sending the hard PTCLF/PTCBF/HOCBF filters through quadprog merely because
% several halfspaces (and, in the snap region, endpoint equalities) coexist.

solve_timer = tic;
n_rows = size(A, 1);
n_u = size(A, 2);
b = b(:);
control_weight = control_weight(:);
if nargin < 4 || isempty(Aeq)
	Aeq = zeros(0, n_u);
end
if nargin < 5 || isempty(beq)
	beq = zeros(size(Aeq, 1), 1);
end
beq = beq(:);
if numel(b) ~= n_rows || numel(control_weight) ~= n_u
	error('Weighted minimum-norm halfspace dimensions are inconsistent.');
end
if size(Aeq, 2) ~= n_u || numel(beq) ~= size(Aeq, 1)
	error('Weighted minimum-norm equality dimensions are inconsistent.');
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
eq_scale = sqrt(sum(Aeq .^ 2, 2));
eq_scale(eq_scale == 0) = 1;
Aeq_scaled = Aeq ./ eq_scale;
beq_scaled = beq ./ eq_scale;
inv_weight = 1 ./ control_weight;
C = A_scaled .* inv_weight';
E = Aeq_scaled .* inv_weight';

feas_tol = 1e-9 * max(1, norm(b_scaled, inf));
dual_tol = 1e-9;
best_objective = inf;
best_q = [];
best_lambda = zeros(n_rows, 1);
best_nu = zeros(size(Aeq, 1), 1);
iterations = 0;

% At the optimum, a subset I of the inequalities is active.  Equality
% multipliers are free, while active inequality multipliers must be >= 0.
% Include mask=0 so equality-only and zero-control solutions are handled by
% exactly the same KKT calculation.
for mask = 0:(2 ^ n_rows - 1)
	iterations = iterations + 1;
	active = find(bitget(mask, 1:n_rows));
	C_active = C(active, :);
	b_active = b_scaled(active);
	G = [E; C_active];
	rhs = [beq_scaled; b_active];
	if isempty(G)
		multipliers = zeros(0, 1);
		q_candidate = zeros(n_u, 1);
	else
		gram = G * G';
		multipliers = -pinv(gram) * rhs;
		q_candidate = -G' * multipliers;
	end
	n_eq = size(E, 1);
	nu_candidate = multipliers(1:n_eq);
	lambda_active = multipliers(n_eq + 1:end);
	if any(lambda_active < -dual_tol)
		continue;
	end
	if ~isempty(E) && ...
			norm(E * q_candidate - beq_scaled, inf) > 10 * feas_tol
		continue;
	end
	if ~isempty(C_active) && ...
			norm(C_active * q_candidate - b_active, inf) > 10 * feas_tol
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
		best_nu = nu_candidate;
	end
end

if isempty(best_q)
	error('Hard weighted minimum-norm constraints are infeasible.');
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
% With no endpoint equality MATLAB can preserve an empty multiplier as a
% 1-by-0 row in some active-set branches.  Avoid multiplying that shape by
% the n_u-by-0 empty E' matrix; the equality contribution is identically 0.
if isempty(E)
	equality_contribution = zeros(n_u, 1);
else
	equality_contribution = ...
		inv_weight .* (-E' * best_nu(:));
end
solve_seconds = toc(solve_timer);
end
