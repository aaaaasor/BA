function w = minnorm_halfspaces(G, c, backend)
%MINNORM_HALFSPACES  min ||w||^2  subject to  G*w >= c, solved exactly.
%
% Enumerates active sets, so it is exact for the small systems here (at most
% five rows in two dimensions).  Rows that are already satisfied cost nothing:
% if every c_i <= 0 the answer is w = 0, and such a row only enters the active
% set when the correction would otherwise push through it.
%
% When no subset is feasible -- opposing gradients whose half-planes have empty
% intersection -- it returns pinv(G)*max(c,0), the least-violation solution,
% which satisfies nothing in particular.  Callers that need to know should test
% G*w >= c themselves and count the failures.
%
% Extracted from safeflow_nn_rollout so pcfm_nn_rollout can use the same
% solver; the body is unchanged.
% backend: 'closed' (default) uses the exact active-set enumeration below.
% 'quadprog' solves the same program with MATLAB's generic QP solver instead.
% The two agree to solver tolerance; only the cost differs.  The infeasible
% fallback (pinv(G)*max(c,0)) is shared so callers see identical semantics.
if nargin < 3 || isempty(backend), backend = 'closed'; end
if strcmp(backend, 'quadprog')
    w = minnorm_halfspaces_qp(G, c);
    return;
end
if all(c <= 0), w = zeros(size(G,2),1); return; end
N = size(G,1);  best_w = [];  best_n = inf;
for mask = 1:(2^N - 1)
    S  = bitget(mask, 1:N) > 0;
    Gs = G(S,:);  cs = c(S);
    Gg = Gs*Gs.';
    if rcond(Gg) < 1e-12
        lam = pinv(Gg) * cs;
    else
        lam = Gg \ cs;
    end
    wc = Gs.' * lam;
    if any(G*wc < c - 1e-9), continue; end     % 必须对全部约束可行
    nw = wc.'*wc;
    if nw < best_n, best_n = nw; best_w = wc; end
end
if isempty(best_w)
    % 退化：所有子集都不可行（约束互相矛盾），取违反最小的最小二乘解
    best_w = pinv(G) * max(c, 0);
end
w = best_w;
end

% =====================================================================

% =====================================================================
function w = minnorm_halfspaces_qp(G, c)
%MINNORM_HALFSPACES_QP  min ||w||^2 s.t. G*w >= c, solved by quadprog.
persistent opts
if isempty(opts)
    opts = optimoptions('quadprog', 'Display', 'off', ...
        'Algorithm', 'interior-point-convex', ...
        'OptimalityTolerance', 1e-12, 'ConstraintTolerance', 1e-12);
end
n = size(G, 2);
if all(c <= 0), w = zeros(n, 1); return; end
[w, ~, exitflag] = quadprog(2 * eye(n), zeros(n, 1), -G, -c, [], [], ...
    [], [], [], opts);
if exitflag <= 0 || isempty(w)
    % Same degenerate branch as the enumeration: no feasible subset exists,
    % so return the least-violation least-squares point.
    w = pinv(G) * max(c, 0);
end
w = w(:);
end
