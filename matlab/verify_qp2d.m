function verify_qp2d(n_trial, seed)
%VERIFY_QP2D 用 quadprog 抽样验证 safeflow_nn_rollout 里的 qp2d 精确解。
%
%   验证的是式 30 的原始形式（含显式 slack 变量）：
%     min_{u,d}  |u|^2 + sum_j d_j^2
%     s.t.       a_j + b_j'u + d_j >= 0,   d_j >= 0
%   变量 x = [u; d] in R^{2+N}，H = 2*I，无线性项。
%   qp2d 解的是消去 d 后的等价分片二次问题，两者最优值应一致。
%
%   verify_qp2d(2000)

if nargin < 1 || isempty(n_trial), n_trial = 2000; end
if nargin < 2 || isempty(seed),    seed = 7; end
rng(seed);

opt = optimoptions('quadprog', 'Display', 'off', ...
    'Algorithm', 'interior-point-convex', ...
    'OptimalityTolerance', 1e-12, 'ConstraintTolerance', 1e-12);

max_dJ = 0; max_du = 0; abs_du = 0; abs_dJ = 0;
n_fail = 0;  t_ours = 0; t_qp = 0;  u_scale = 0;
for k = 1:n_trial
    N  = randi([1 5]);
    % 覆盖多种尺度：既有几乎不违反的，也有严重违反的
    sc = 10^(randi([-2 1]));
    a  = sc * (randn(N,1) - 0.5*rand());
    B  = sc * randn(N,2);
    if rand() < 0.15, B(1,:) = B(min(2,N),:) * (1+1e-9); end   % 近平行梯度

    tic; [u1, ~] = qp2d_ref(a, B);              t_ours = t_ours + toc;
    J1 = u1.'*u1 + sum(min(0, a + B*u1).^2);

    H = 2*eye(2+N);
    A = [-B, -eye(N)];  b = a;                  % -(a+Bu) - d <= 0
    lb = [-inf; -inf; zeros(N,1)];
    tic;
    [x2, ~, ef] = quadprog(H, zeros(2+N,1), A, b, [], [], lb, [], [], opt);
    t_qp = t_qp + toc;
    if ef <= 0, continue; end
    u2 = x2(1:2);  d2 = x2(3:end);
    J2 = u2.'*u2 + d2.'*d2;

    dJ = abs(J1 - J2) / max(1, abs(J2));
    du = norm(u1 - u2)  / max(1, norm(u2));
    max_dJ = max(max_dJ, dJ);  max_du = max(max_du, du);
    abs_du = max(abs_du, norm(u1 - u2));          % 绝对判据 ||u_enum-u_qp||
    abs_dJ = max(abs_dJ, abs(J1 - J2));
    u_scale = max(u_scale, norm(u2));
    if dJ > 1e-8, n_fail = n_fail + 1;
        fprintf('  ! trial %d  N=%d  J_ours=%.12g  J_qp=%.12g\n', k, N, J1, J2);
    end
end

fprintf('抽样 %d 次   (|u_qp| 最大量级 %.3g)\n', n_trial, u_scale);
fprintf('  绝对   max ||u_enum - u_qp||  : %.3e   (判据 < 1e-9: %s)\n', ...
    abs_du, ternary(abs_du < 1e-9, '通过', '未通过'));
fprintf('  绝对   max |J_enum - J_qp|    : %.3e\n', abs_dJ);
fprintf('  相对   max ||du||/max(1,|u|) : %.3e\n', max_du);
fprintf('  相对   max |dJ|/max(1,|J|)   : %.3e\n', max_dJ);
fprintf('  相对偏差超出 1e-8 的次数      : %d\n', n_fail);
fprintf('  耗时: qp2d %.3f s (%.1f us/次), quadprog %.3f s (%.1f us/次), 快 %.0fx\n', ...
    t_ours, t_ours/n_trial*1e6, t_qp, t_qp/n_trial*1e6, t_qp/t_ours);
end

% =====================================================================
function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end

% =====================================================================
% 与 safeflow_nn_rollout.m 中 qp2d 完全一致的副本（该函数是嵌套的，
% 无法从外部调用；任何一处改动需同步另一处）。
function [u, used_slack] = qp2d_ref(a, Bm)
N = numel(a);
best_u = zeros(2,1);
best_f = sum(max(0,-a).^2);
for mask = 1:(2^N - 1)
    V = bitget(mask, 1:N) > 0;
    Bv = Bm(V,:);  av = a(V);
    M  = eye(2) + (Bv.' * Bv);
    uc = M \ (-(Bv.' * av));
    r  = a + Bm*uc;
    f  = uc.'*uc + sum(max(0,-r).^2);
    if f < best_f, best_f = f; best_u = uc; end
end
u = best_u;
used_slack = any(a + Bm*u < -1e-9);
end
