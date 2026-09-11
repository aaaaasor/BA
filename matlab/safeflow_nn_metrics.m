function m = safeflow_nn_metrics(r)
%SAFEFLOW_NN_METRICS  Archive metrics for one safeflow_nn_rollout result.
%
% Extracted verbatim from safeflow_nn_demo's local compute_metrics so that a
% single run can be re-archived on its own without going through the demo.
% Safety uses the true boundary h >= -1e-8, with no guidance margin.
P = r.points;                      % (65, n, 2)
[nPt, n, ~] = size(P);

% ---- Safety：真实边界 h >= 0，不含 margin ----
ok = true(1,n); hmin_o = inf; hmin_b = inf;
for i = 1:n
    for k = 1:nPt
        p = [P(k,i,1); P(k,i,2)];
        for jo = 1:size(r.obstacle.centers,2)
            h = obstacle_level_and_gradient(p, r.obstacle, jo);
            hmin_o = min(hmin_o, h);
            if h < -1e-8, ok(i) = false; end
        end
        for bi = 1:2
            hr = evaluate_track_implicit_field(r.geometry.implicit_fields, bi, p);
            hmin_b = min(hmin_b, hr);          % 不减 margin
            if hr < -1e-8, ok(i) = false; end
        end
    end
end
m.safety = mean(ok);
m.min_obstacle_h = hmin_o;
m.min_boundary_h = hmin_b;

% ---- CS / AS ----
cs = nan(n,1); as = nan(n,1);
for i = 1:n
    xy = squeeze(P(:,i,1:2));
    w  = diff(xy,1,1);
    L  = sqrt(sum(w.^2,2));
    den = L(1:end-1).*L(2:end);
    ct  = ones(size(den));
    v_  = den > eps;
    num = sum(w(1:end-1,:).*w(2:end,:),2);
    ct(v_) = num(v_)./den(v_);
    cs(i) = mean(1 - min(max(ct,-1),1));
    acc = diff(xy,2,1);
    as(i) = mean(sqrt(sum(acc.^2,2)));
end
m.cs = mean(cs); m.as = mean(as);
m.cs_per = cs;   m.as_per = as;

% ---- Time：采样 + rollout + 终端滤波，除以条数 ----
m.time_seconds = r.total_seconds_per_traj;
m.sample_seconds = r.sample_seconds;
m.rollout_seconds = r.rollout_seconds;
m.terminal_seconds = r.terminal_seconds;
m.terminal_failed = r.terminal_failed;
m.slack_active = r.slack_active;
m.n_qp = r.n_qp;

% ---- KL：复用主流程的 Racing_KL_Reference.mat（同网格同带宽，跨方法可比）----
m.final_xy = squeeze(P(end,:,1:2));
[m.kl, m.kl_details] = safeflow_nn_kl(m.final_xy);
end
