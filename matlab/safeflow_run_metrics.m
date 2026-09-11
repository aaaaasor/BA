function m = safeflow_run_metrics(r, net)
%SAFEFLOW_RUN_METRICS  Metrics for one safeflow_nn_rollout result.
%
% Safety/CS/AS/KL use exactly the definitions in safeflow_nn_demo's local
% compute_metrics (safety tolerance -1e-8 on the true boundary, no margin).
% Two extras are added for the divergence diagnostics: the worst h anywhere
% along the path and the largest single-step physical displacement, which
% require record_path to have been on.

% Identical safety/CS/AS/KL definitions to safeflow_nn_demo's compute_metrics.
P = r.points; [nPt, n, ~] = size(P);
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
            hmin_b = min(hmin_b, hr);
            if hr < -1e-8, ok(i) = false; end
        end
    end
end
m.safety = mean(ok);
m.min_obstacle_h = hmin_o;
m.min_boundary_h = hmin_b;

cs = nan(n,1); as = nan(n,1);
for i = 1:n
    xy = squeeze(P(:,i,1:2));
    w = diff(xy,1,1); L = sqrt(sum(w.^2,2));
    den = L(1:end-1).*L(2:end); ct = ones(size(den)); v_ = den > eps;
    ct(v_) = sum(w(1:end-1,:).*w(2:end,:),2)./den(v_);
    cs(i) = mean(1 - min(max(ct,-1),1));
    acc = diff(xy,2,1); as(i) = mean(sqrt(sum(acc.^2,2)));
end
m.cs = mean(cs); m.as = mean(as);
m.time_seconds = r.total_seconds_per_traj;
m.terminal_failed = r.terminal_failed;
m.slack_active = r.slack_active;
m.n_qp = r.n_qp;
m.kl = safeflow_nn_kl(squeeze(P(end,:,1:2)));

if isempty(r.u_trace)
    m.u_mean = 0; m.u_p95 = 0; m.u_max = 0;
else
    nu = sqrt(sum(double(r.u_trace).^2, 4));
    m.u_mean = mean(nu(:)); m.u_p95 = quantile(nu(:),0.95); m.u_max = max(nu(:));
end

% Worst h anywhere along the path, and the largest single-step move: the two
% signatures of a front-end blow-up.
[H, D] = safeflow_path_stats(r, net);
m.h_path_min = min(H(:));
m.max_step_disp = max(D(:));
end
