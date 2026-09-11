function [H, D] = safeflow_path_stats(r, net)
%SAFEFLOW_PATH_STATS  Per-step worst CBF value and largest point displacement.
%
% Requires the rollout to have been run with record_path on.
% H is (step x trajectory), D is (step-1 x trajectory), both in physical units.

% H: (step x trajectory) worst h over the 65 points, obstacles and boundary.
% D: (step x trajectory) largest physical displacement of any point in a step.
[n_t, Dd, ng] = size(r.z_path);
nF = 4; nP = Dd / nF;
H = zeros(n_t, ng);
XY = zeros(n_t, nP, ng, 2);
for k = 1:n_t
    X = squeeze(r.z_path(k,:,:)) .* net.sd_d + net.mu_d;
    F = permute(reshape(X, nF, nP, ng), [2 3 1]);
    XY(k,:,:,1) = F(:,:,1); XY(k,:,:,2) = F(:,:,2);
    for s = 1:ng
        hm = inf;
        for ip = 1:nP
            p = [F(ip,s,1); F(ip,s,2)];
            for jo = 1:size(r.obstacle.centers,2)
                hm = min(hm, obstacle_level_and_gradient(p, r.obstacle, jo));
            end
            for bi = 1:2
                hm = min(hm, evaluate_track_implicit_field( ...
                    r.geometry.implicit_fields, bi, p));
            end
        end
        H(k,s) = hm;
    end
end
dxy = diff(XY, 1, 1);
D = squeeze(max(sqrt(sum(dxy.^2, 4)), [], 2));   % (step-1 x trajectory)
end
