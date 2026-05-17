% Evaluate the PT-CBF-constrained velocity field from one LocalGP slice.
% This function reuses one LocalGP class method to obtain:
% - predictive mean mu,
% - scalar LocalGP predictive variance,
% - spatial gradient of that variance.
% The correction term u is then computed from the barrier inequality.
function v = constrained_velocity_field(model_collection, t, x, constraint_cfg)
t_slices = model_collection.t_slices(:);
if numel(t_slices) == 1
    index = 1;
else
    [~, index] = min(abs(t_slices - t));
end
model = model_collection.models{index};
x_col = reshape(x, [], 1);
[mu, variance_now, grad_x] = model.local_gp.predict_variance_grad(x_col);
mu = reshape(mu, [], 1);

%% Return the nominal velocity when the constraint is disabled
if nargin < 4 || ~constraint_cfg.enabled
    v = mu;
    return;
end

%% PT-CBF correction
normalized_t = min(max(t, 0.0), 1.0);
remaining = max(1.0 - normalized_t, constraint_cfg.time_eps);
phi_t = constraint_cfg.omega_gain / (remaining ^ 2);
h = constraint_cfg.sigma2_max - variance_now;
rhs = phi_t * constraint_cfg.alpha_gain * h - sum(grad_x .* mu', 2);
grad_norm_sq = sum(grad_x .^ 2, 2);

u = zeros(size(mu'));
active = (rhs < 0.0) & (grad_norm_sq > constraint_cfg.grad_tol);
if any(active)
    scale = rhs(active) ./ grad_norm_sq(active);
    u(active, :) = grad_x(active, :) .* scale;
end
v = mu + u';
end
