% Evaluate the PT-CBF-constrained trajectory-space velocity field.
function v = constrained_velocity_field(model_collection, t, x, constraint_cfg)
%% GP Prediction
x_col = reshape(x, [], 1);
model = model_collection.model;
gp_input = [t; x_col];
[mu, variance_now, grad_all] = model.local_gp.predict_variance_grad(gp_input);
variance_t = grad_all(1);
grad_x = grad_all(2:end);
mu = reshape(mu, [], 1);

%% Nominal Velocity
if nargin < 4 || ~constraint_cfg.enabled
    v = mu;
    return;
end

%% PT-CBF correction
normalized_t = min(max(t, 0.0), 1.0);
remaining = max(1.0 - normalized_t, constraint_cfg.time_eps);
phi_t = constraint_cfg.omega_gain / (remaining ^ 2);
h = constraint_cfg.sigma2_max - variance_now;
rhs = phi_t * constraint_cfg.alpha_gain * h - variance_t - sum(grad_x .* mu', 2);
grad_norm_sq = sum(grad_x .^ 2, 2);

u = zeros(size(mu'));
active = (rhs < 0.0) & (grad_norm_sq > constraint_cfg.grad_tol);
if any(active)
    scale = rhs(active) ./ grad_norm_sq(active);
    u(active, :) = grad_x(active, :) .* scale;
end
v = mu + u';
end
