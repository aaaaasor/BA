function v = constrained_velocity_field(model_collection, t, x, constraint_cfg)
mu = velocity_field(model_collection, t, x);
if nargin < 4 || ~constraint_cfg.enabled
    v = mu;
    return;
end

slice_times = model_collection.slice_times(:);
if numel(slice_times) == 1
    index = 1;
else
    [~, index] = min(abs(slice_times - t));
end
model = model_collection.models{index};
x_row = reshape(x, 1, []);

[variance_now, grad_x] = predict_variance_scalar_and_grad_x_scalar(model, x_row);
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


function [y_var, grad] = predict_variance_scalar_and_grad_x_scalar(model, x_test)
[grad_x, var_x] = predict_gp_variance_grad(model.vx, x_test);
[grad_y, var_y] = predict_gp_variance_grad(model.vy, x_test);
y_var = 0.5 * (var_x + var_y);
grad = 0.5 * (grad_x + grad_y);
end
