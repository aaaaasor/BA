% Evaluate the PT-CBF-constrained trajectory-space velocity field.
function v = constrained_velocity_field(model_collection, t, x, ...
    constraint_cfg, fixed_mask)
%% GP Prediction
x_col = reshape(x, [], 1);
if nargin < 5 || isempty(fixed_mask)
    fixed_mask = false(size(x_col));
end
fixed_mask = reshape(fixed_mask, [], 1);
model = model_collection.model;
gp_input = [t; x_col];
if nargin < 4
    v = predict_mean_velocity(model, gp_input);
    return;
end

%% GP Prediction With Uncertainty Gradient
if isfield(model, 'output_models')
    y_dim = numel(model.output_models);
    mu = zeros(y_dim, 1);
    variance_set = zeros(y_dim, 1);
    grad_set = zeros(y_dim, numel(gp_input));
    for output_idx = 1:y_dim
        [mu_now, variance_now, grad_now] = ...
            model.output_models{output_idx}.predict_variance_grad(gp_input);
        mu(output_idx) = mu_now;
        variance_set(output_idx) = variance_now;
        grad_set(output_idx, :) = grad_now;
    end
    uncertainty_now = sqrt(max(sum(variance_set), 0.0));
    if uncertainty_now > eps
        grad_all = sum(grad_set, 1) ./ (2.0 * uncertainty_now);
    else
        grad_all = zeros(1, numel(gp_input));
    end
else
    [mu, variance_now, variance_grad] = ...
        model.local_gp.predict_variance_grad(gp_input);
    uncertainty_now = sqrt(max(variance_now, 0.0));
    if uncertainty_now > eps
        grad_all = variance_grad ./ (2.0 * uncertainty_now);
    else
        grad_all = zeros(1, numel(gp_input));
    end
end
uncertainty_t = grad_all(1);
grad_x = grad_all(2:end);
grad_x(fixed_mask') = 0.0;
mu = reshape(mu, [], 1);

%% PT-CBF correction
normalized_t = min(max(t, 0.0), 1.0);
remaining = 1.0 - normalized_t;
phi_t = constraint_cfg.omega_gain / (remaining ^ 2);
if isfield(constraint_cfg, 'uncertainty_max')
    uncertainty_max = constraint_cfg.uncertainty_max;
else
    uncertainty_max = constraint_cfg.sigma2_max;
end
h = uncertainty_max - uncertainty_now;
rhs = phi_t * constraint_cfg.alpha_gain * h - uncertainty_t - ...
    sum(grad_x .* mu', 2);
grad_norm_sq = sum(grad_x .^ 2, 2);

u = zeros(size(mu'));
active = (rhs < 0.0) & (grad_norm_sq > constraint_cfg.grad_tol);
if any(active)
    scale = rhs(active) ./ grad_norm_sq(active);
    u(active, :) = grad_x(active, :) .* scale;
end
v = mu + u';
end

function mu = predict_mean_velocity(model, gp_input)
if isfield(model, 'output_models')
    y_dim = numel(model.output_models);
    mu = zeros(y_dim, 1);
    for output_idx = 1:y_dim
        mu(output_idx) = model.output_models{output_idx}.predict( ...
            gp_input, 0.0);
    end
else
    mu = model.local_gp.predict(gp_input, zeros(model.local_gp.y_dim, 1));
end
mu = reshape(mu, [], 1);
end
