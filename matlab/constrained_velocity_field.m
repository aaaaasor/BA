% Evaluate the PT-CBF-constrained trajectory-space velocity field.
function [v, diagnostics] = constrained_velocity_field(model_collection, t, x, ...
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
    diagnostics = [];
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
    if numel(fixed_mask) == y_dim
        variance_set(fixed_mask) = 0.0;
        grad_set(fixed_mask, :) = 0.0;
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
if isfield(constraint_cfg, 'uncertainty_max')
    uncertainty_max = constraint_cfg.uncertainty_max;
else
    uncertainty_max = constraint_cfg.sigma2_max;
end
grad_norm_sq = sum(grad_x .^ 2, 2);

u = zeros(size(mu'));
scale = 0.0;
slack = 0.0;
constraint_value = uncertainty_now - uncertainty_max;
[h_bar, h_bar_dot] = ptzf_bound(t, constraint_cfg);
gamma_gain = struct_field_default(constraint_cfg, 'ptzf_gamma', ...
    constraint_cfg.alpha_gain);
rho = uncertainty_t + sum(grad_x .* mu', 2) - ...
    gamma_gain * (h_bar - constraint_value) - h_bar_dot;
active = (rho > 0.0) & (grad_norm_sq > constraint_cfg.grad_tol);
rhs = rho;
if any(active)
    slack_weight = struct_field_default(constraint_cfg, ...
        'slack_weight', 100.0);
    denom = grad_norm_sq(active) + 1.0 / slack_weight;
    scale = -rho(active) ./ denom;
    u(active, :) = grad_x(active, :) .* scale;
    slack = rho(active) ./ (slack_weight * denom);
end
v = mu + u';
diagnostics.uncertainty = uncertainty_now;
diagnostics.h = uncertainty_max - uncertainty_now;
diagnostics.constraint_value = constraint_value;
diagnostics.h_bar = h_bar;
diagnostics.h_bar_dot = h_bar_dot;
diagnostics.rhs = rhs;
diagnostics.active = any(active);
diagnostics.slack = slack;
diagnostics.correction_norm = norm(u);
diagnostics.mu_norm = norm(mu);
diagnostics.velocity_norm = norm(v);
diagnostics.grad_norm = norm(grad_x);
diagnostics.uncertainty_t = uncertainty_t;
diagnostics.scale = scale;
end

function [h_bar, h_bar_dot] = ptzf_bound(t, constraint_cfg)
initial_bound = struct_field_default(constraint_cfg, ...
    'ptzf_initial_bound', 1.0);
gamma_gain = struct_field_default(constraint_cfg, 'ptzf_gamma', ...
    constraint_cfg.alpha_gain);
tau = min(max(t, 0.0), 1.0 - eps);
remaining_tau = 1.0 - tau;
h_bar = initial_bound * exp(-gamma_gain * tau / remaining_tau);
h_bar_dot = -h_bar * gamma_gain / ...
    (remaining_tau ^ 2);
end

function value = struct_field_default(value_struct, field_name, default_value)
if isfield(value_struct, field_name) && ~isempty(value_struct.(field_name))
    value = value_struct.(field_name);
else
    value = default_value;
end
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
