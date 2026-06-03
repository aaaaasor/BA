% Roll out trajectory-space states with a fixed-step classical RK4 integrator.
function [times, path] = rk4_rollout(model_collection, ...
    x_init, t0, t1, n_steps, constraint_cfg, fixed_state_mask, ...
    fixed_state_values)

%% Default Arguments
if nargin < 6
    constraint_cfg.enabled = false;
end
if nargin < 7 || isempty(fixed_state_mask)
    fixed_state_mask = false(size(x_init));
end
if nargin < 8 || isempty(fixed_state_values)
    fixed_state_values = x_init;
end

%% Allocate Path
times = linspace(t0, t1, n_steps + 1)';
dt = (t1 - t0) / n_steps;
n_samples = size(x_init, 1);
state_dim = size(x_init, 2);
if isvector(fixed_state_mask)
    fixed_state_mask = repmat(reshape(fixed_state_mask, 1, []), ...
        n_samples, 1);
end
if isvector(fixed_state_values)
    fixed_state_values = repmat(reshape(fixed_state_values, 1, []), ...
        n_samples, 1);
end
path = zeros(n_steps + 1, n_samples, state_dim);
path(1, :, :) = reshape(x_init, 1, n_samples, state_dim);

%% RK4 Integration
for sample_idx = 1:n_samples
    x_now = x_init(sample_idx, :)';
    fixed_mask_now = fixed_state_mask(sample_idx, :)';
    fixed_values_now = fixed_state_values(sample_idx, :)';
    x_now(fixed_mask_now) = fixed_values_now(fixed_mask_now);
    path(1, sample_idx, :) = x_now;
    for step_idx = 1:n_steps
        t_now = times(step_idx);
        k1 = constrained_velocity_field(model_collection, t_now, x_now, ...
            constraint_cfg, fixed_mask_now);
        k1(fixed_mask_now) = 0.0;
        k2 = constrained_velocity_field(model_collection, ...
            t_now + 0.5 * dt, ...
            apply_fixed_state_values(x_now + 0.5 * dt * k1, ...
            fixed_mask_now, fixed_values_now), constraint_cfg, ...
            fixed_mask_now);
        k2(fixed_mask_now) = 0.0;
        k3 = constrained_velocity_field(model_collection, ...
            t_now + 0.5 * dt, ...
            apply_fixed_state_values(x_now + 0.5 * dt * k2, ...
            fixed_mask_now, fixed_values_now), constraint_cfg, ...
            fixed_mask_now);
        k3(fixed_mask_now) = 0.0;
        k4 = constrained_velocity_field(model_collection, t_now + dt, ...
            apply_fixed_state_values(x_now + dt * k3, fixed_mask_now, ...
            fixed_values_now), constraint_cfg, fixed_mask_now);
        k4(fixed_mask_now) = 0.0;
        x_now = x_now + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        x_now(fixed_mask_now) = fixed_values_now(fixed_mask_now);
        path(step_idx + 1, sample_idx, :) = x_now;
    end
    if mod(sample_idx, 10) == 0 || sample_idx == n_samples
        fprintf('  RK4 rolled out %d / %d samples...\n', sample_idx, n_samples);
    end
end
end

function x_fixed = apply_fixed_state_values(x, fixed_mask, fixed_values)
x_fixed = x;
x_fixed(fixed_mask) = fixed_values(fixed_mask);
end
