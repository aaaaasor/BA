% Roll out trajectory-space states with a fixed-step classical RK4 integrator.
function [times, path] = rk4_rollout(model_collection, ...
    x_init, t0, t1, n_steps, constraint_cfg)

%% Default Arguments
if nargin < 6
    constraint_cfg.enabled = false;
end

%% Allocate Path
times = linspace(t0, t1, n_steps + 1)';
dt = (t1 - t0) / n_steps;
n_samples = size(x_init, 1);
state_dim = size(x_init, 2);
path = zeros(n_steps + 1, n_samples, state_dim);
path(1, :, :) = reshape(x_init, 1, n_samples, state_dim);

%% RK4 Integration
for sample_idx = 1:n_samples
    x_now = x_init(sample_idx, :)';
    path(1, sample_idx, :) = x_now;
    for step_idx = 1:n_steps
        t_now = times(step_idx);
        k1 = constrained_velocity_field(model_collection, t_now, x_now, constraint_cfg);
        k2 = constrained_velocity_field(model_collection, t_now + 0.5 * dt, ...
            x_now + 0.5 * dt * k1, constraint_cfg);
        k3 = constrained_velocity_field(model_collection, t_now + 0.5 * dt, ...
            x_now + 0.5 * dt * k2, constraint_cfg);
        k4 = constrained_velocity_field(model_collection, t_now + dt, ...
            x_now + dt * k3, constraint_cfg);
        x_now = x_now + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        path(step_idx + 1, sample_idx, :) = x_now;
    end
end
end
