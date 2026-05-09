function [times, path] = ode45_rollout(model_collection, x_init, t0, t1, n_steps, constraint_cfg)
if nargin < 6
    constraint_cfg.enabled = false;
end

times = linspace(t0, t1, n_steps + 1)';
n_samples = size(x_init, 1);
path = zeros(n_steps + 1, n_samples, 2);

for i = 1:n_samples
    x_start = x_init(i, :)';
    ode_fun = @(t, x) constrained_velocity_field(model_collection, t, x, constraint_cfg);
    [t_sol, x_sol] = ode45(ode_fun, times, x_start);
    path(:, i, 1) = interp1(t_sol, x_sol(:, 1), times, 'linear', 'extrap');
    path(:, i, 2) = interp1(t_sol, x_sol(:, 2), times, 'linear', 'extrap');
end
end
