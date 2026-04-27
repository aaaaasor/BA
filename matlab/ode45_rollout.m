function [times, path] = ode45_rollout(model_collection, x_init, t0, t1, n_steps)
times = linspace(t0, t1, n_steps + 1)';
path = zeros(n_steps + 1, numel(x_init));

for i = 1:numel(x_init)
    x_start = x_init(i);
    ode_fun = @(t, x) velocity_field(model_collection, t, x);
    [t_sol, x_sol] = ode45(ode_fun, times, x_start);
    path(:, i) = interp1(t_sol, x_sol, times, 'linear', 'extrap');
end
end
