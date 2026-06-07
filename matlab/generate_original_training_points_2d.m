% Recreate original 2D trajectories with tangent features [dx/ds, dy/ds].
function points = generate_original_training_points_2d(n_points, ...
    n_trajectories)
if nargin < 1 || isempty(n_points)
    n_points = 17;
end
if nargin < 2 || isempty(n_trajectories)
    n_trajectories = 500;
end

this_dir = fileparts(mfilename('fullpath'));
trajectory_data_dir = fullfile(this_dir, 'trajectory_data');
if exist(trajectory_data_dir, 'dir') && ~contains(path, trajectory_data_dir)
    addpath(trajectory_data_dir);
end

previous_rng_state = rng;
restore_rng_state = onCleanup(@() rng(previous_rng_state));
rng(0);

p_quantity = 3;
r_set = nan(1, p_quantity);
p_0_set = nan(2, p_quantity);
theta_min_set = nan(1, p_quantity);
theta_max_set = nan(1, p_quantity);

p_nr = 1;
r_set(p_nr) = 0.1;
p_0_set(:, p_nr) = [0.1; 0.1];
theta_min_set(p_nr) = deg2rad(-30);
theta_max_set(p_nr) = deg2rad(30);

p_nr = 2;
r_set(p_nr) = 0.2;
p_0_set(:, p_nr) = [0.5; 0.5];
theta_min_set(p_nr) = deg2rad(60);
theta_max_set(p_nr) = deg2rad(120);

p_nr = 3;
r_set(p_nr) = 0.1;
p_0_set(:, p_nr) = [0.9; 0.9];
theta_min_set(p_nr) = deg2rad(-30);
theta_max_set(p_nr) = deg2rad(30);

feature_dim = 4;
points = nan(n_points, n_trajectories, feature_dim);
for trajectory_nr = 1:n_trajectories
    p_set = nan(2, p_quantity);
    dp_set = nan(2, p_quantity);
    for p_nr = 1:p_quantity
        r_i = r_set(p_nr);
        p_0_i = p_0_set(:, p_nr);
        theta_i_min = theta_min_set(p_nr);
        theta_i_max = theta_max_set(p_nr);

        alpha_i = 2 * pi * rand(1);
        p_i = r_i * rand(1) * [cos(alpha_i); sin(alpha_i)] + p_0_i;
        theta_i = rand(1) * (theta_i_max - theta_i_min) + theta_i_min;
        dp_i = [cos(theta_i); sin(theta_i)];

        p_set(:, p_nr) = p_i;
        dp_set(:, p_nr) = dp_i;
    end

    t_set = (0:(p_quantity - 1)) / (p_quantity - 1);
    polynomial_coefficient_x = GPFM_DataGeneration_fit_Polynomial( ...
        t_set, p_set(1, :), dp_set(1, :));
    polynomial_coefficient_y = GPFM_DataGeneration_fit_Polynomial( ...
        t_set, p_set(2, :), dp_set(2, :));

    t_dense_set = linspace(0, 1, 5000);
    x_dense_set = ppval(polynomial_coefficient_x, t_dense_set);
    y_dense_set = ppval(polynomial_coefficient_y, t_dense_set);
    ds_dense_set = sqrt((x_dense_set(2:end) - x_dense_set(1:(end - 1))) .^ 2 + ...
        (y_dense_set(2:end) - y_dense_set(1:(end - 1))) .^ 2);
    s_dense_set = [0, cumsum(ds_dense_set)];
    trajectory_length = s_dense_set(end);

    s_set = linspace(0, trajectory_length, n_points);
    t_train_i = interp1(s_dense_set, t_dense_set, s_set);
    x_train_i = ppval(polynomial_coefficient_x, t_train_i);
    y_train_i = ppval(polynomial_coefficient_y, t_train_i);
    points(:, trajectory_nr, 1) = x_train_i;
    points(:, trajectory_nr, 2) = y_train_i;

    dx_dt_train_i = ppval(pp_derivative(polynomial_coefficient_x), ...
        t_train_i);
    dy_dt_train_i = ppval(pp_derivative(polynomial_coefficient_y), ...
        t_train_i);
    speed_train_i = sqrt(dx_dt_train_i .^ 2 + dy_dt_train_i .^ 2);
    speed_train_i = max(speed_train_i, eps);

    points(:, trajectory_nr, 3) = dx_dt_train_i ./ speed_train_i;
    points(:, trajectory_nr, 4) = dy_dt_train_i ./ speed_train_i;
end
end

function derivative_pp = pp_derivative(pp)
breaks = pp.breaks;
coefs = pp.coefs;
order = pp.order;
powers = (order - 1):-1:1;
derivative_coefs = coefs(:, 1:(end - 1)) .* powers;
derivative_pp = mkpp(breaks, derivative_coefs, pp.dim);
end
