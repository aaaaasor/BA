% Main entry point for the MATLAB 10D trajectory-space LocalGP ODE demo.
clear;
clc;

%% Configuration
cfg = get_config();
rng(cfg.random_seed);

%% Training Data
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
trajectory_mat_path = fullfile(this_dir, 'trajectory_data', 'data_2D.mat');
[flow_slices, x_slices, y_slices, target_points, source_points, ...
    target_data, source_data, trajectory_t_slices] = build_training_data( ...
    trajectory_mat_path, cfg.t_min, 1.0, cfg.n_train, cfg.n_time_slices);

%% Fit LocalGP Flow Slices
model_collection = fit_time_slice_gp_models(flow_slices, x_slices, y_slices, cfg.gp);

%% RK4 Rollout
n_available_traj = size(source_data, 2);
traj_idx = randperm(n_available_traj, min(cfg.n_trajectories, n_available_traj));
x_init = source_data(:, traj_idx)';
n_eval = size(x_init, 1);
[rollout_times, traj_path_10d] = rk4_rollout(model_collection, ...
    x_init, cfg.t_min, 1.0, cfg.time_steps, cfg.variance_constraint);

final_trajectory_data = squeeze(traj_path_10d(end, :, :));
if n_eval == 1
    final_trajectory_data = reshape(final_trajectory_data, 1, []);
end

reconstructed_points = zeros(numel(trajectory_t_slices), n_eval, 2);
source_plot_points = source_points(:, traj_idx, :);
for sample_idx = 1:n_eval
    reconstructed_curve = reshape(final_trajectory_data(sample_idx, :), 2, [])';
    reconstructed_points(:, sample_idx, :) = reconstructed_curve;
end

%% Predictive Variance
traj_gp_vars = zeros(numel(rollout_times), n_eval, 1);
for sample_idx = 1:n_eval
    for time_idx = 1:numel(rollout_times)
        s_now = rollout_times(time_idx);
        [~, slice_idx] = min(abs(model_collection.t_slices - s_now));
        model = model_collection.models{slice_idx};
        z_now = squeeze(traj_path_10d(time_idx, sample_idx, :));
        [~, variance_now] = model.local_gp.predict_variance_grad(z_now);
        traj_gp_vars(time_idx, sample_idx, 1) = variance_now;
    end
end

%% Plot Results
plot_results(cfg, target_points, source_plot_points, reconstructed_points);
plot_variance_vs_time(cfg, rollout_times, traj_gp_vars);

%% Export Results
output_dir = fullfile(this_dir, 'outputs');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
reconstruction_headers = ["trajectory_index", ...
    arrayfun(@(idx) "coord_" + string(idx), ...
    1:size(final_trajectory_data, 2))];
reconstruction_table = [(traj_idx(:) - 1), final_trajectory_data];
writematrix(reconstruction_headers, fullfile(output_dir, 'trajectory_reconstruction_samples.csv'));
writematrix(reconstruction_table, ...
    fullfile(output_dir, 'trajectory_reconstruction_samples.csv'), ...
    'WriteMode', 'append');

traj_gp_var_table = [rollout_times, traj_gp_vars(:, :, 1)];
traj_gp_var_headers = ["s", ...
    arrayfun(@(idx) "path_" + string(idx) + "_var", ...
    0:(size(traj_gp_vars, 2) - 1))];
writematrix(traj_gp_var_headers, fullfile(output_dir, 'trajectory_gp_predictive_variances.csv'));
writematrix(traj_gp_var_table, ...
    fullfile(output_dir, 'trajectory_gp_predictive_variances.csv'), ...
    'WriteMode', 'append');

%% Console Summary
disp(['Training trajectories loaded: ', num2str(size(target_data, 2))]);
disp(['Trajectory points per sample: ', num2str(numel(trajectory_t_slices))]);
disp(['10D ODE flow slices: ', num2str(numel(flow_slices))]);
disp(['LocalGP input dimension: ', num2str(size(x_slices, 3))]);
disp(['LocalGP output dimension: ', num2str(size(y_slices, 3))]);
disp(['Total training trajectories: ', num2str(size(x_slices, 2))]);
disp(['Rolled-out trajectories: ', num2str(n_eval)]);
disp(['Variance threshold sigma^2_max: ', num2str(cfg.variance_constraint.sigma2_max)]);
disp(['Variance-vs-time figure: ', ...
    fullfile(output_dir, 'trajectory_gp_variance_vs_time_matlab.emf')]);
disp(['Final 10D trajectory LocalGP variance mean: ', num2str(mean(traj_gp_vars(end, :, 1)))]);
disp(['Final 10D trajectory LocalGP variance max: ', num2str(max(traj_gp_vars(end, :, 1)))]);
