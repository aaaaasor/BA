% Main entry point for the MATLAB 2D time-slice LocalGP demo.
% This script:
% 1. loads pre-generated 2D trajectory samples,
% 2. builds x_t -> v_t training data from those trajectories,
% 3. fits one LocalGP velocity model per time slice,
% 4. reconstructs a subset of trajectories through RK4 rollout,
% 5. evaluates LocalGP predictive variance along the rollout trajectories,
% 6. exports figures and CSV summaries.
clear;
clc;

cfg = get_config();
rng(cfg.random_seed);
%% Training data generation from pre-generated 2D trajectories
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
trajectory_mat_path = fullfile(this_dir, 'trajectory_data', 'data_2D.mat');
[x_slices, y_slices, trajectory_points, x_train, y_train, trajectory_data, t_slices] = build_training_data(trajectory_mat_path, cfg.t_min, 1.0, cfg.n_train);

model_collection = fit_time_slice_gp_models(t_slices, x_slices, y_slices, cfg.gp);

%% Rollout-based trajectory reconstruction from x_t -> v_t LocalGP
n_available_traj = size(trajectory_points, 2);
traj_idx = randperm(n_available_traj, min(cfg.n_trajectories, n_available_traj));
selected_points = trajectory_points(:, traj_idx, :);
n_eval = numel(traj_idx);
x_init = squeeze(selected_points(1, :, :));
if n_eval == 1
    x_init = reshape(x_init, 1, 2);
end
[rollout_times, reconstructed_points] = rk4_rollout(model_collection, ...
    x_init, cfg.t_min, 1.0, cfg.time_steps, cfg.variance_constraint);

traj_gp_vars = zeros(numel(rollout_times), n_eval, 2);
for sample_idx = 1:n_eval
    for time_idx = 1:numel(rollout_times)
        t_now = rollout_times(time_idx);
        [~, slice_idx] = min(abs(model_collection.t_slices - t_now));
        model = model_collection.models{slice_idx};
        x_now = squeeze(reconstructed_points(time_idx, sample_idx, :));
        [~, variance_now] = model.local_gp.predict_variance_grad(x_now);
        traj_gp_vars(time_idx, sample_idx, :) = variance_now;
    end
end

%% Plotting
plot_results(cfg, trajectory_points, reconstructed_points, x_train, y_train);
plot_variance_vs_time(cfg, rollout_times, traj_gp_vars);

%% Export results
output_dir = fullfile(this_dir, 'outputs');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
reconstructed_data = zeros(n_eval, 2 * size(reconstructed_points, 1));
for sample_idx = 1:n_eval
    reconstructed_curve = squeeze(reconstructed_points(:, sample_idx, :));
    reconstructed_data(sample_idx, :) = reshape(reconstructed_curve', 1, []);
end
reconstruction_headers = ["trajectory_index", arrayfun(@(idx) "coord_" + string(idx), 1:size(reconstructed_data, 2))];
reconstruction_table = [(traj_idx(:) - 1), reconstructed_data];
writematrix(reconstruction_headers, fullfile(output_dir, 'trajectory_reconstruction_samples.csv'));
writematrix(reconstruction_table, fullfile(output_dir, 'trajectory_reconstruction_samples.csv'), 'WriteMode', 'append');

traj_gp_var_table = [rollout_times, traj_gp_vars(:, :, 1), traj_gp_vars(:, :, 2)];
traj_gp_var_headers = ["t", ...
    arrayfun(@(idx) "path_" + string(idx) + "_var_vx", 0:(size(traj_gp_vars, 2) - 1)), ...
    arrayfun(@(idx) "path_" + string(idx) + "_var_vy", 0:(size(traj_gp_vars, 2) - 1))];
writematrix(traj_gp_var_headers, fullfile(output_dir, 'trajectory_gp_predictive_variances.csv'));
writematrix(traj_gp_var_table, fullfile(output_dir, 'trajectory_gp_predictive_variances.csv'), 'WriteMode', 'append');

%% Console summary
disp(['Training trajectories loaded: ', num2str(size(trajectory_data, 2))]);
disp(['Trajectory points per sample: ', num2str(numel(t_slices))]);
disp(['LocalGP input dimension: ', num2str(size(x_slices, 3))]);
disp(['Total training trajectories: ', num2str(size(x_slices, 2))]);
disp(['Reconstructed trajectories: ', num2str(n_eval)]);
disp(['Variance threshold sigma^2_max: ', num2str(cfg.variance_constraint.sigma2_max)]);
disp(['Variance-vs-time figure: ', fullfile(output_dir, 'trajectory_gp_variance_vs_time_matlab.png')]);
disp(['Final trajectory LocalGP variance mean [vx, vy]: ', num2str(mean(squeeze(traj_gp_vars(end, :, :)), 1))]);
disp(['Final trajectory LocalGP variance max [vx, vy]: ', num2str(max(squeeze(traj_gp_vars(end, :, :)), [], 1))]);
