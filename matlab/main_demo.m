clear;
clc;

cfg = get_config();
rng(cfg.random_seed);
%% Training
x0_train = sample_source(cfg.n_train);
x1_train = sample_target(cfg.n_train, cfg.mixture);
[slice_times, x_slices, y_slices, x_train, y_train] = build_training_data(x0_train, x1_train, cfg.n_time_slices, cfg.t_min, 1.0);

gp_model = fit_time_slice_gp_models(slice_times, x_slices, y_slices, cfg.gp);
%%
x0_traj = sample_source(cfg.n_trajectories);
[~, traj_path] = rk4_rollout(gp_model, x0_traj, cfg.t_min, 1.0, cfg.time_steps, cfg.variance_constraint);
traj_times = linspace(cfg.t_min, 1.0, cfg.time_steps + 1)';
traj_gp_vars = zeros(cfg.time_steps + 1, cfg.n_trajectories, 2);
slice_times = gp_model.slice_times(:);
for step_idx = 1:(cfg.time_steps + 1)
    [~, slice_idx] = min(abs(slice_times - traj_times(step_idx)));
    slice_model = gp_model.models{slice_idx};
    x_now = squeeze(traj_path(step_idx, :, :));
    traj_gp_vars(step_idx, :, 1) = predict_gp_variance(slice_model.vx, x_now);
    traj_gp_vars(step_idx, :, 2) = predict_gp_variance(slice_model.vy, x_now);
end

plot_results(cfg, traj_path, x_train, y_train);
plot_variance_vs_time(cfg, traj_times, traj_gp_vars);

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
output_dir = fullfile(this_dir, 'outputs');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
traj_table = [traj_times, traj_path(:, :, 1), traj_path(:, :, 2)];
traj_headers = ["t", ...
    arrayfun(@(idx) "path_" + string(idx) + "_x", 0:(size(traj_path, 2) - 1)), ...
    arrayfun(@(idx) "path_" + string(idx) + "_y", 0:(size(traj_path, 2) - 1))];
writematrix(traj_headers, fullfile(output_dir, 'trajectory_samples.csv'));
writematrix(traj_table, fullfile(output_dir, 'trajectory_samples.csv'), 'WriteMode', 'append');

traj_gp_var_table = [traj_times, traj_gp_vars(:, :, 1), traj_gp_vars(:, :, 2)];
traj_gp_var_headers = ["t", ...
    arrayfun(@(idx) "path_" + string(idx) + "_var_vx", 0:(size(traj_gp_vars, 2) - 1)), ...
    arrayfun(@(idx) "path_" + string(idx) + "_var_vy", 0:(size(traj_gp_vars, 2) - 1))];
writematrix(traj_gp_var_headers, fullfile(output_dir, 'trajectory_gp_predictive_variances.csv'));
writematrix(traj_gp_var_table, fullfile(output_dir, 'trajectory_gp_predictive_variances.csv'), 'WriteMode', 'append');

disp(['Training samples: ', num2str(cfg.n_train)]);
disp(['Time slices: ', num2str(cfg.n_time_slices)]);
disp(['Total training pairs: ', num2str(size(x_train, 1))]);
disp(['Trajectory samples: ', num2str(cfg.n_trajectories)]);
disp(['Variance constraint enabled: ', num2str(cfg.variance_constraint.enabled)]);
disp(['Variance threshold sigma^2_max: ', num2str(cfg.variance_constraint.sigma2_max)]);
disp(['Variance-vs-time figure: ', fullfile(output_dir, 'trajectory_gp_variance_vs_time_matlab.png')]);
disp(['Final trajectory GP variance mean [vx, vy]: ', num2str(mean(squeeze(traj_gp_vars(end, :, :)), 1))]);
disp(['Final trajectory GP variance max [vx, vy]: ', num2str(max(squeeze(traj_gp_vars(end, :, :)), [], 1))]);
