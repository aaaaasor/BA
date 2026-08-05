% Fast first-level rollout for tuning HOCBF/PTCBF bounds.
% - Uses cached GP model (built once, then reused).
% - Runs rollout with current cfg, dumps compact summary to stdout.
% - Deletes the first-level rollout cache so every run re-computes with current config.
try
    project_dir = 'C:\Users\JieLi\OneDrive - LBGruppe\Documents\New project\matlab';
    cd(project_dir);
    cfg = get_config();

    % force re-rollout every iteration
    rollout_cache_full = fullfile(project_dir, cfg.cache.first_level_rollout_path);
    if isfile(rollout_cache_full); delete(rollout_cache_full); end

    rng(cfg.random_seed);

    % Training data (same setup as main_demo.m)
    [first_level_target_points, track_segment] = scenario_training_points( ...
        cfg, 5, cfg.n_train);
    if ~isempty(track_segment) && ...
            strcmpi(struct_field_default(cfg, 'scenario', ''), 'racing') && ...
            struct_field_default(cfg.obstacle, 'enabled', false)
        cfg.obstacle = configure_racing_obstacles(track_segment, cfg.obstacle);
    end
    if ~cfg.first_level_use_tangent_features
        error('Loop expects 20D tangent-feature setup; toggle disabled.');
    end
    rng(cfg.first_level_data_seed);
    [s_slices, x_slices, y_slices, target_points, source_points, ...
        target_data, source_data, trajectory_t_slices, data_transform] = ...
        build_training_data(cfg.t_min, 1.0, cfg.n_time_slices, first_level_target_points);
    first_level_feature_dim = size(target_points, 3);

    % LoG-GP model — will be loaded from cache
    rng(cfg.first_level_hyperparameter_seed);
    first_gp = cfg.gp;
    first_gp.hyperparameter_mat_path = cfg.cache.first_level_hyperparameter_path;
    first_gp.n_pretrain = cfg.gp.first_level_n_pretrain;
    first_gp = optimize_gp_hyperparameters(x_slices, y_slices, first_gp, s_slices);
    first_gp.training_accuracy_threshold = cfg.gp.first_level_training_accuracy_threshold;
    rng(cfg.first_level_fit_seed);
    first_model_cache_path = fullfile(project_dir, cfg.cache.first_level_model_path);
    model_collection = fit_or_load_loggp_model(s_slices, x_slices, y_slices, ...
        first_gp, first_model_cache_path, 'first-level');

    % Rollout
    n_rows = size(source_data, 1);
    rng(cfg.first_level_rollout_seed);
    n_eval = cfg.first_level_generation_samples;
    x_init = randn(n_eval, n_rows);
    first_rollout_constraint = make_level_variance_constraint(cfg, 'first_level');
    if first_rollout_constraint.obstacle_enabled
        first_rollout_constraint.obstacle_point_maps = ...
            build_obstacle_point_maps(data_transform, first_level_feature_dim, ...
            5, first_rollout_constraint.obstacle_points, 'absolute');
    end
    [rollout_times, traj_path_10d, first_rollout_diagnostics] = rk4_rollout(...
        model_collection, x_init, cfg.t_min, cfg.rollout_t_max, ...
        cfg.first_level_time_steps, first_rollout_constraint);

    % Extract compact summary
    trace = first_rollout_diagnostics.hocbf;
    tt = trace.trace_t(:);
    smp = trace.trace_sample_idx(:);
    ib = trace.trace_integral_bound(:);
    tb = trace.trace_terminal_bound(:);
    hres = trace.trace_hocbf_constraint_residual(:);
    tres = trace.trace_terminal_constraint_residual(:);
    hslack = trace.trace_hocbf_slack(:);
    tslack = trace.trace_terminal_slack(:);

    fprintf('\n=== FIRST-LEVEL SUMMARY ===\n');
    fprintf('samples=%d, trace points=%d, t range=[%.3f, %.3f]\n', ...
        n_eval, numel(tt), min(tt), max(tt));

    % Aggregate by time bin (10 bins over t range)
    tmin = min(tt); tmax = max(tt);
    n_bin = 10;
    edges = linspace(tmin, tmax, n_bin + 1);
    fprintf('\ntime bins:                integral_bound(HOCBF)    terminal_bound(PTCBF)   which_smaller\n');
    for k = 1:n_bin
        mask = tt >= edges(k) & tt < edges(k+1) + eps;
        if ~any(mask); continue; end
        ib_med = median(ib(mask), 'omitnan');
        tb_med = median(tb(mask), 'omitnan');
        if ib_med < tb_med; who = 'HOCBF'; else; who = 'PTCBF'; end
        fprintf('  s in [%.3f, %.3f]:  %+10.3f          %+10.3f          %s\n', ...
            edges(k), edges(k+1), ib_med, tb_med, who);
    end

    fprintf('\n--- constraint residuals (should be <= 0 for satisfied) ---\n');
    fprintf('HOCBF residual max = %+.4f, mean = %+.4f, frac(>0) = %.3f\n', ...
        max(hres), mean(hres), mean(hres > 1e-6));
    fprintf('PTCBF residual max = %+.4f, mean = %+.4f, frac(>0) = %.3f\n', ...
        max(tres), mean(tres), mean(tres > 1e-6));
    fprintf('HOCBF slack max = %.4f, mean = %.4f\n', max(hslack), mean(hslack));
    fprintf('PTCBF slack max = %.4f, mean = %.4f\n', max(tslack), mean(tslack));

    % Terminal beta at final step
    if isfield(trace, 'trace_terminal_inequality_h')
        tih = trace.trace_terminal_inequality_h(:);
        end_mask = tt >= 0.98 * max(tt);
        fprintf('\nsigma2 - beta_final at s>=0.98*max: mean=%+.3f, max=%+.3f (must be <= 0 for hard PTCBF ok)\n', ...
            mean(tih(end_mask)), max(tih(end_mask)));
    end

    % Cumulative variance at end
    cv = trace.trace_cumulative_variance(:);
    for si = 1:n_eval
        m = smp == si;
        if any(m)
            fprintf('sample %d: max cumulative_variance = %+.3f (budget B*t_max = %.3f)\n', ...
                si, max(cv(m)), first_rollout_constraint.integral_uncertainty_budget * max(tt));
        end
    end

    % Trajectory quality — deviation from target curves for final points
    traj_path_plot = zeros(size(traj_path_10d));
    for time_idx = 1:numel(rollout_times)
        states_now = squeeze(traj_path_10d(time_idx, :, :));
        if n_eval == 1
            states_now = reshape(states_now, 1, []);
        end
        states_plot = states_now .* data_transform.std' + data_transform.mean';
        traj_path_plot(time_idx, :, :) = reshape(states_plot, 1, n_eval, []);
    end
    final_data = squeeze(traj_path_plot(end, :, :));
    if n_eval == 1
        final_data = reshape(final_data, 1, []);
    end
    reconstructed = zeros(numel(trajectory_t_slices), n_eval, first_level_feature_dim);
    for si = 1:n_eval
        curve = reshape(final_data(si, :), first_level_feature_dim, [])';
        reconstructed(:, si, :) = curve;
    end
    reconstructed = normalize_tangent_features(reconstructed);
    % Compare xy of reconstructed to nearest target trajectory (best of n_train)
    n_pts = size(reconstructed, 1);
    max_dev = zeros(n_eval, 1);
    for si = 1:n_eval
        rec_xy = squeeze(reconstructed(:, si, 1:2));
        d = inf;
        for ti = 1:size(target_points, 2)
            tgt_xy = squeeze(target_points(:, ti, 1:2));
            di = mean(vecnorm(rec_xy - tgt_xy, 2, 2));
            d = min(d, di);
        end
        max_dev(si) = d;
    end
    fprintf('\nper-sample mean-distance-to-nearest-target-trajectory (physical units):\n');
    for si = 1:n_eval
        fprintf('  sample %d: %.4f\n', si, max_dev(si));
    end
    fprintf('median across samples: %.4f\n', median(max_dev));

    % Save current rollout for later plot inspection
    tune_out_path = fullfile(project_dir, 'outputs', 'tune_first_level_last.mat');
    save(tune_out_path, 'rollout_times', 'traj_path_10d', ...
        'first_rollout_diagnostics', 'first_rollout_constraint', ...
        'reconstructed', 'target_points', '-v7.3');
    fprintf('\nSaved: %s\n', tune_out_path);
    fprintf('=== DONE ===\n');
catch err
    fprintf('ERROR: %s\n', err.message);
    fprintf('%s\n', err.getReport());
    exit(1);
end
exit(0);
