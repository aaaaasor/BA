function diag = run_safeflow_omega20_width32_batch1(omega)
%RUN_SAFEFLOW_OMEGA20_WIDTH32_BATCH1 Run and archive the requested stress test.
% One trajectory is integrated per safeflow_nn_rollout call (true rollout
% batch size 1). Terminal failures are counted and archived, never rescued.

if nargin < 1 || isempty(omega), omega = 20; end

root = 'C:\Users\JieJi\BA\matlab';
cd(root);
out_dir = fullfile(root, 'outputs', [char([36187 36710]) 'safeflow']);
net_file = fullfile(root, 'outputs', 'NN_Size_Sweep', 'FM_Width032', ...
    'FM_Width032_Net.mat');
seed_file = fullfile(out_dir, 'Racing_SafeFlow_NN_Rollout.mat');

N = load(net_file, 'net');
A = load(seed_file, 'rollout');
net = N.net;
seeds = A.rollout.trajectory_seeds(:);
n_gen = numel(seeds);
n_train_steps = net.n_train_steps;
assert(net.training_options.hidden_width == 32, 'Expected the 32^3 network.');

base_opts = struct('u_per_stage', false, 'fmincon_fallback', false, ...
    'record_path', true, 'activation_time', 0, 'phi1_switch_time', 0, ...
    'phi1_form', 'second_order', 'phi1_omega', omega, ...
    'slack_weight', 1e4, 'slack_enabled', true, ...
    'terminal_filter', true, 'terminal_failure_fail_fast', false, ...
    'rollout_batch_size', 1);

fprintf(['Running SafeFlow stress diagnostic: width 32^3, rollout batch 1, ' ...
    'omega %g, slack weight 1e4, %d fixed trajectories.\n'], omega, n_gen);
ws1 = warning('off', 'MATLAB:nearlySingularMatrix');
ws2 = warning('off', 'MATLAB:singularMatrix');
cleanup_warning = onCleanup(@() restore_warnings(ws1, ws2)); %#ok<NASGU>

checkpoint_file = fullfile(root, 'outputs', sprintf( ...
    'SafeFlow_Omega%g_Width32_Batch1_checkpoint.mat', omega));
r = []; start_i = 1; prior_wall_seconds = 0;
if isfile(checkpoint_file)
    C = load(checkpoint_file, 'r', 'last_i', 'saved_seeds', 'prior_wall_seconds');
    if isequal(C.saved_seeds(:), seeds(:))
        r = C.r; start_i = C.last_i + 1;
        if isfield(C, 'prior_wall_seconds'), prior_wall_seconds = C.prior_wall_seconds; end
        fprintf('Resuming checkpoint at trajectory %d/%d.\n', start_i, n_gen);
    end
end
wall_tic = tic;
for i = start_i:n_gen
    oi = base_opts;
    oi.trajectory_seeds = seeds(i);
    q = safeflow_nn_rollout(net, 'safeflow', 1, oi);
    if isempty(r), r = q; else, r = append_rollout(r, q); end
    if mod(i, 10) == 0 || i == n_gen
        last_i = i; saved_seeds = seeds; %#ok<NASGU>
        prior_wall_seconds = prior_wall_seconds + toc(wall_tic); %#ok<NASGU>
        save(checkpoint_file, 'r', 'last_i', 'saved_seeds', ...
            'prior_wall_seconds', '-v7.3');
        wall_tic = tic;
        fprintf('  completed %d/%d; terminal failed points so far %d\n', ...
            i, n_gen, r.terminal_failed);
    end
end
r.total_seconds_per_traj = ...
    (r.sample_seconds + r.rollout_seconds + r.terminal_seconds) / r.n_gen;
r.rollout_batch_size = 1;
r.options.rollout_batch_size = 1;
r.wall_seconds = prior_wall_seconds + toc(wall_tic);
r.metrics = safeflow_nn_metrics(r);

[H, D] = safeflow_path_stats(r, net);
ns = size(D, 1);
t_step = (0:ns-1)' * r.t_max / ns;
front_mask = t_step < 0.5;
front_max = max(D(front_mask, :), [], 'all');
back_max = max(D(~front_mask, :), [], 'all');
onset = nan(n_gen, 1);
for i = 1:n_gen
    k = find(D(:,i) > 1, 1);
    if ~isempty(k), onset(i) = t_step(k); end
end

geom = final_geometry_diagnostic(r);
diag = struct('network_hidden_widths', [32 32 32], ...
    'rollout_batch_size', 1, 'omega', omega, 'slack_weight', 1e4, ...
    'options', base_opts, 'n_gen', n_gen, 'step_displacement', D, ...
    'h_path', H, 't_step', t_step, 'front_max_step_m', front_max, ...
    'back_max_step_m', back_max, 'divergence_threshold_m', 1, ...
    'divergence_onset', onset, ...
    'n_diverged_trajectories', nnz(~isnan(onset)), ...
    'n_diverged_before_t05', nnz(onset < 0.5), ...
    'terminal_failed_points', r.terminal_failed, ...
    'terminal_failed_trajectories', find(r.terminal_failed_per_trajectory > 0), ...
    'terminal_failed_per_trajectory', r.terminal_failed_per_trajectory, ...
    'terminal_rescued_by_fmincon', r.terminal_rescued_by_fmincon, ...
    'final_geometry', geom, 'metrics', r.metrics, ...
    'wall_seconds', r.wall_seconds, ...
    'note', ['Diagnostic stress run with omega/(1-t)^2 active from t=0. ' ...
        'Terminal failures are recorded without fmincon rescue.']);

cfg = get_config();
R = struct('safeflow', r);
archive_safeflow_nn_run(R, net, cfg, n_train_steps, n_gen, 'safeflow');
save(fullfile(out_dir, 'Racing_SafeFlow_NN_Diagnostics.mat'), 'diag', '-v7.3');
write_text_summary(fullfile(out_dir, 'Racing_SafeFlow_NN_Diagnostics.txt'), diag);
write_archive_sha256(out_dir);
if isfile(checkpoint_file), delete(checkpoint_file); end

fprintf('\n=== archived diagnostic ===\n');
fprintf('Safety %.2f%% | KL %.6g | CS %.6g | AS %.6g\n', ...
    100*r.metrics.safety, r.metrics.kl, r.metrics.cs, r.metrics.as);
fprintf('front max step %.6g m | back max step %.6g m\n', front_max, back_max);
fprintf('diverged %d/%d; before t=0.5: %d\n', ...
    diag.n_diverged_trajectories, n_gen, diag.n_diverged_before_t05);
fprintf('terminal failed points %d in %d trajectories; fmincon rescues %d\n', ...
    diag.terminal_failed_points, numel(diag.terminal_failed_trajectories), ...
    diag.terminal_rescued_by_fmincon);
fprintf('final geometry: obstacle %d, boundary %d, either %d of %d points\n', ...
    geom.inside_obstacle, geom.outside_track, geom.either, geom.total_points);
fprintf('archive: %s\n', out_dir);
end

function a = append_rollout(a, b)
a.points = cat(2, a.points, b.points);
a.features = cat(2, a.features, b.features);
a.state = cat(2, a.state, b.state);
a.z0 = cat(2, a.z0, b.z0);
a.z_path = cat(3, a.z_path, b.z_path);
a.trajectory_seeds = [a.trajectory_seeds; b.trajectory_seeds];
a.u_trace = cat(2, a.u_trace, b.u_trace);
a.u_trace_slack = cat(2, a.u_trace_slack, b.u_trace_slack);
a.terminal_failed_per_trajectory = [a.terminal_failed_per_trajectory; ...
    b.terminal_failed_per_trajectory];
sum_fields = {'sample_seconds','rollout_seconds','terminal_seconds','n_qp', ...
    'slack_active','slack_active_soft','infeasible_hard','terminal_corrected', ...
    'terminal_failed','terminal_rescued_by_fmincon','n_gen'};
for k = 1:numel(sum_fields)
    f = sum_fields{k}; a.(f) = a.(f) + b.(f);
end
end

function g = final_geometry_diagnostic(r)
P = r.points; [np, ng, ~] = size(P);
bad_o = false(np, ng); bad_b = false(np, ng);
min_o = inf(np, ng); min_b = inf(np, ng);
for s = 1:ng
    for ip = 1:np
        p = [P(ip,s,1); P(ip,s,2)];
        for jo = 1:size(r.obstacle.centers, 2)
            min_o(ip,s) = min(min_o(ip,s), ...
                obstacle_level_and_gradient(p, r.obstacle, jo));
        end
        for bi = 1:2
            min_b(ip,s) = min(min_b(ip,s), ...
                evaluate_track_implicit_field(r.geometry.implicit_fields, bi, p));
        end
    end
end
bad_o(:) = min_o < -1e-8; bad_b(:) = min_b < -1e-8;
g = struct('total_points', numel(bad_o), 'inside_obstacle', nnz(bad_o), ...
    'outside_track', nnz(bad_b), 'either', nnz(bad_o | bad_b), ...
    'max_obstacle_violation_h', max(max(-min_o, 0), [], 'all'), ...
    'max_boundary_violation_h', max(max(-min_b, 0), [], 'all'), ...
    'violating_trajectory_indices', find(any(bad_o | bad_b, 1))');
end

function write_text_summary(path, d)
fid = fopen(path, 'w'); c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'SafeFlow omega=%g diagnostic (width 32^3, rollout batch 1)\n', d.omega);
fprintf(fid, 'slack_weight = %.0f; gain = omega/(1-t)^2 from t=0\n', d.slack_weight);
fprintf(fid, 'Safety = %.8f%%\nKL = %.10g\nCS = %.10g\nAS = %.10g\n', ...
    100*d.metrics.safety, d.metrics.kl, d.metrics.cs, d.metrics.as);
fprintf(fid, 'front_max_step_m = %.10g\nback_max_step_m = %.10g\n', ...
    d.front_max_step_m, d.back_max_step_m);
fprintf(fid, 'diverged_trajectories = %d/%d\ndiverged_before_t0.5 = %d\n', ...
    d.n_diverged_trajectories, d.n_gen, d.n_diverged_before_t05);
fprintf(fid, 'terminal_failed_points = %d\nterminal_failed_trajectories = %s\n', ...
    d.terminal_failed_points, mat2str(d.terminal_failed_trajectories'));
fprintf(fid, 'terminal_rescued_by_fmincon = %d\n', d.terminal_rescued_by_fmincon);
g = d.final_geometry;
fprintf(fid, 'final_geometry = obstacle %d, boundary %d, either %d / %d points\n', ...
    g.inside_obstacle, g.outside_track, g.either, g.total_points);
fprintf(fid, 'max_obstacle_violation_h = %.10g\nmax_boundary_violation_h = %.10g\n', ...
    g.max_obstacle_violation_h, g.max_boundary_violation_h);
fprintf(fid, 'wall_seconds = %.3f\n', d.wall_seconds);
end

function restore_warnings(w1, w2)
warning(w1); warning(w2);
end
