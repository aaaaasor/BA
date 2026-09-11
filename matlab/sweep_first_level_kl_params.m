function results = sweep_first_level_kl_params(stage)
% Parameter-only first-level KL sweep.
% The rollout grid, safety margins/inflation and hard/soft ordering are
% copied from get_config and never changed here. Only first-level numerical
% gains/bounds and (in the explicitly named stages) switch times vary.

if nargin < 1, stage = 'omega'; end
stage = lower(string(stage));
root = fileparts(mfilename('fullpath'));
cfg0 = get_config();
out_dir = fullfile(root, 'outputs', 'FirstLevel_KL_ParameterSweep');
if ~isfolder(out_dir), mkdir(out_dir); end

% Stage A isolates omega. Stage B changes only switch times while retaining
% the configured hard->soft variance and soft->hard geometry directions.
activation_values = nan(1, 6);
beta_values = 0.01 * ones(1, 6);
gamma_values = 0.3 * ones(1, 6);
alpha_values = 8.0 * ones(1, 6);
budget_values = 10 * ones(1, 6);
hbar_values = 5 * ones(1, 6);
psi1_values = 2 * ones(1, 6);
alpha2_values = 3 * ones(1, 6);
if stage == "timing"
    variance_switch_values = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80];
    geometry_hard_values =   [0.60, 0.65, 0.70, 0.75, 0.80, 0.85];
    omega_values = 0.40 * ones(size(variance_switch_values));
    output_stem = 'timing_sweep';
elseif stage == "timing_fine"
    variance_switch_values = [0.60, 0.625, 0.65, 0.675, 0.70, 0.65, 0.65, 0.65, 0.65];
    geometry_hard_values =   [0.70, 0.70,  0.70, 0.70,  0.70, 0.65, 0.675, 0.725, 0.75];
    omega_values = 0.40 * ones(size(variance_switch_values));
    activation_values = nan(size(variance_switch_values));
    output_stem = 'timing_fine_sweep';
elseif stage == "activation"
    activation_values = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65];
    variance_switch_values = 0.65 * ones(size(activation_values));
    geometry_hard_values = 0.70 * ones(size(activation_values));
    omega_values = 0.40 * ones(size(activation_values));
    output_stem = 'activation_sweep';
elseif stage == "terminal"
    beta_values = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, ...
                   0.01, 0.01, 0.01, 0.01, 0.01, 0.01, ...
                   0.01, 0.01, 0.01, 0.01];
    gamma_values = [0.3 * ones(1, 6), 0.05, 0.10, 0.20, 0.50, 0.75, 1.0, ...
                    0.3, 0.3, 0.3, 0.3];
    alpha_values = [8.0 * ones(1, 12), 2.0, 4.0, 12.0, 16.0];
    omega_values = 0.40 * ones(size(beta_values));
    variance_switch_values = 0.70 * ones(size(beta_values));
    geometry_hard_values = 0.70 * ones(size(beta_values));
    activation_values = 0.60 * ones(size(beta_values));
    budget_values = 10 * ones(size(beta_values));
    hbar_values = 5 * ones(size(beta_values));
    psi1_values = 2 * ones(size(beta_values));
    alpha2_values = 3 * ones(size(beta_values));
    output_stem = 'terminal_sweep';
elseif stage == "hocbf"
    budget_values = [3, 5, 7.5, 10, 15, 20, 30, 40, 10, 10, 10, 10, 10, 10];
    hbar_values =  [5 * ones(1, 8), 1, 2.5, 10, 20, 5, 5];
    psi1_values =  [2 * ones(1, 12), 0.5, 5];
    alpha2_values = 3 * ones(size(budget_values));
    beta_values = 0.01 * ones(size(budget_values));
    gamma_values = 0.3 * ones(size(budget_values));
    alpha_values = 8.0 * ones(size(budget_values));
    omega_values = 0.40 * ones(size(budget_values));
    variance_switch_values = 0.70 * ones(size(budget_values));
    geometry_hard_values = 0.70 * ones(size(budget_values));
    activation_values = 0.60 * ones(size(budget_values));
    output_stem = 'hocbf_sweep';
else
    variance_switch_values = nan(1, 6);
    geometry_hard_values = nan(1, 6);
    omega_values = [0.015, 0.05, 0.10, 0.20, 0.40, 0.80];
    output_stem = 'omega_sweep';
end
n_cases = numel(omega_values);

% Expand the six-entry defaults for stages with a different case count.
if numel(beta_values) ~= n_cases, beta_values = 0.01 * ones(1, n_cases); end
if numel(gamma_values) ~= n_cases, gamma_values = 0.3 * ones(1, n_cases); end
if numel(alpha_values) ~= n_cases, alpha_values = 8.0 * ones(1, n_cases); end
if numel(budget_values) ~= n_cases, budget_values = 10 * ones(1, n_cases); end
if numel(hbar_values) ~= n_cases, hbar_values = 5 * ones(1, n_cases); end
if numel(psi1_values) ~= n_cases, psi1_values = 2 * ones(1, n_cases); end
if numel(alpha2_values) ~= n_cases, alpha2_values = 3 * ones(1, n_cases); end

fprintf('Preparing shared first-level data/model for %d cases...\n', n_cases);
rng(cfg0.random_seed);
[first_level_target_points, track_segment] = scenario_training_points( ...
    cfg0, 5, cfg0.n_train);
if ~isempty(track_segment)
    cfg0.track_segment = track_segment;
    if strcmpi(struct_field_default(cfg0, 'scenario', ''), 'racing') && ...
            struct_field_default(cfg0.obstacle, 'enabled', false)
        cfg0.obstacle = configure_racing_obstacles(track_segment, cfg0.obstacle);
    end
    if isfield(cfg0, 'track_boundary') && ...
            struct_field_default(cfg0.track_boundary, 'enabled', false)
        cfg0.track_boundary.geometry = build_track_boundary_geometry( ...
            track_segment, struct_field_default(cfg0.track_boundary, ...
            'n_spline_points', 400), ...
            struct_field_default(cfg0.track_boundary, 'spline_type', 'spline'));
    end
end
assert(cfg0.first_level_use_tangent_features, ...
    'This sweep expects the current 20D tangent-feature first level.');
rng(cfg0.first_level_data_seed);
[s_slices, x_slices, y_slices, target_points, ~, ~, source_data, ~, ...
    data_transform] = build_training_data(cfg0.t_min, 1.0, ...
    cfg0.n_time_slices, first_level_target_points);
feature_dim = size(target_points, 3);
n_rows = size(source_data, 1);
n_eval = cfg0.first_level_generation_samples;

rng(cfg0.first_level_hyperparameter_seed);
first_gp = cfg0.gp;
first_gp.o_ratio = cfg0.gp.first_level_o_ratio;
first_gp.hyperparameter_mat_path = cfg0.cache.first_level_hyperparameter_path;
first_gp.n_pretrain = cfg0.gp.first_level_n_pretrain;
first_gp = optimize_gp_hyperparameters(x_slices, y_slices, first_gp, s_slices);
first_gp.training_accuracy_threshold = ...
    cfg0.gp.first_level_training_accuracy_threshold;
rng(cfg0.first_level_fit_seed);
model_path = fullfile(root, cfg0.cache.first_level_model_path);
model_collection = fit_or_load_loggp_model( ...
    s_slices, x_slices, y_slices, first_gp, model_path, 'first-level');
model_collection = strip_model_for_prediction(model_collection, 'First-level sweep');

R = load(fullfile(root, cfg0.safe_flow_evaluation.kl_reference_path), ...
    'kl_reference');
reference = R.kl_reference;
filter_cfg = cfg0.first_level_seed_filter;
initial_beta_bound = seed_filter_initial_variance_bound( ...
    model_collection, filter_cfg, n_eval, 1, n_rows, cfg0.t_min);

template = struct('case_index', 0, 'omega', nan, 'beta_final', nan, ...
    'gamma', nan, 'alpha', nan, 'variance_switch_time', nan, ...
    'geometry_hard_time', nan, 'integral_budget', nan, ...
    'hocbf_relaxation_bound', nan, 'psi1_margin', nan, ...
    'hocbf_alpha2', nan, 'kl', nan, 'mean_dx', nan, ...
    'mean_dy', nan, 'std_ratio_x', nan, 'std_ratio_y', nan, ...
    'min_track_h', nan, 'min_obstacle_distance', nan, ...
    'unsafe_anchor_trajectories', nan, 'candidate_count', nan, ...
    'max_attempts', nan, 'rollout_seconds', nan);
rows = repmat(template, n_cases, 1);
best_kl = inf;
best_case = struct();

for case_idx = 1:n_cases
    cfg = cfg0;
    cfg.variance_constraint.first_level_joint_safety_phi1_omega = ...
        omega_values(case_idx);
    cfg.variance_constraint.first_level_terminal_variance_beta_final = ...
        beta_values(case_idx);
    cfg.variance_constraint.first_level_terminal_variance_ptzf_gamma = ...
        gamma_values(case_idx);
    cfg.variance_constraint.first_level_terminal_variance_alpha = ...
        alpha_values(case_idx);
    cfg.variance_constraint.first_level_integral_uncertainty_budget = ...
        budget_values(case_idx);
    cfg.variance_constraint.first_level_hocbf_relaxation_bound = ...
        hbar_values(case_idx);
    cfg.variance_constraint.first_level_psi1_margin = psi1_values(case_idx);
    cfg.variance_constraint.first_level_hocbf_alpha2 = alpha2_values(case_idx);
    if any(stage == ["timing", "timing_fine", "activation", "terminal", "hocbf"])
        cfg.variance_constraint.first_level_slack_switch_time = ...
            variance_switch_values(case_idx);
        cfg.variance_constraint.first_level_track_boundary_slack_hard_after_time = ...
            geometry_hard_values(case_idx);
        cfg.variance_constraint.first_level_obstacle_slack_hard_after_time = ...
            geometry_hard_values(case_idx);
    end
    if any(stage == ["activation", "terminal", "hocbf"])
        cfg.variance_constraint.first_level_track_boundary_activation_time = ...
            activation_values(case_idx);
        cfg.variance_constraint.first_level_obstacle_activation_time = ...
            activation_values(case_idx);
    end
    constraint = make_level_variance_constraint(cfg, 'first_level');
    constraint.terminal_variance_ptzf_hbar0_fixed = max( ...
        initial_beta_bound - constraint.terminal_variance_beta_final, 0.0) + ...
        constraint.terminal_variance_ptzf_initial_margin;
    if constraint.obstacle_enabled
        constraint.obstacle_point_maps = build_obstacle_point_maps( ...
            data_transform, feature_dim, 5, constraint.obstacle_points, 'absolute');
    end
    if constraint.track_boundary_enabled
        constraint.track_boundary_point_maps = build_obstacle_point_maps( ...
            data_transform, feature_dim, 5, ...
            constraint.track_boundary_points, 'absolute');
        [constraint.track_boundary_reference_s_min_targets, ...
            constraint.track_boundary_reference_s_max_targets] = ...
            build_track_boundary_reference_windows(n_eval, 1, 5, ...
            constraint.track_boundary_points);
    end

    fprintf('\nCASE %d/%d: omega=%.6g, beta=%.6g, gamma=%.6g, alpha=%.6g\n', ...
        case_idx, n_cases, constraint.joint_safety_phi1_omega, ...
        constraint.terminal_variance_beta_final, ...
        constraint.terminal_variance_ptzf_gamma, ...
        constraint.terminal_variance_alpha);
    [times, path, diagnostics, x_init, seeds, acceptance, uncertainty] = ...
        generate_valid_second_level_rollouts(model_collection, constraint, ...
        n_eval, 1, n_rows, 5, data_transform, cfg.t_min, cfg.rollout_t_max, ...
        cfg.first_level_time_steps, [], cfg.parallel, filter_cfg);

    [G, points] = first_level_physical_outputs(path, data_transform, feature_dim);
    [kl, mean_delta, std_ratio] = endpoint_stats(G, reference);
    [min_track, min_obstacle, unsafe_count] = anchor_safety(points, constraint);

    row = template;
    row.case_index = case_idx;
    row.omega = constraint.joint_safety_phi1_omega;
    row.beta_final = constraint.terminal_variance_beta_final;
    row.gamma = constraint.terminal_variance_ptzf_gamma;
    row.alpha = constraint.terminal_variance_alpha;
    row.variance_switch_time = ...
        constraint.terminal_variance_slack_late_start_time;
    row.geometry_hard_time = constraint.obstacle_slack_hard_after_time;
    row.integral_budget = constraint.integral_uncertainty_budget;
    row.hocbf_relaxation_bound = constraint.hocbf_relaxation_bound;
    row.psi1_margin = constraint.psi1_margin;
    row.hocbf_alpha2 = constraint.hocbf_alpha2;
    row.kl = kl;
    row.mean_dx = mean_delta(1);
    row.mean_dy = mean_delta(2);
    row.std_ratio_x = std_ratio(1);
    row.std_ratio_y = std_ratio(2);
    row.min_track_h = min_track;
    row.min_obstacle_distance = min_obstacle;
    row.unsafe_anchor_trajectories = unsafe_count;
    row.candidate_count = acceptance.total_candidate_count;
    row.max_attempts = max(acceptance.attempt_counts);
    row.rollout_seconds = diagnostics.rollout_elapsed_seconds;
    rows(case_idx) = row;
    fprintf(['RESULT omega=%.6g KL=%.9g stdRatio=[%.5f %.5f] ', ...
        'meanDelta=[%+.3g %+.3g] unsafe=%d candidates=%d time=%.1fs\n'], ...
        row.omega, row.kl, row.std_ratio_x, row.std_ratio_y, ...
        row.mean_dx, row.mean_dy, row.unsafe_anchor_trajectories, ...
        row.candidate_count, row.rollout_seconds);

    if kl < best_kl && unsafe_count == 0
        best_kl = kl;
        best_case = struct('row', row, 'times', times, 'path', path, ...
            'diagnostics', diagnostics, 'x_init', x_init, 'seeds', seeds, ...
            'acceptance', acceptance, 'uncertainty', uncertainty, ...
            'constraint', constraint, 'generated_final_xy', G);
    end
    results = struct2table(rows(1:case_idx));
    writetable(results, fullfile(out_dir, output_stem + ".csv"));
    save(fullfile(out_dir, output_stem + ".mat"), 'results', 'best_case', ...
        'omega_values', 'variance_switch_values', 'geometry_hard_values', ...
        'activation_values', 'beta_values', 'gamma_values', 'alpha_values', ...
        'budget_values', 'hbar_values', 'psi1_values', 'alpha2_values', '-v7.3');
end

results = struct2table(rows);
disp(sortrows(results, 'kl'));
fprintf('Best safe case: omega=%.6g KL=%.9g\n', ...
    best_case.row.omega, best_case.row.kl);
end

function [endpoints, points] = first_level_physical_outputs(path, transform, feature_dim)
z = squeeze(path(end, :, :));
x = z .* transform.std' + transform.mean';
n = size(x, 1);
points = permute(reshape(x', feature_dim, [], n), [2, 3, 1]);
endpoints = squeeze(points(end, :, 1:2));
end

function [kl, mean_delta, std_ratio] = endpoint_stats(generated, reference)
data = reference.dataset_final_xy;
[gx, gy] = meshgrid(reference.x_grid, reference.y_grid);
query = [gx(:), gy(:)];
bw = reference.bandwidth;
q = zeros(size(query, 1), 1);
for j = 1:size(generated, 1)
    d = (query - generated(j, :)) ./ bw;
    q = q + exp(-0.5 * sum(d .^ 2, 2));
end
q = max(q / (size(generated, 1) * 2 * pi * prod(bw)), realmin('double'));
p = reference.dataset_density(:);
area = (reference.x_grid(2) - reference.x_grid(1)) * ...
    (reference.y_grid(2) - reference.y_grid(1));
mask = p > 0;
kl = max(0, sum(p(mask) .* (log(p(mask)) - log(q(mask)))) * area);
mean_delta = mean(generated, 1) - mean(data, 1);
std_ratio = std(generated, 0, 1) ./ std(data, 0, 1);
end

function [min_track, min_obstacle, unsafe_count] = anchor_safety(points, constraint)
n_points = size(points, 1);
n = size(points, 2);
track_h = inf(n_points, n);
obstacle_d = inf(n_points, n);
fields = constraint.track_boundary_geometry.implicit_fields;
obstacles = constraint.obstacle_physical_geometry;
for i = 1:n
    for j = 1:n_points
        p = squeeze(points(j, i, 1:2));
        track_h(j, i) = min(evaluate_track_implicit_field(fields, 1, p), ...
            evaluate_track_implicit_field(fields, 2, p));
        for k = 1:size(obstacles.centers, 2)
            [h, g] = obstacle_level_and_gradient(p, obstacles, k);
            obstacle_d(j, i) = min(obstacle_d(j, i), ...
                h / max(norm(g), 1e-12));
        end
    end
end
min_track = min(track_h, [], 'all');
min_obstacle = min(obstacle_d, [], 'all');
unsafe_count = nnz(any(track_h < 0 | obstacle_d < 0, 1));
end
