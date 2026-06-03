% Roll out segment states and project each step onto smooth fixed-anchor curves.
function [times, path] = rk4_rollout_segmented_smooth(model_collection, ...
    x_init, t0, t1, n_steps, constraint_cfg, fixed_state_mask, ...
    fixed_state_values, n_trajectories, n_segments, n_points_per_segment, ...
    curvature_weight, max_projection_strength)

if nargin < 11 || isempty(curvature_weight)
    curvature_weight = 10.0;
end
if nargin < 12 || isempty(max_projection_strength)
    max_projection_strength = 1.0;
end

times = linspace(t0, t1, n_steps + 1)';
dt = (t1 - t0) / n_steps;
n_samples = size(x_init, 1);
state_dim = size(x_init, 2);
if n_samples ~= n_trajectories * n_segments
    error('Segment sample count must equal n_trajectories * n_segments.');
end

path = zeros(n_steps + 1, n_samples, state_dim);
x_now_all = x_init;
x_now_all(fixed_state_mask) = fixed_state_values(fixed_state_mask);
path(1, :, :) = reshape(x_now_all, 1, n_samples, state_dim);

for step_idx = 1:n_steps
    t_now = times(step_idx);
    x_next_all = zeros(size(x_now_all));
    for sample_idx = 1:n_samples
        fixed_mask_now = fixed_state_mask(sample_idx, :)';
        fixed_values_now = fixed_state_values(sample_idx, :)';
        x_now = x_now_all(sample_idx, :)';

        k1 = constrained_velocity_field(model_collection, t_now, x_now, ...
            constraint_cfg);
        k1(fixed_mask_now) = 0.0;
        k2 = constrained_velocity_field(model_collection, ...
            t_now + 0.5 * dt, ...
            apply_fixed_state_values(x_now + 0.5 * dt * k1, ...
            fixed_mask_now, fixed_values_now), constraint_cfg);
        k2(fixed_mask_now) = 0.0;
        k3 = constrained_velocity_field(model_collection, ...
            t_now + 0.5 * dt, ...
            apply_fixed_state_values(x_now + 0.5 * dt * k2, ...
            fixed_mask_now, fixed_values_now), constraint_cfg);
        k3(fixed_mask_now) = 0.0;
        k4 = constrained_velocity_field(model_collection, t_now + dt, ...
            apply_fixed_state_values(x_now + dt * k3, fixed_mask_now, ...
            fixed_values_now), constraint_cfg);
        k4(fixed_mask_now) = 0.0;

        x_next = x_now + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        x_next(fixed_mask_now) = fixed_values_now(fixed_mask_now);
        x_next_all(sample_idx, :) = x_next';
    end

    x_projected_all = project_segment_states(x_next_all, n_trajectories, ...
        n_segments, n_points_per_segment, curvature_weight);
    projection_strength = (times(step_idx + 1) - t0) / max(t1 - t0, eps);
    projection_strength = min(max(projection_strength, 0.0), 1.0);
    projection_strength = projection_strength * max_projection_strength;
    x_next_all = apply_safe_projection(model_collection, ...
        times(step_idx + 1), x_next_all, x_projected_all, ...
        projection_strength, constraint_cfg, fixed_state_mask, ...
        fixed_state_values, n_trajectories, n_segments);
    x_next_all(fixed_state_mask) = fixed_state_values(fixed_state_mask);
    x_now_all = x_next_all;
    path(step_idx + 1, :, :) = reshape(x_now_all, 1, n_samples, state_dim);

    if mod(step_idx, 10) == 0 || step_idx == n_steps
        fprintf('  RK4 smooth step %d / %d...\n', step_idx, n_steps);
    end
end
end

function x_safe_all = apply_safe_projection(model_collection, t_now, ...
    x_raw_all, x_projected_all, projection_strength, constraint_cfg, ...
    fixed_state_mask, fixed_state_values, n_trajectories, n_segments)
if projection_strength <= 0
    x_safe_all = x_raw_all;
    return;
end

x_safe_all = x_raw_all;
candidate_strengths = projection_strength * [1.0, 0.5, 0.25, ...
    0.125, 0.0625, 0.0];
if ~constraint_cfg.enabled
    x_safe_all = (1.0 - projection_strength) * x_raw_all + ...
        projection_strength * x_projected_all;
    x_safe_all(fixed_state_mask) = fixed_state_values(fixed_state_mask);
    return;
end

for traj_idx = 1:n_trajectories
    sample_idx = ((traj_idx - 1) * n_segments + 1):(traj_idx * n_segments);
    raw_now = x_raw_all(sample_idx, :);
    projected_now = x_projected_all(sample_idx, :);
    fixed_mask_now = fixed_state_mask(sample_idx, :);
    fixed_values_now = fixed_state_values(sample_idx, :);
    best_now = raw_now;
    best_uncertainty = max_segment_uncertainty(model_collection, t_now, ...
        raw_now);
    accepted = best_uncertainty <= constraint_cfg.sigma2_max;

    for strength_idx = 1:numel(candidate_strengths)
        strength = candidate_strengths(strength_idx);
        candidate_now = (1.0 - strength) * raw_now + ...
            strength * projected_now;
        candidate_now(fixed_mask_now) = fixed_values_now(fixed_mask_now);
        candidate_uncertainty = max_segment_uncertainty(model_collection, ...
            t_now, candidate_now);
        if candidate_uncertainty < best_uncertainty
            best_uncertainty = candidate_uncertainty;
            best_now = candidate_now;
        end
        if candidate_uncertainty <= constraint_cfg.sigma2_max
            best_now = candidate_now;
            accepted = true;
            break;
        end
    end

    if ~accepted
        best_now(fixed_mask_now) = fixed_values_now(fixed_mask_now);
    end
    x_safe_all(sample_idx, :) = best_now;
end
end

function max_uncertainty = max_segment_uncertainty(model_collection, ...
    t_now, segment_states)
max_uncertainty = 0.0;
for sample_idx = 1:size(segment_states, 1)
    z_now = [t_now; segment_states(sample_idx, :)'];
    uncertainty_now = predict_uncertainty(model_collection.model, z_now);
    max_uncertainty = max(max_uncertainty, uncertainty_now);
end
end

function uncertainty_now = predict_uncertainty(model, gp_input)
if isfield(model, 'output_models')
    variance_set = zeros(numel(model.output_models), 1);
    for output_idx = 1:numel(model.output_models)
        [~, variance_set(output_idx)] = ...
            model.output_models{output_idx}.predict_variance_grad(gp_input);
    end
    uncertainty_now = sqrt(max(sum(variance_set), 0.0));
else
    [~, uncertainty_now] = model.local_gp.predict_variance_grad(gp_input);
end
end

function x_fixed = apply_fixed_state_values(x, fixed_mask, fixed_values)
x_fixed = x;
x_fixed(fixed_mask) = fixed_values(fixed_mask);
end

function segment_states = project_segment_states(segment_states, ...
    n_trajectories, n_segments, n_points_per_segment, curvature_weight)
trajectory_points = stitch_segment_points(segment_states, n_trajectories, ...
    n_segments, n_points_per_segment);
anchor_idx = 1:(n_points_per_segment - 1):size(trajectory_points, 1);
trajectory_points = smooth_fixed_anchor_points(trajectory_points, ...
    anchor_idx, curvature_weight);
segment_states = unstitch_segment_points(trajectory_points, n_segments, ...
    n_points_per_segment);
end

function segment_states = unstitch_segment_points(trajectory_points, ...
    n_segments, n_points_per_segment)
n_trajectories = size(trajectory_points, 2);
n_rows = 2 * n_points_per_segment;
segment_states = zeros(n_trajectories * n_segments, n_rows);
for traj_idx = 1:n_trajectories
    for segment_idx = 1:n_segments
        sample_idx = (traj_idx - 1) * n_segments + segment_idx;
        start_idx = (segment_idx - 1) * (n_points_per_segment - 1) + 1;
        point_idx = start_idx:(start_idx + n_points_per_segment - 1);
        segment_curve = squeeze(trajectory_points(point_idx, traj_idx, :));
        segment_states(sample_idx, :) = reshape(segment_curve', 1, []);
    end
end
end
