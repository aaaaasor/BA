% Roll out trajectory-space states with a fixed-step classical RK4 integrator.
function [times, path, diagnostics] = rk4_rollout(model_collection, ...
    x_init, t0, t1, n_steps, constraint_cfg, fixed_state_mask, ...
    fixed_state_values, fixed_clf_cfg)

%% Default Arguments
if nargin < 6
    constraint_cfg = [];
end
if nargin < 7 || isempty(fixed_state_mask)
    fixed_state_mask = false(size(x_init));
end
if nargin < 8 || isempty(fixed_state_values)
    fixed_state_values = x_init;
end
if nargin < 9 || isempty(fixed_clf_cfg)
    fixed_clf_cfg.alpha = 5.0;
    fixed_clf_cfg.grad_tol = 1e-10;
    fixed_clf_cfg.diagnostics = false;
end
if ~isfield(fixed_clf_cfg, 'alpha') || isempty(fixed_clf_cfg.alpha)
    fixed_clf_cfg.alpha = 5.0;
end
if ~isfield(fixed_clf_cfg, 'grad_tol') || isempty(fixed_clf_cfg.grad_tol)
    fixed_clf_cfg.grad_tol = 1e-10;
end
if ~isfield(fixed_clf_cfg, 'diagnostics') || isempty(fixed_clf_cfg.diagnostics)
    fixed_clf_cfg.diagnostics = false;
end
%% Allocate Path
times = linspace(t0, t1, n_steps + 1)';
dt = (t1 - t0) / n_steps;
n_samples = size(x_init, 1);
state_dim = size(x_init, 2);
if isvector(fixed_state_mask)
    fixed_state_mask = repmat(reshape(fixed_state_mask, 1, []), ...
        n_samples, 1);
end
if isvector(fixed_state_values)
    fixed_state_values = repmat(reshape(fixed_state_values, 1, []), ...
        n_samples, 1);
end
path = zeros(n_steps + 1, n_samples, state_dim);
path(1, :, :) = reshape(x_init, 1, n_samples, state_dim);
use_constraint = ~isempty(constraint_cfg);
diagnostics.max_clf_correction_norm = 0.0;
diagnostics.max_fixed_reference_error = 0.0;
diagnostics.max_linear_clf_correction_norm = 0.0;
diagnostics.max_linear_reference_error = 0.0;
diagnostics.ptcbf = init_ptcbf_diagnostics();

%% RK4 Integration
for sample_idx = 1:n_samples
    x_now = x_init(sample_idx, :)';
    fixed_mask_now = fixed_state_mask(sample_idx, :)';
    fixed_values_now = fixed_state_values(sample_idx, :)';
    path(1, sample_idx, :) = x_now;
    for step_idx = 1:n_steps
        t_now = times(step_idx);
        [k1, ptcbf_info] = rollout_velocity(model_collection, t_now, x_now, ...
            constraint_cfg, use_constraint, fixed_mask_now);
        diagnostics.ptcbf = update_ptcbf_diagnostics( ...
            diagnostics.ptcbf, ptcbf_info, sample_idx, step_idx, 'k1');
        [k1, diagnostics] = apply_fixed_clf(k1, x_now, ...
            fixed_mask_now, fixed_values_now, fixed_clf_cfg, diagnostics, ...
            sample_idx);
        [k2, ptcbf_info] = rollout_velocity(model_collection, ...
            t_now + 0.5 * dt, ...
            x_now + 0.5 * dt * k1, constraint_cfg, use_constraint, ...
            fixed_mask_now);
        diagnostics.ptcbf = update_ptcbf_diagnostics( ...
            diagnostics.ptcbf, ptcbf_info, sample_idx, step_idx, 'k2');
        [k2, diagnostics] = apply_fixed_clf(k2, ...
            x_now + 0.5 * dt * k1, fixed_mask_now, fixed_values_now, ...
            fixed_clf_cfg, diagnostics, sample_idx);
        [k3, ptcbf_info] = rollout_velocity(model_collection, ...
            t_now + 0.5 * dt, ...
            x_now + 0.5 * dt * k2, constraint_cfg, use_constraint, ...
            fixed_mask_now);
        diagnostics.ptcbf = update_ptcbf_diagnostics( ...
            diagnostics.ptcbf, ptcbf_info, sample_idx, step_idx, 'k3');
        [k3, diagnostics] = apply_fixed_clf(k3, ...
            x_now + 0.5 * dt * k2, fixed_mask_now, fixed_values_now, ...
            fixed_clf_cfg, diagnostics, sample_idx);
        [k4, ptcbf_info] = rollout_velocity(model_collection, t_now + dt, ...
            x_now + dt * k3, constraint_cfg, use_constraint, ...
            fixed_mask_now);
        diagnostics.ptcbf = update_ptcbf_diagnostics( ...
            diagnostics.ptcbf, ptcbf_info, sample_idx, step_idx, 'k4');
        [k4, diagnostics] = apply_fixed_clf(k4, x_now + dt * k3, ...
            fixed_mask_now, fixed_values_now, fixed_clf_cfg, diagnostics, ...
            sample_idx);
        x_now = x_now + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        path(step_idx + 1, sample_idx, :) = x_now;
    end
    if mod(sample_idx, 10) == 0 || sample_idx == n_samples
        fprintf('  RK4 rolled out %d / %d samples...\n', sample_idx, n_samples);
    end
end
if fixed_clf_cfg.diagnostics
    disp(['Fixed-mask CLF max correction norm: ', ...
        num2str(diagnostics.max_clf_correction_norm)]);
    disp(['Fixed-mask CLF max reference error: ', ...
        num2str(diagnostics.max_fixed_reference_error)]);
    disp(['Linear CLF max correction norm: ', ...
        num2str(diagnostics.max_linear_clf_correction_norm)]);
    disp(['Linear CLF max reference error: ', ...
        num2str(diagnostics.max_linear_reference_error)]);
end
if use_constraint && isfield(constraint_cfg, 'diagnostics') && ...
        constraint_cfg.diagnostics
    print_ptcbf_diagnostics(diagnostics.ptcbf);
end
end

function [v, ptcbf_info] = rollout_velocity(model_collection, t, x, constraint_cfg, ...
    use_constraint, fixed_mask)
if use_constraint
    [v, ptcbf_info] = constrained_velocity_field(model_collection, t, x, ...
        constraint_cfg, fixed_mask);
else
    v = constrained_velocity_field(model_collection, t, x);
    ptcbf_info = [];
end
end

function [v, diagnostics] = apply_fixed_clf(v, x, fixed_mask, ...
    fixed_values, fixed_clf_cfg, diagnostics, sample_idx)
if any(fixed_mask)
    e = x(fixed_mask) - fixed_values(fixed_mask);
    v_fixed = v(fixed_mask);
    error_norm = norm(e);
    diagnostics.max_fixed_reference_error = max( ...
        diagnostics.max_fixed_reference_error, error_norm);
    if error_norm > fixed_clf_cfg.grad_tol
        V = 0.5 * sum(e .^ 2);
        clf_value = e' * v_fixed + fixed_clf_cfg.alpha * V;
        if clf_value > 0.0
            u = - (clf_value / (sum(e .^ 2) + eps)) * e;
            v(fixed_mask) = v_fixed + u;
            diagnostics.max_clf_correction_norm = max( ...
                diagnostics.max_clf_correction_norm, norm(u));
        end
    end
end

if ~isfield(fixed_clf_cfg, 'linear_clf') || ...
        ~isfield(fixed_clf_cfg.linear_clf, 'enabled') || ...
        ~fixed_clf_cfg.linear_clf.enabled
    return;
end
linear_cfg = fixed_clf_cfg.linear_clf;
if sample_idx > size(linear_cfg.A, 1)
    return;
end
A = squeeze(linear_cfg.A(sample_idx, :, :));
b = squeeze(linear_cfg.b(sample_idx, :))';
if isempty(A) || isempty(b)
    return;
end
g = A * x - b;
error_norm = norm(g);
diagnostics.max_linear_reference_error = max( ...
    diagnostics.max_linear_reference_error, error_norm);
if error_norm <= fixed_clf_cfg.grad_tol
    return;
end
V = 0.5 * sum(g .^ 2);
grad_v = A' * g;
clf_value = grad_v' * v + fixed_clf_cfg.alpha * V;
if clf_value <= 0.0
    return;
end
u = - (clf_value / (sum(grad_v .^ 2) + eps)) * grad_v;
v = v + u;
diagnostics.max_clf_correction_norm = max( ...
    diagnostics.max_clf_correction_norm, norm(u));
diagnostics.max_linear_clf_correction_norm = max( ...
    diagnostics.max_linear_clf_correction_norm, norm(u));
end

function ptcbf_diag = init_ptcbf_diagnostics()
ptcbf_diag.call_count = 0;
ptcbf_diag.active_count = 0;
ptcbf_diag.max_uncertainty = 0.0;
ptcbf_diag.min_h = inf;
ptcbf_diag.min_rhs = inf;
ptcbf_diag.max_rhs = -inf;
ptcbf_diag.max_correction_norm = 0.0;
ptcbf_diag.max_mu_norm = 0.0;
ptcbf_diag.max_velocity_norm = 0.0;
ptcbf_diag.max_grad_norm = 0.0;
ptcbf_diag.min_active_grad_norm = inf;
ptcbf_diag.max_abs_scale = 0.0;
ptcbf_diag.max_slack = 0.0;
ptcbf_diag.sum_slack = 0.0;
ptcbf_diag.slack_count = 0;
ptcbf_diag.max_uncertainty_t_abs = 0.0;
ptcbf_diag.max_correction_to_mu_ratio = 0.0;
ptcbf_diag.max_correction_sample_idx = nan;
ptcbf_diag.max_correction_step_idx = nan;
ptcbf_diag.max_correction_stage = '';
ptcbf_diag.max_correction_uncertainty = nan;
ptcbf_diag.max_correction_h = nan;
ptcbf_diag.max_correction_rhs = nan;
ptcbf_diag.max_correction_grad_norm = nan;
ptcbf_diag.max_correction_scale = nan;
end

function ptcbf_diag = update_ptcbf_diagnostics(ptcbf_diag, info, ...
    sample_idx, step_idx, stage_name)
if isempty(info)
    return;
end
ptcbf_diag.call_count = ptcbf_diag.call_count + 1;
ptcbf_diag.active_count = ptcbf_diag.active_count + double(info.active);
ptcbf_diag.max_uncertainty = max(ptcbf_diag.max_uncertainty, ...
    info.uncertainty);
ptcbf_diag.min_h = min(ptcbf_diag.min_h, info.h);
ptcbf_diag.min_rhs = min(ptcbf_diag.min_rhs, info.rhs);
ptcbf_diag.max_rhs = max(ptcbf_diag.max_rhs, info.rhs);
if info.correction_norm > ptcbf_diag.max_correction_norm
    ptcbf_diag.max_correction_norm = info.correction_norm;
    ptcbf_diag.max_correction_sample_idx = sample_idx;
    ptcbf_diag.max_correction_step_idx = step_idx;
    ptcbf_diag.max_correction_stage = stage_name;
    ptcbf_diag.max_correction_uncertainty = info.uncertainty;
    ptcbf_diag.max_correction_h = info.h;
    ptcbf_diag.max_correction_rhs = info.rhs;
    ptcbf_diag.max_correction_grad_norm = info.grad_norm;
    ptcbf_diag.max_correction_scale = info.scale;
end
ptcbf_diag.max_mu_norm = max(ptcbf_diag.max_mu_norm, info.mu_norm);
ptcbf_diag.max_velocity_norm = max(ptcbf_diag.max_velocity_norm, ...
    info.velocity_norm);
ptcbf_diag.max_grad_norm = max(ptcbf_diag.max_grad_norm, info.grad_norm);
if info.active
    ptcbf_diag.min_active_grad_norm = min( ...
        ptcbf_diag.min_active_grad_norm, info.grad_norm);
    ptcbf_diag.max_abs_scale = max(ptcbf_diag.max_abs_scale, ...
        abs(info.scale));
end
if isfield(info, 'slack')
    ptcbf_diag.max_slack = max(ptcbf_diag.max_slack, info.slack);
    ptcbf_diag.sum_slack = ptcbf_diag.sum_slack + info.slack;
    ptcbf_diag.slack_count = ptcbf_diag.slack_count + 1;
end
ptcbf_diag.max_uncertainty_t_abs = max( ...
    ptcbf_diag.max_uncertainty_t_abs, abs(info.uncertainty_t));
if info.mu_norm > eps
    ptcbf_diag.max_correction_to_mu_ratio = max( ...
        ptcbf_diag.max_correction_to_mu_ratio, ...
        info.correction_norm / info.mu_norm);
end
end

function print_ptcbf_diagnostics(ptcbf_diag)
if ptcbf_diag.call_count == 0
    return;
end
active_ratio = ptcbf_diag.active_count / ptcbf_diag.call_count;
disp(['PTCBF diagnostics call count: ', ...
    num2str(ptcbf_diag.call_count)]);
disp(['PTCBF active ratio: ', num2str(active_ratio)]);
disp(['PTCBF max uncertainty: ', ...
    num2str(ptcbf_diag.max_uncertainty)]);
disp(['PTCBF min h threshold margin: ', num2str(ptcbf_diag.min_h)]);
disp(['PTCBF rhs min/max: ', num2str(ptcbf_diag.min_rhs), ' / ', ...
    num2str(ptcbf_diag.max_rhs)]);
disp(['PTCBF max correction norm: ', ...
    num2str(ptcbf_diag.max_correction_norm)]);
disp(['PTCBF max GP mean velocity norm: ', ...
    num2str(ptcbf_diag.max_mu_norm)]);
disp(['PTCBF max corrected velocity norm: ', ...
    num2str(ptcbf_diag.max_velocity_norm)]);
disp(['PTCBF max uncertainty grad norm: ', ...
    num2str(ptcbf_diag.max_grad_norm)]);
disp(['PTCBF min active uncertainty grad norm: ', ...
    num2str(ptcbf_diag.min_active_grad_norm)]);
disp(['PTCBF max abs correction scale: ', ...
    num2str(ptcbf_diag.max_abs_scale)]);
if ptcbf_diag.slack_count > 0
    disp(['PTCBF slack mean/max: ', ...
        num2str(ptcbf_diag.sum_slack / ptcbf_diag.slack_count), ...
        ' / ', num2str(ptcbf_diag.max_slack)]);
end
disp(['PTCBF max |uncertainty_t|: ', ...
    num2str(ptcbf_diag.max_uncertainty_t_abs)]);
disp(['PTCBF max correction / mean velocity ratio: ', ...
    num2str(ptcbf_diag.max_correction_to_mu_ratio)]);
disp(['PTCBF max correction location sample/step/stage: ', ...
    num2str(ptcbf_diag.max_correction_sample_idx), ' / ', ...
    num2str(ptcbf_diag.max_correction_step_idx), ' / ', ...
    ptcbf_diag.max_correction_stage]);
disp(['PTCBF max correction details uncertainty/h/rhs/grad/scale: ', ...
    num2str(ptcbf_diag.max_correction_uncertainty), ' / ', ...
    num2str(ptcbf_diag.max_correction_h), ' / ', ...
    num2str(ptcbf_diag.max_correction_rhs), ' / ', ...
    num2str(ptcbf_diag.max_correction_grad_norm), ' / ', ...
    num2str(ptcbf_diag.max_correction_scale)]);
end
