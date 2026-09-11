function exp_step_refine_ek(step_list)
%EXP_STEP_REFINE_EK  Grid-refinement test for the origin of the terminal
% h-undershoot in the no-variance level-2 ablation.
%
% Re-runs the SAME level-2 rollout (same config, same constraint, same
% x_init, u_per_stage = false) on 100 / 200 / 400 RK4 steps and measures
%
%     e_k := h_k * exp(-phi0*dt) - h_{k+1}          (positive = undershoot)
%
% i.e. how far each step falls below what the CBF inequality
% hdot >= -phi0*h allows.  Scaling of e_k with dt identifies the source:
%
%     RK4 truncation  O(dt^5)  -> 32x per halving
%     frozen-u (ZOH)  O(dt^2)  ->  4x per halving
%     first-order leak O(dt)   ->  2x per halving

if nargin < 1 || isempty(step_list), step_list = [100 200 400]; end
root = 'C:\Users\JieJi\BA\matlab';
cd(root);

C = load(fullfile('outputs','Racing_NoVariance_L2T0999_Omega05_Config.mat'));
cfg = C.test_config;
t1  = C.second_level_rollout_t_max;
R = load(fullfile('outputs','Racing_NoVariance_L2T0999_Omega05_SecondLevel_Rollout.mat'), ...
    'saved_segment_rollout_constraint','segment_x_init','segment_traj_path_10d', ...
    'segment_rollout_times');
cst  = R.saved_segment_rollout_constraint;
x0   = R.segment_x_init;

M = load(fullfile(root, cfg.cache.second_level_model_path));
mf = fieldnames(M);
model_collection = [];
for i = 1:numel(mf)
    v = M.(mf{i});
    if isstruct(v) && isfield(v, 'model'), model_collection = v; break; end
end
assert(~isempty(model_collection), 'model_collection not found in %s', ...
    cfg.cache.second_level_model_path);
model_collection = strip_model_for_prediction(model_collection);

assert(~struct_field_default(cst, 'u_per_stage', false), ...
    'u_per_stage must be false for this test.');
phi0 = struct_field_default(cst, 'joint_safety_phi0', 15.0);
fprintf('phi0 = %g   t1 = %.6f   n_samples = %d\n', phi0, t1, size(x0,1));

res = struct([]);
for si = 1:numel(step_list)
    ns = step_list(si);
    fprintf('\n===== n_steps = %d =====\n', ns);
    tt = tic;
    [times, path] = rk4_rollout(model_collection, x0, cfg.t_min, t1, ns, ...
        cst, [], cfg.parallel);
    fprintf('rollout %.1f s\n', toc(tt));
    if ns == 100
        d = max(abs(path(:) - R.segment_traj_path_10d(:)));
        fprintf('reproduces cached 100-step path: max|diff| = %.3e\n', d);
    end
    r = analyze(times, path, cst, phi0, 0.95);
    r.n_steps = ns;  r.dt = times(2) - times(1);
    if isempty(res), res = r; else, res(end+1) = r; end %#ok<AGROW>
end

fprintf('\n================ SUMMARY ================\n');
fprintf('%-8s %-10s %-12s %-12s %-12s %-8s %-12s\n', 'steps', 'dt', ...
    'e_max', 'e_p99', 'e_median+', 'n_viol', 'h_min_end');
for i = 1:numel(res)
    fprintf('%-8d %-10.5f %-12.4e %-12.4e %-12.4e %-8d %-12.4e\n', ...
        res(i).n_steps, res(i).dt, res(i).e_max, res(i).e_p99, ...
        res(i).e_med_pos, res(i).n_violating_terminal, res(i).h_min_terminal);
end
fprintf('\n--- shrink factors per halving of dt ---\n');
for i = 2:numel(res)
    fprintf('%d -> %d :  e_max %.2fx   e_p99 %.2fx   e_median+ %.2fx\n', ...
        res(i-1).n_steps, res(i).n_steps, ...
        res(i-1).e_max / max(res(i).e_max, realmin), ...
        res(i-1).e_p99 / max(res(i).e_p99, realmin), ...
        res(i-1).e_med_pos / max(res(i).e_med_pos, realmin));
end
fprintf('\nreference: O(dt^5)=32x  O(dt^2)=4x  O(dt)=2x\n');
save(fullfile('outputs','Exp_StepRefine_Ek.mat'), 'res', 'step_list');
end

% =====================================================================
function r = analyze(times, path, cst, phi0, t_window)
n_t = numel(times);
n_s = size(path, 2);
k0 = find(times >= t_window, 1, 'first');
if isempty(k0), k0 = n_t - 1; end
k0 = max(k0 - 1, 1);
dt = diff(times);

e_cell = cell(n_s, 1);
elast_cell = cell(n_s, 1);
hend_cell = cell(n_s, 1);
parfor s = 1:n_s
    c = per_sample_cfg(cst, s);
    hs = [];
    for k = k0:n_t
        x = reshape(path(k, s, :), [], 1);
        hv = eval_h(x, c, times(k));
        if isempty(hs), hs = nan(n_t - k0 + 1, numel(hv)); end
        if numel(hv) == size(hs, 2), hs(k - k0 + 1, :) = hv(:)'; end
    end
    dtw = dt(k0:end);
    allowed = hs(1:end-1, :) .* exp(-phi0 * dtw);
    e = allowed - hs(2:end, :);
    ok = isfinite(e) & hs(1:end-1, :) >= 0;
    e_cell{s} = e(ok);
    lastrow = e(end, :); lastok = ok(end, :);
    elast_cell{s} = lastrow(lastok)';
    hend_cell{s} = hs(end, :)';
end
e_all = vertcat(e_cell{:});
e_last = vertcat(elast_cell{:});
h_end = vertcat(hend_cell{:});
pos = e_all(e_all > 0);
r = struct( ...
    'e_max', max([e_all; -inf]), ...
    'e_p99', prctile(e_all, 99), ...
    'e_med_pos', median([pos; 0]), ...
    'n_pos', numel(pos), ...
    'e_last_max', max([e_last; -inf]), ...
    'e_last_med', median(e_last), ...
    'n_violating_terminal', sum(h_end < 0), ...
    'h_min_terminal', min(h_end));
end

function c = per_sample_cfg(cst, s)
c = cst;
if isfield(cst, 'anchor_clf_targets')
    c.anchor_clf_target = cst.anchor_clf_targets(s, :)';
end
if isfield(cst, 'track_boundary_reference_s_min_targets')
    c.track_boundary_reference_s_min = cst.track_boundary_reference_s_min_targets(s, :);
    c.track_boundary_reference_s_max = cst.track_boundary_reference_s_max_targets(s, :);
end
end

function h = eval_h(x, c, t)
stats = struct('x', x(:), 'mu', zeros(numel(x), 1), 'sigma2', 0);
oi = obstacle_cbf_info(stats, c, t);
bi = track_boundary_cbf_info(stats, c, t);
ji = joint_safety_softmin_info(oi, bi, stats, c, t);
h = ji.h_values;
end
