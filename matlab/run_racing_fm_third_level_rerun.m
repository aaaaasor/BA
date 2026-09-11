function run_racing_fm_third_level_rerun()
% Racing FM comparison: retain hierarchical endpoint PTCLF/snap only.
% All GP-variance and geometric-safety guidance is disabled.  The archived
% Ours seeds are replayed once per sample/segment without rejection.

root = 'C:\Users\JieJi\BA\matlab';
out = fullfile(root, 'outputs');
archive_dir = fullfile(out, '赛车baseline');
archive_path = fullfile(archive_dir, ...
    'diagnostics', 'Racing_SafeFlow_Metrics_Variance_U.mat');
a = load(archive_path, 'saved_cfg', 'seed_bundle');
baseline_cfg = a.saved_cfg;
cfg = baseline_cfg;
baseline_seed_bundle = a.seed_bundle;
tag = 'Racing_FM';

assert(strcmpi(cfg.scenario, 'racing'));
assert(isequal(size(baseline_seed_bundle.first_level_seeds), [100 1]));
assert(isequal(size(baseline_seed_bundle.second_level_seeds), [100 4]));
assert(isequal(size(baseline_seed_bundle.third_level_seeds), [100 16]));

% Identical model, hyperparameters, time grids, sample count and seed values.
% A fixed seed matrix in an enabled=false filter means deterministic replay
% only: each seed runs exactly once and no result is rejected or retried.
cfg.first_level_seed_filter.enabled = false;
cfg.second_level_seed_filter.enabled = false;
cfg.third_level_seed_filter.enabled = false;
cfg.first_level_seed_filter.fixed_seed_matrix = ...
    baseline_seed_bundle.first_level_seeds;
cfg.second_level_seed_filter.fixed_seed_matrix = ...
    baseline_seed_bundle.second_level_seeds;
cfg.third_level_seed_filter.fixed_seed_matrix = ...
    baseline_seed_bundle.third_level_seeds;

levels = {'first_level', 'second_level', 'third_level'};
for k = 1:numel(levels)
    p = levels{k};
    cfg.variance_constraint.([p '_hocbf_enabled']) = false;
    cfg.variance_constraint.([p '_ptcbf_enabled']) = false;
    cfg.variance_constraint.([p '_obstacle_enabled']) = false;
    cfg.variance_constraint.([p '_track_boundary_enabled']) = false;
    cfg.variance_constraint.([p '_joint_safety_softmin_enabled']) = false;
end

% Keep only the common hierarchical connection mechanism.
cfg.variance_constraint.first_level_ptclf_enabled = false;
cfg.variance_constraint.second_level_ptclf_enabled = true;
cfg.variance_constraint.third_level_ptclf_enabled = true;
cfg.variance_constraint.second_level_anchor_clf_ptzf_enabled = true;
cfg.variance_constraint.third_level_anchor_clf_ptzf_enabled = true;
assert(cfg.variance_constraint.second_level_anchor_snap_flow_steps > 0);
assert(cfg.variance_constraint.third_level_anchor_snap_flow_steps > 0);

% Terminal safety/feasibility correction remains disabled.  The endpoint
% snap-and-hold above is retained because it is the requested common
% hierarchy connection rule, not a terminal safety projection.
cfg.variance_constraint.terminal_safety_filter_enabled = false;
cfg.variance_constraint.third_level_terminal_feasibility_qp_enabled = false;
cfg.variance_constraint.third_level_post_endpoint_overwrite_enabled = false;

% Disable auxiliary comparison reruns and expensive mechanism-only plots.
cfg.first_level_run_no_obstacle_baseline = false;
cfg.second_level_run_no_obstacle_baseline = false;
cfg.third_level_run_no_obstacle_baseline = false;
cfg.third_level_variance_violation_diagnostic_enabled = false;
cfg.third_level_cbf_trace_plot_enabled = false;
cfg.output.live_second_level_rk4_video_enabled = false;
cfg.output.live_third_level_rk4_video_enabled = false;
cfg.animation.enabled = false;

cfg.output.experiment_prefix = tag;
cfg.cache.first_level_rollout_path = fullfile('outputs', ...
    [tag '_FirstLevel_Rollout.mat']);
cfg.cache.second_level_rollout_path = fullfile('outputs', ...
    [tag '_SecondLevel_Rollout.mat']);
cfg.cache.third_level_rollout_path = fullfile('outputs', ...
    [tag '_ThirdLevel_Rollout.mat']);
cfg.safe_flow_evaluation.enabled = true;
cfg.safe_flow_evaluation.output_path = fullfile('outputs', ...
    [tag '_Metrics_Variance_U.mat']);

% Reuse the already verified first/second-level caches. Rebuild only the
% third-level rollout and the final metric file for a clean timing repeat.
assert(isfile(fullfile(root, cfg.cache.first_level_rollout_path)), ...
    'Verified first-level cache is missing.');
assert(isfile(fullfile(root, cfg.cache.second_level_rollout_path)), ...
    'Verified second-level cache is missing.');
delete_if_present(fullfile(root, cfg.cache.third_level_rollout_path));
delete_if_present(fullfile(root, cfg.safe_flow_evaluation.output_path));

fm_config = cfg;
save(fullfile(out, [tag '_Config.mat']), 'fm_config', ...
    'baseline_cfg', 'baseline_seed_bundle', 'archive_path', '-v7.3');

verify_fm_switches(cfg, baseline_cfg, baseline_seed_bundle, tag);
main_text = fileread(fullfile(root, 'main_demo.m'));
main_text = regexprep(main_text, '^clear;\s*clc;\s*', '', 'once');
main_text = strrep(main_text, 'cfg = get_config();', ...
    '% cfg supplied by run_racing_fm_baseline');
main_text = strrep(main_text, 'first_level_accepted_trajectory_seeds', ...
    [tag '_FirstLevel_Seeds']);
main_text = strrep(main_text, 'second_level_accepted_trajectory_seeds', ...
    [tag '_SecondLevel_Seeds']);
main_text = strrep(main_text, 'third_level_segment_seeds', ...
    [tag '_ThirdLevel_Segment_Seeds']);

fprintf('\n===== Racing FM third-level timing rerun: same config/seeds =====\n');
fprintf('Archived seed sizes: L1=%s, L2=%s, L3=%s\n', ...
    mat2str(size(baseline_seed_bundle.first_level_seeds)), ...
    mat2str(size(baseline_seed_bundle.second_level_seeds)), ...
    mat2str(size(baseline_seed_bundle.third_level_seeds)));
cd(root);
eval(main_text);

verify_saved_fm_seeds(out, tag, baseline_seed_bundle);
fprintf('RACING FM THIRD-LEVEL RERUN COMPLETE AND SEEDS VERIFIED.\n');
end

function verify_fm_switches(cfg, baseline_cfg, seeds, tag)
levels = {'first_level', 'second_level', 'third_level'};
for k = 1:numel(levels)
    p = levels{k};
    assert(~cfg.variance_constraint.([p '_hocbf_enabled']));
    assert(~cfg.variance_constraint.([p '_ptcbf_enabled']));
    assert(~cfg.variance_constraint.([p '_obstacle_enabled']));
    assert(~cfg.variance_constraint.([p '_track_boundary_enabled']));
end
assert(~cfg.first_level_seed_filter.enabled);
assert(~cfg.second_level_seed_filter.enabled);
assert(~cfg.third_level_seed_filter.enabled);
assert(cfg.variance_constraint.second_level_ptclf_enabled);
assert(cfg.variance_constraint.third_level_ptclf_enabled);
assert(~cfg.variance_constraint.terminal_safety_filter_enabled);
assert(~cfg.variance_constraint.third_level_terminal_feasibility_qp_enabled);

% Require structural equality with the archived baseline after applying only
% the requested constraint/seed-replay changes and non-algorithm output
% identity/diagnostic changes.  This prevents an unnoticed tuning change
% from entering the ablation.
expected = baseline_cfg;
expected.first_level_seed_filter.enabled = false;
expected.second_level_seed_filter.enabled = false;
expected.third_level_seed_filter.enabled = false;
expected.first_level_seed_filter.fixed_seed_matrix = seeds.first_level_seeds;
expected.second_level_seed_filter.fixed_seed_matrix = seeds.second_level_seeds;
expected.third_level_seed_filter.fixed_seed_matrix = seeds.third_level_seeds;
for k = 1:numel(levels)
    p = levels{k};
    expected.variance_constraint.([p '_hocbf_enabled']) = false;
    expected.variance_constraint.([p '_ptcbf_enabled']) = false;
    expected.variance_constraint.([p '_obstacle_enabled']) = false;
    expected.variance_constraint.([p '_track_boundary_enabled']) = false;
    expected.variance_constraint.([p '_joint_safety_softmin_enabled']) = false;
end
expected.variance_constraint.first_level_ptclf_enabled = false;
expected.variance_constraint.second_level_ptclf_enabled = true;
expected.variance_constraint.third_level_ptclf_enabled = true;
expected.variance_constraint.second_level_anchor_clf_ptzf_enabled = true;
expected.variance_constraint.third_level_anchor_clf_ptzf_enabled = true;
expected.variance_constraint.terminal_safety_filter_enabled = false;
expected.variance_constraint.third_level_terminal_feasibility_qp_enabled = false;
expected.variance_constraint.third_level_post_endpoint_overwrite_enabled = false;
expected.first_level_run_no_obstacle_baseline = false;
expected.second_level_run_no_obstacle_baseline = false;
expected.third_level_run_no_obstacle_baseline = false;
expected.third_level_variance_violation_diagnostic_enabled = false;
expected.third_level_cbf_trace_plot_enabled = false;
expected.output.live_second_level_rk4_video_enabled = false;
expected.output.live_third_level_rk4_video_enabled = false;
expected.animation.enabled = false;
expected.output.experiment_prefix = tag;
expected.cache.first_level_rollout_path = fullfile('outputs', ...
    [tag '_FirstLevel_Rollout.mat']);
expected.cache.second_level_rollout_path = fullfile('outputs', ...
    [tag '_SecondLevel_Rollout.mat']);
expected.cache.third_level_rollout_path = fullfile('outputs', ...
    [tag '_ThirdLevel_Rollout.mat']);
expected.safe_flow_evaluation.enabled = true;
expected.safe_flow_evaluation.output_path = fullfile('outputs', ...
    [tag '_Metrics_Variance_U.mat']);
assert(isequaln(cfg, expected), ...
    'A configuration field outside the declared ablation changed.');
end

function verify_saved_fm_seeds(out, tag, seeds)
l1 = load(fullfile(out,[tag '_FirstLevel_Rollout.mat']), ...
    'accepted_first_level_seeds','first_level_seed_acceptance');
l2 = load(fullfile(out,[tag '_SecondLevel_Rollout.mat']), ...
    'accepted_second_level_seeds','second_level_seed_acceptance');
l3 = load(fullfile(out,[tag '_ThirdLevel_Rollout.mat']), ...
    'accepted_third_level_seeds','third_level_seed_acceptance');
assert(~l1.first_level_seed_acceptance.enabled);
assert(~l2.second_level_seed_acceptance.enabled);
assert(~l3.third_level_seed_acceptance.enabled);
assert(isequal(l1.accepted_first_level_seeds, double(seeds.first_level_seeds)));
assert(isequal(l2.accepted_second_level_seeds, double(seeds.second_level_seeds)));
assert(isequal(l3.accepted_third_level_seeds, double(seeds.third_level_seeds)));
end

function delete_if_present(path)
if isfile(path), delete(path); end
end
