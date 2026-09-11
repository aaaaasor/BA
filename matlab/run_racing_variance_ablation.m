function run_racing_variance_ablation()
% Racing ablation: reproduce the archived baseline with GP-variance
% constraints removed. Geometric obstacle/track-boundary safety and the
% hierarchical PTCLF/endpoint-snap mechanism remain at baseline settings.

root = 'C:\Users\JieJi\BA\matlab';
out = fullfile(root, 'outputs');
% Keep the source ASCII-only so MATLAB reads it consistently on machines
% whose script encoding is not UTF-8. char([36187 36710]) is "racing car" in
% Chinese and selects the user-designated outputs/<racing-car>baseline archive.
baseline_dir = fullfile(out, [char([36187 36710]), 'baseline']);
archive_path = fullfile(baseline_dir, 'diagnostics', ...
    'Racing_SafeFlow_Metrics_Variance_U.mat');
archive = load(archive_path, 'saved_cfg', 'seed_bundle');
cfg = archive.saved_cfg;
baseline_seed_bundle = archive.seed_bundle;
tag = 'Racing_NoVariance';

assert(strcmpi(cfg.scenario, 'racing'));
assert(isequal(size(baseline_seed_bundle.first_level_seeds), [100 1]));
assert(isequal(size(baseline_seed_bundle.second_level_seeds), [100 4]));
assert(isequal(size(baseline_seed_bundle.third_level_seeds), [100 16]));

% Replay the exact archived seeds once. Leaving a variance-based seed filter
% enabled would reintroduce a variance constraint outside the rollout QP.
cfg.first_level_seed_filter.enabled = false;
cfg.second_level_seed_filter.enabled = false;
cfg.third_level_seed_filter.enabled = false;
cfg.first_level_seed_filter.fixed_seed_matrix = ...
    baseline_seed_bundle.first_level_seeds;
cfg.second_level_seed_filter.fixed_seed_matrix = ...
    baseline_seed_bundle.second_level_seeds;
cfg.third_level_seed_filter.fixed_seed_matrix = ...
    baseline_seed_bundle.third_level_seeds;

% The six intended ablation switches: integral variance HOCBF and terminal
% variance PTCBF at all three hierarchy levels.
levels = {'first_level', 'second_level', 'third_level'};
for k = 1:numel(levels)
    level = levels{k};
    cfg.variance_constraint.([level '_hocbf_enabled']) = false;
    cfg.variance_constraint.([level '_ptcbf_enabled']) = false;
end

% These are diagnostics of the removed variance mechanisms, not constraints.
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

% Use dedicated output names. Existing baseline caches are never deleted or
% overwritten; only an interrupted run of this same ablation is replaced.
delete_if_present(fullfile(root, cfg.cache.first_level_rollout_path));
delete_if_present(fullfile(root, cfg.cache.second_level_rollout_path));
delete_if_present(fullfile(root, cfg.cache.third_level_rollout_path));
delete_if_present(fullfile(root, cfg.safe_flow_evaluation.output_path));

variance_ablation_config = cfg;
save(fullfile(out, [tag '_Config.mat']), 'variance_ablation_config', ...
    'baseline_seed_bundle', 'archive_path', '-v7.3');

verify_ablation_switches(cfg, archive.saved_cfg);
main_text = fileread(fullfile(root, 'main_demo.m'));
main_text = regexprep(main_text, '^clear;\s*clc;\s*', '', 'once');
main_text = strrep(main_text, 'cfg = get_config();', ...
    '% cfg supplied by run_racing_variance_ablation');
main_text = strrep(main_text, 'first_level_accepted_trajectory_seeds', ...
    [tag '_FirstLevel_Seeds']);
main_text = strrep(main_text, 'second_level_accepted_trajectory_seeds', ...
    [tag '_SecondLevel_Seeds']);
main_text = strrep(main_text, 'third_level_segment_seeds', ...
    [tag '_ThirdLevel_Segment_Seeds']);

fprintf('\n===== Racing ablation: all GP-variance constraints OFF =====\n');
fprintf('Exact archived baseline config and seeds; geometric safety retained.\n');
cd(root);
eval(main_text);

verify_saved_seeds(out, tag, baseline_seed_bundle);
fprintf('RACING NO-VARIANCE ABLATION COMPLETE AND SEEDS VERIFIED.\n');
end

function verify_ablation_switches(cfg, baseline_cfg)
levels = {'first_level', 'second_level', 'third_level'};
for k = 1:numel(levels)
    level = levels{k};
    assert(~cfg.variance_constraint.([level '_hocbf_enabled']));
    assert(~cfg.variance_constraint.([level '_ptcbf_enabled']));
    % Geometric safety must be identical to the archived baseline.
    assert(isequaln(cfg.variance_constraint.([level '_obstacle_enabled']), ...
        baseline_cfg.variance_constraint.([level '_obstacle_enabled'])));
    assert(isequaln(cfg.variance_constraint.([level '_track_boundary_enabled']), ...
        baseline_cfg.variance_constraint.([level '_track_boundary_enabled'])));
    joint_field = [level '_joint_safety_softmin_enabled'];
    if isfield(baseline_cfg.variance_constraint, joint_field)
        assert(isequaln(cfg.variance_constraint.(joint_field), ...
            baseline_cfg.variance_constraint.(joint_field)));
    end
end
assert(~cfg.first_level_seed_filter.enabled);
assert(~cfg.second_level_seed_filter.enabled);
assert(~cfg.third_level_seed_filter.enabled);
end

function verify_saved_seeds(out, tag, seeds)
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
