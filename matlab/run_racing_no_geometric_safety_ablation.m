function run_racing_no_geometric_safety_ablation()
% Racing baseline ablation: disable both obstacle and track-boundary CBFs
% at all three levels. Variance constraints, variance seed filters, joint
% soft-min configuration, time grids, models and all other settings are
% inherited from the archived racing baseline.

root = 'C:\Users\JieJi\BA\matlab';
out = fullfile(root, 'outputs');
archive_path = fullfile(out, [char([36187 36710]), 'baseline'], ...
    'diagnostics', 'Racing_SafeFlow_Metrics_Variance_U.mat');
a = load(archive_path, 'saved_cfg', 'seed_bundle');
baseline_cfg = a.saved_cfg;
baseline_seed_bundle = a.seed_bundle;
cfg = baseline_cfg;
tag = 'Racing_NoGeometricSafety';

assert(strcmpi(cfg.scenario, 'racing'));
levels = {'first_level', 'second_level', 'third_level'};
for k = 1:numel(levels)
    p = levels{k};
    cfg.variance_constraint.([p '_obstacle_enabled']) = false;
    cfg.variance_constraint.([p '_track_boundary_enabled']) = false;
end

% Output identity changes are required only to protect the archived baseline.
cfg.output.experiment_prefix = tag;
cfg.cache.first_level_rollout_path = fullfile('outputs', ...
    [tag '_FirstLevel_Rollout.mat']);
cfg.cache.second_level_rollout_path = fullfile('outputs', ...
    [tag '_SecondLevel_Rollout.mat']);
cfg.cache.third_level_rollout_path = fullfile('outputs', ...
    [tag '_ThirdLevel_Rollout.mat']);
cfg.safe_flow_evaluation.output_path = fullfile('outputs', ...
    [tag '_Metrics_Variance_U.mat']);

verify_only_geometric_safety_disabled(cfg, baseline_cfg, levels);

protected_outputs = { ...
    fullfile(root, cfg.cache.first_level_rollout_path), ...
    fullfile(root, cfg.cache.second_level_rollout_path), ...
    fullfile(root, cfg.cache.third_level_rollout_path), ...
    fullfile(root, cfg.safe_flow_evaluation.output_path), ...
    fullfile(out, [tag '_Config.mat'])};
for k = 1:numel(protected_outputs)
    assert(~isfile(protected_outputs{k}), ...
        'Dedicated output already exists; refusing to overwrite: %s', ...
        protected_outputs{k});
end

obstacle_ablation_config = cfg;
save(fullfile(out, [tag '_Config.mat']), ...
    'obstacle_ablation_config', 'baseline_cfg', ...
    'baseline_seed_bundle', 'archive_path', '-v7.3');

main_text = fileread(fullfile(root, 'main_demo.m'));
main_text = regexprep(main_text, '^clear;\s*clc;\s*', '', 'once');
main_text = strrep(main_text, 'cfg = get_config();', ...
    '% cfg supplied by run_racing_no_geometric_safety_ablation');
main_text = strrep(main_text, 'first_level_accepted_trajectory_seeds', ...
    [tag '_FirstLevel_Seeds']);
main_text = strrep(main_text, 'second_level_accepted_trajectory_seeds', ...
    [tag '_SecondLevel_Seeds']);
main_text = strrep(main_text, 'third_level_segment_seeds', ...
    [tag '_ThirdLevel_Segment_Seeds']);

fprintf('\n===== Racing ablation: obstacle + track boundary OFF =====\n');
fprintf(['Variance constraints and all three variance seed filters retain ', ...
    'the archived baseline settings.\n']);
for k = 1:numel(levels)
    p = levels{k};
    fprintf('%s: obstacle=%d, track=%d, variance HOCBF=%d, variance PTCBF=%d, seed filter=%d\n', ...
        p, cfg.variance_constraint.([p '_obstacle_enabled']), ...
        cfg.variance_constraint.([p '_track_boundary_enabled']), ...
        cfg.variance_constraint.([p '_hocbf_enabled']), ...
        cfg.variance_constraint.([p '_ptcbf_enabled']), ...
        cfg.([p '_seed_filter']).enabled);
end
cd(root);
eval(main_text);
fprintf('RACING NO-GEOMETRIC-SAFETY ABLATION COMPLETE.\n');
end

function verify_only_geometric_safety_disabled(cfg, baseline_cfg, levels)
for k = 1:numel(levels)
    p = levels{k};
    assert(baseline_cfg.variance_constraint.([p '_obstacle_enabled']));
    assert(~cfg.variance_constraint.([p '_obstacle_enabled']));
    assert(baseline_cfg.variance_constraint.([p '_track_boundary_enabled']));
    assert(~cfg.variance_constraint.([p '_track_boundary_enabled']));
    fields = {'hocbf_enabled', 'ptcbf_enabled', ...
        'joint_safety_softmin_enabled'};
    for j = 1:numel(fields)
        f = [p '_' fields{j}];
        if isfield(baseline_cfg.variance_constraint, f)
            assert(isequaln(cfg.variance_constraint.(f), ...
                baseline_cfg.variance_constraint.(f)));
        end
    end
    sf = [p '_seed_filter'];
    assert(isequaln(cfg.(sf), baseline_cfg.(sf)));
end

% Validate the complete algorithm config by applying the allowed output
% identity edits to a reference copy and requiring exact structural equality.
expected = baseline_cfg;
for k = 1:numel(levels)
    p = levels{k};
    expected.variance_constraint.([p '_obstacle_enabled']) = false;
    expected.variance_constraint.([p '_track_boundary_enabled']) = false;
end
expected.output.experiment_prefix = cfg.output.experiment_prefix;
expected.cache.first_level_rollout_path = cfg.cache.first_level_rollout_path;
expected.cache.second_level_rollout_path = cfg.cache.second_level_rollout_path;
expected.cache.third_level_rollout_path = cfg.cache.third_level_rollout_path;
expected.safe_flow_evaluation.output_path = ...
    cfg.safe_flow_evaluation.output_path;
assert(isequaln(cfg, expected), ...
    'A non-obstacle, non-output configuration field changed.');
end
