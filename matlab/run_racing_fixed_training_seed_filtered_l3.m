function run_racing_fixed_training_seed_filtered_l3()
% Reuse the exact archived L1/L2 rollout caches, then refit and rerun only L3
% with a dedicated FM training-data seed and the normal variance seed filter.
root = 'C:\Users\JieJi\BA\matlab';
out = fullfile(root, 'outputs');
archive_path = fullfile(out, '赛车baseline总诊断文件.mat');
archive = load(archive_path, 'saved_cfg', 'cache_manifest');
cfg = archive.saved_cfg;
assert(strcmpi(cfg.scenario, 'racing'));
archived_cfg = cfg;

tag = 'Racing_FixedTrainingSeed_FilteredL3';
cfg.third_level_data_seed = cfg.second_level_fit_seed;
cfg.third_level_seed_filter.enabled = true;
cfg.third_level_seed_filter = rmfield_if_present( ...
    cfg.third_level_seed_filter, 'fixed_seed_matrix');
% Do not replace any archived algorithm parameter. The old saved_cfg is the
% single source of truth; only the previously missing dedicated L3 data seed
% is added here.
assert(cfg.third_level_seed_filter.enabled, ...
    'The archived baseline did not enable the L3 seed filter.');

% These are the exact archived L1/L2 caches verified against the old run.
cfg.cache.first_level_rollout_path = fullfile('outputs', ...
    '赛车baseline一层诊断缓存.mat');
cfg.cache.second_level_rollout_path = fullfile('outputs', ...
    '赛车baseline二层诊断缓存.mat');
% Keep the archived saved_cfg model and hyperparameter paths unchanged.

% Give the new deterministic L3 model/rollout isolated names.
cfg.cache.third_level_model_path = fullfile('outputs', ...
    [tag, '_ThirdLevel_Model.mat']);
cfg.cache.third_level_hyperparameter_path = fullfile('outputs', ...
    [tag, '_ThirdLevel_Hyperparameter.mat']);
cfg.cache.third_level_rollout_path = fullfile('outputs', ...
    [tag, '_ThirdLevel_Rollout.mat']);
copy_third_hyperparameters(out, tag, archive.cache_manifest);

cfg.output.experiment_prefix = tag;
cfg.output.live_second_level_rk4_video_enabled = false;
cfg.output.live_third_level_rk4_video_enabled = false;
cfg.output.plot_first_level_joint_softmin_safe_set = false;
cfg.output.plot_pchip_vs_raw_track = false;
cfg.animation.enabled = false;
cfg.safe_flow_evaluation.enabled = true;
cfg.safe_flow_evaluation.output_path = fullfile('outputs', ...
    [tag, '_Metrics_Variance_U.mat']);

% Force only the deterministic L3 fit and rollout to be rebuilt.
delete_if_present(fullfile(out, ...
    [tag, '_ThirdLevel_Model_EndpointWindowARDLengthScaleV9.mat']));
delete_if_present(fullfile(root, cfg.cache.third_level_rollout_path));
delete_if_present(fullfile(root, cfg.safe_flow_evaluation.output_path));
delete_if_present(fullfile(out, [tag, '_ThirdLevel_Segment_Seeds.mat']));
delete_if_present(fullfile(out, [tag, '_ThirdLevel_Segment_Seeds.csv']));

main_text = fileread(fullfile(root, 'main_demo.m'));
main_text = regexprep(main_text, '^clear;\s*clc;\s*', '', 'once');
main_text = strrep(main_text, 'cfg = get_config();', ...
    '% cfg supplied by fixed-L3 runner');
main_text = strrep(main_text, 'first_level_accepted_trajectory_seeds', ...
    [tag, '_FirstLevel_Accepted_Seeds']);
main_text = strrep(main_text, 'second_level_accepted_trajectory_seeds', ...
    [tag, '_SecondLevel_Accepted_Seeds']);
main_text = strrep(main_text, 'third_level_segment_seeds', ...
    [tag, '_ThirdLevel_Segment_Seeds']);

fprintf('\n===== Fixed-training-seed filtered L3 racing run =====\n');
fprintf('Reusing exact archived L1/L2 rollout caches.\n');
fprintf('Third-level FM training-data seed: %d\n', cfg.third_level_data_seed);
fprintf('Third-level seed filter enabled: %d\n', ...
    cfg.third_level_seed_filter.enabled);
cd(root);
eval(main_text);

run_manifest = struct();
run_manifest.archive_path = archive_path;
run_manifest.completed_at = datetime('now');
run_manifest.cfg = cfg;
run_manifest.archived_cfg = archived_cfg;
run_manifest.third_level_data_seed = cfg.third_level_data_seed;
run_manifest.third_level_seed_filter_enabled = ...
    cfg.third_level_seed_filter.enabled;
save(fullfile(out, [tag, '_RunManifest.mat']), 'run_manifest', '-v7.3');

result = load(fullfile(root, cfg.cache.third_level_rollout_path), ...
    'third_level_seed_acceptance');
acceptance = result.third_level_seed_acceptance;
assert(acceptance.enabled, 'Third-level seed filter was not enabled.');
assert(~isfield(acceptance, 'fixed_seed_replay') || ...
    ~acceptance.fixed_seed_replay, 'Unexpected fixed-seed replay mode.');
assert(max(acceptance.final_max_excess, [], 'all') <= ...
    acceptance.tolerance, ...
    'A retained L3 segment violates the hard variance cap.');
fprintf('VERIFIED: retained L3 segments satisfy the hard variance cap.\n');
end

function copy_third_hyperparameters(out, tag, manifest)
copyfile(manifest.hyperparameter_paths{3}, fullfile(out, ...
    [tag, '_ThirdLevel_Hyperparameter.mat']));
copyfile(manifest.hyperparameter_paths{4}, fullfile(out, ...
    [tag, '_ThirdLevel_Hyperparameter_EndpointARDStartWindowV3.mat']));
copyfile(manifest.hyperparameter_paths{5}, fullfile(out, ...
    [tag, '_ThirdLevel_Hyperparameter_EndpointARDEndWindowV3.mat']));
end

function value = rmfield_if_present(value, name)
if isfield(value, name)
    value = rmfield(value, name);
end
end

function delete_if_present(path)
if isfile(path)
    delete(path);
end
end
