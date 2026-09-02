function run_racing_archived_baseline_current_code()
% Re-run the archived guided racing baseline with the current implementation
% while replaying every archived accepted seed exactly once.
root = 'C:\Users\JieJi\BA\matlab';
out = fullfile(root, 'outputs');
archive_path = fullfile(out, '赛车baseline总诊断文件.mat');
archive = load(archive_path, 'saved_cfg', 'seed_bundle', 'cache_manifest');
cfg = archive.saved_cfg;
seed_bundle = archive.seed_bundle;
assert(strcmpi(cfg.scenario, 'racing'));
assert(isequal(size(seed_bundle.first_level_seeds), [100 1]));
assert(isequal(size(seed_bundle.second_level_seeds), [100 4]));
assert(isequal(size(seed_bundle.third_level_seeds), [100 16]));

tag = 'Racing_ArchivedBaseline_CurrentCode';
cfg.output.experiment_prefix = tag;
cfg.output.live_second_level_rk4_video_enabled = false;
cfg.output.live_third_level_rk4_video_enabled = false;
cfg.output.plot_first_level_joint_softmin_safe_set = false;
cfg.output.plot_pchip_vs_raw_track = false;
cfg.animation.enabled = false;
cfg.safe_flow_evaluation.enabled = true;
cfg.safe_flow_evaluation.output_path = fullfile('outputs', ...
    [tag, '_Metrics_Variance_U.mat']);

cfg.first_level_seed_filter.fixed_seed_matrix = ...
    seed_bundle.first_level_seeds;
cfg.second_level_seed_filter.fixed_seed_matrix = ...
    seed_bundle.second_level_seeds;
cfg.third_level_seed_filter.fixed_seed_matrix = ...
    seed_bundle.third_level_seeds;

cfg.cache.first_level_rollout_path = fullfile('outputs', ...
    [tag, '_FirstLevel_Rollout.mat']);
cfg.cache.second_level_rollout_path = fullfile('outputs', ...
    [tag, '_SecondLevel_Rollout.mat']);
cfg.cache.third_level_rollout_path = fullfile('outputs', ...
    [tag, '_ThirdLevel_Rollout.mat']);

% Isolate model/hyperparameter caches so current cache validation can never
% modify the archived baseline artifacts.
cfg.cache.first_level_model_path = fullfile('outputs', ...
    [tag, '_FirstLevel_Model.mat']);
cfg.cache.second_level_model_path = fullfile('outputs', ...
    [tag, '_SecondLevel_Model.mat']);
cfg.cache.third_level_model_path = fullfile('outputs', ...
    [tag, '_ThirdLevel_Model.mat']);
cfg.cache.first_level_hyperparameter_path = fullfile('outputs', ...
    [tag, '_FirstLevel_Hyperparameter.mat']);
cfg.cache.second_level_hyperparameter_path = fullfile('outputs', ...
    [tag, '_SecondLevel_Hyperparameter.mat']);
cfg.cache.third_level_hyperparameter_path = fullfile('outputs', ...
    [tag, '_ThirdLevel_Hyperparameter.mat']);
prepare_isolated_caches(out, tag, archive.cache_manifest);

% Force a fresh rollout while leaving isolated fitted-model caches reusable.
delete_if_present(fullfile(root, cfg.cache.first_level_rollout_path));
delete_if_present(fullfile(root, cfg.cache.second_level_rollout_path));
delete_if_present(fullfile(root, cfg.cache.third_level_rollout_path));
delete_if_present(fullfile(root, cfg.safe_flow_evaluation.output_path));

main_text = fileread(fullfile(root, 'main_demo.m'));
main_text = regexprep(main_text, '^clear;\s*clc;\s*', '', 'once');
main_text = strrep(main_text, 'cfg = get_config();', ...
    '% cfg supplied by archived-baseline replay runner');
main_text = strrep(main_text, 'first_level_accepted_trajectory_seeds', ...
    [tag, '_FirstLevel_Accepted_Seeds']);
main_text = strrep(main_text, 'second_level_accepted_trajectory_seeds', ...
    [tag, '_SecondLevel_Accepted_Seeds']);
main_text = strrep(main_text, 'third_level_segment_seeds', ...
    [tag, '_ThirdLevel_Segment_Seeds']);

fprintf('\n===== Archived racing baseline replay on current code =====\n');
fprintf('Archive: %s\n', archive_path);
fprintf('Seeds: L1=%s, L2=%s, L3=%s\n', ...
    mat2str(size(seed_bundle.first_level_seeds)), ...
    mat2str(size(seed_bundle.second_level_seeds)), ...
    mat2str(size(seed_bundle.third_level_seeds)));
cd(root);
eval(main_text);

replay_manifest = struct();
replay_manifest.archive_path = archive_path;
replay_manifest.tag = tag;
replay_manifest.completed_at = datetime('now');
replay_manifest.archived_saved_cfg = archive.saved_cfg;
replay_manifest.replay_cfg = cfg;
replay_manifest.fixed_seed_bundle = seed_bundle;
replay_manifest.current_git_head = strtrim(system_text('git rev-parse HEAD'));
replay_manifest.current_git_status = system_text('git status --short');
save(fullfile(out, [tag, '_ReplayManifest.mat']), 'replay_manifest', '-v7.3');
verify_replay_outputs(out, tag, seed_bundle);
fprintf('ARCHIVED BASELINE CURRENT-CODE REPLAY COMPLETE.\n');
end

function prepare_isolated_caches(out, tag, manifest)
copy_if_needed(manifest.model_paths{1}, ...
    fullfile(out, [tag, '_FirstLevel_Model.mat']));
copy_if_needed(manifest.model_paths{2}, ...
    fullfile(out, [tag, '_SecondLevel_Model.mat']));
copy_if_needed(manifest.model_paths{3}, ...
    fullfile(out, [tag, '_ThirdLevel_Model_EndpointWindowARDLengthScaleV9.mat']));
copy_if_needed(manifest.hyperparameter_paths{1}, ...
    fullfile(out, [tag, '_FirstLevel_Hyperparameter.mat']));
copy_if_needed(manifest.hyperparameter_paths{2}, ...
    fullfile(out, [tag, '_SecondLevel_Hyperparameter.mat']));
copy_if_needed(manifest.hyperparameter_paths{3}, ...
    fullfile(out, [tag, '_ThirdLevel_Hyperparameter.mat']));
copy_if_needed(manifest.hyperparameter_paths{4}, ...
    fullfile(out, [tag, ...
    '_ThirdLevel_Hyperparameter_EndpointARDStartWindowV3.mat']));
copy_if_needed(manifest.hyperparameter_paths{5}, ...
    fullfile(out, [tag, ...
    '_ThirdLevel_Hyperparameter_EndpointARDEndWindowV3.mat']));
end

function copy_if_needed(source, destination)
if ~isfile(destination)
    fprintf('Copying isolated cache: %s\n', destination);
    copyfile(source, destination);
end
end

function delete_if_present(path)
if isfile(path)
    delete(path);
end
end

function verify_replay_outputs(out, tag, archived_seeds)
l1 = load(fullfile(out, [tag, '_FirstLevel_Rollout.mat']), ...
    'accepted_first_level_seeds');
l2 = load(fullfile(out, [tag, '_SecondLevel_Rollout.mat']), ...
    'accepted_second_level_seeds');
l3 = load(fullfile(out, [tag, '_ThirdLevel_Rollout.mat']), ...
    'accepted_third_level_seeds');
assert(isequal(l1.accepted_first_level_seeds, ...
    archived_seeds.first_level_seeds));
assert(isequal(l2.accepted_second_level_seeds, ...
    archived_seeds.second_level_seeds));
assert(isequal(l3.accepted_third_level_seeds, ...
    archived_seeds.third_level_seeds));
fprintf('VERIFIED: all archived L1/L2/L3 seeds were replayed exactly.\n');
end

function value = system_text(command)
[status, value] = system(command);
if status ~= 0
    value = sprintf('command failed with status %d', status);
end
end
