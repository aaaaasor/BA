function run_racing_filtered_baseline_current_code()
% Re-run the archived guided racing baseline with the current implementation
% using the archived configuration and base seeds, while applying the normal
% variance seed-filter acceptance/retry logic at all three levels.
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

tag = 'Racing_FilteredBaseline_CurrentCode';
cfg.output.experiment_prefix = tag;
cfg.output.live_second_level_rk4_video_enabled = false;
cfg.output.live_third_level_rk4_video_enabled = false;
cfg.output.plot_first_level_joint_softmin_safe_set = false;
cfg.output.plot_pchip_vs_raw_track = false;
cfg.animation.enabled = false;
cfg.safe_flow_evaluation.enabled = true;
cfg.safe_flow_evaluation.output_path = fullfile('outputs', ...
    [tag, '_Metrics_Variance_U.mat']);

% Do not set fixed_seed_matrix here. That replay-only mode intentionally keeps
% every supplied seed and therefore does not replace candidates that violate
% the hard variance acceptance conditions.
cfg.first_level_seed_filter = rmfield_if_present( ...
    cfg.first_level_seed_filter, 'fixed_seed_matrix');
cfg.second_level_seed_filter = rmfield_if_present( ...
    cfg.second_level_seed_filter, 'fixed_seed_matrix');
cfg.third_level_seed_filter = rmfield_if_present( ...
    cfg.third_level_seed_filter, 'fixed_seed_matrix');

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

fprintf('\n===== Filtered racing baseline on current code =====\n');
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
replay_manifest.archived_accepted_seed_bundle_for_comparison = seed_bundle;
replay_manifest.current_git_head = strtrim(system_text('git rev-parse HEAD'));
replay_manifest.current_git_status = system_text('git status --short');
save(fullfile(out, [tag, '_ReplayManifest.mat']), 'replay_manifest', '-v7.3');
verify_filtered_outputs(out, tag);
fprintf('FILTERED BASELINE CURRENT-CODE RUN COMPLETE.\n');
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

function verify_filtered_outputs(out, tag)
files = {[tag, '_FirstLevel_Rollout.mat'], ...
    [tag, '_SecondLevel_Rollout.mat'], ...
    [tag, '_ThirdLevel_Rollout.mat']};
names = {'first_level_seed_acceptance', ...
    'second_level_seed_acceptance', 'third_level_seed_acceptance'};
for level = 1:3
    loaded = load(fullfile(out, files{level}), names{level});
    acceptance = loaded.(names{level});
    assert(~isfield(acceptance, 'fixed_seed_replay') || ...
        ~acceptance.fixed_seed_replay, ...
        'Level %d unexpectedly used fixed-seed replay.', level);
    assert(max(acceptance.final_max_excess, [], 'all') <= ...
        acceptance.tolerance, ...
        'Level %d retained a hard variance-cap violation.', level);
end
fprintf('VERIFIED: all L1/L2/L3 accepted outputs satisfy the hard variance cap.\n');
end

function value = rmfield_if_present(value, name)
if isfield(value, name)
    value = rmfield(value, name);
end
end

function value = system_text(command)
[status, value] = system(command);
if status ~= 0
    value = sprintf('command failed with status %d', status);
end
end
