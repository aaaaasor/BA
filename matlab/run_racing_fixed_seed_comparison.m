function run_racing_fixed_seed_comparison()
root = 'C:\Users\JieJi\BA\matlab';
addpath(root); cd(root);
main_text = fileread(fullfile(root, 'main_demo.m'));
main_text = regexprep(main_text, '^clear;\s*clc;\s*', '', 'once');
main_text = strrep(main_text, 'cfg = get_config();', '% cfg supplied by fixed-seed comparison runner');
needle = '    third_segment_gp = cfg.gp;';
hook = sprintf(['    verify_fixed_training_data(cfg, third_segment_s_slices, ...\n', ...
 '        third_segment_x_slices, third_segment_y_slices, this_dir);\n', ...
 '    third_segment_gp = cfg.gp;']);
assert(contains(main_text, needle));
main_text = strrep(main_text, needle, hook);

out = fullfile(root, 'outputs');
partial_old = load(fullfile(out, '赛车无guidance部分数据总诊断文件.mat'), 'saved_cfg');
full_old = load(fullfile(out, '赛车无guidance所有数据总诊断文件.mat'), 'saved_cfg');
seed = partial_old.saved_cfg.second_level_fit_seed;
assert(seed == full_old.saved_cfg.second_level_fit_seed);
run_one(partial_old.saved_cfg, false, seed, main_text, root);
run_one(full_old.saved_cfg, true, seed, main_text, root);
publish_results(root, seed);
end

function run_one(cfg, all_data, seed, main_text, root)
tag = ternary(all_data, 'AllData', 'Partial');
prefix = ['Racing_FixedSeed_', tag];
fprintf('\n===== %s: fixed third-level FM data seed %d =====\n', tag, seed);
assert(strcmpi(cfg.scenario, 'racing'));
cfg.third_level_data_seed = seed;
cfg.comparison.unguided_partial_gp_enabled = true;
cfg.comparison.all_training_points_enabled = all_data;
cfg.gp.training_point_selection_enabled = ~all_data;
cfg.output.experiment_prefix = prefix;
cfg.output.live_second_level_rk4_video_enabled = false;
cfg.output.live_third_level_rk4_video_enabled = false;
cfg.output.plot_first_level_joint_softmin_safe_set = false;
cfg.output.plot_pchip_vs_raw_track = false;
cfg.animation.enabled = false;
cfg.cache.first_level_rollout_path = fullfile('outputs', [prefix, '_FirstLevel_Rollout.mat']);
cfg.cache.second_level_rollout_path = fullfile('outputs', [prefix, '_SecondLevel_Rollout.mat']);
cfg.cache.third_level_rollout_path = fullfile('outputs', [prefix, '_ThirdLevel_Rollout.mat']);
cfg.cache.third_level_model_path = fullfile('outputs', [prefix, '_ThirdLevel_Model.mat']);
cfg.safe_flow_evaluation.output_path = fullfile('outputs', [prefix, '_Metrics_Variance_U.mat']);
% Existing L1/L2 models and shared hyperparameters remain valid: their data
% seeds were already fixed and are identical in the two historical configs.
eval(main_text);
end

function verify_fixed_training_data(cfg, s, x, y, root)
path = fullfile(root, 'outputs', 'Racing_FixedSeed_ThirdLevel_TrainingData.mat');
if cfg.comparison.all_training_points_enabled
    ref = load(path, 'training_data_seed', 's', 'x', 'y');
    assert(ref.training_data_seed == cfg.third_level_data_seed);
    assert(isequaln(ref.s, s) && isequaln(ref.x, x) && isequaln(ref.y, y), ...
        'Partial/all third-level FM data are not identical.');
    fprintf('VERIFIED: all-data run uses exactly the partial run third-level X/Y.\n');
else
    training_data_seed = cfg.third_level_data_seed;
    save(path, 'training_data_seed', 's', 'x', 'y', '-v7.3');
    fprintf('SAVED: common fixed third-level FM training X/Y.\n');
end
end

function publish_results(root, seed)
out = fullfile(root, 'outputs');
entries = { ...
 'Partial','Racing_ThirdLevel_Model_EndpointWindowARDLengthScaleV9.mat','Racing_PartialGP_Unguided_ThirdLevel_Rollout.mat','赛道无guidance部分数据三层诊断缓存.mat','Racing_PartialGP_Unguided_Metrics_Variance_U.mat','赛车无guidance部分数据总诊断文件.mat','赛车无guidance部分数据三图.emf','赛车无guidance部分数据归一化uncertainty.emf','赛车无guidance部分数据uncertainty.emf'; ...
 'AllData','赛车无guidance 所有数据三层训练模型.mat','Racing_AllDataGP_Unguided_ThirdLevel_Rollout.mat','赛车无guidance所有数据三层诊断缓存.mat','Racing_AllDataGP_Unguided_Metrics_Variance_U.mat','赛车无guidance所有数据总诊断文件.mat','赛车无guidance所有数据三图.emf','赛车无guidance所有数据归一化uncertainty.emf','赛车无guidance所有数据uncertainty.emf'};
for i=1:size(entries,1)
 tag=entries{i,1}; prefix=['Racing_FixedSeed_',tag];
 model=dir(fullfile(out,[prefix,'_ThirdLevel_Model_EndpointWindowARDLengthScaleV9.mat']));
 assert(numel(model)==1);
 copies={model.name,entries{i,2};[prefix,'_ThirdLevel_Rollout.mat'],entries{i,3};[prefix,'_ThirdLevel_Rollout.mat'],entries{i,4}; ...
  [prefix,'_Metrics_Variance_U.mat'],entries{i,5};[prefix,'_Metrics_Variance_U.mat'],entries{i,6}; ...
  [prefix,'_ThreePanel.emf'],entries{i,7};[prefix,'_third_level_Variance.emf'],entries{i,8}; ...
  [prefix,'_third_level_Variance_Raw.emf'],entries{i,9}};
 for j=1:size(copies,1), assert(isfile(fullfile(out,copies{j,1}))); copyfile(fullfile(out,copies{j,1}),fullfile(out,copies{j,2}),'f'); end
end
save(fullfile(out,'Racing_FixedSeed_Comparison_Complete.mat'),'seed');
fprintf('PUBLISHED fixed-seed racing MAT/EMF aliases (seed=%d).\n',seed);
end

function v=ternary(tf,a,b), if tf,v=a;else,v=b;end,end
