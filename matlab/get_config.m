function cfg = get_config()
%% Training Data
cfg.n_train = 30;
cfg.first_level_generation_samples = 5;
cfg.first_level_use_tangent_features = true;
cfg.n_time_slices = 15;
cfg.first_level_time_steps = 100;
cfg.second_level_time_steps = 100;
% 在 [0.99, rollout_t_max] 这段尾部（PTZF 急剧收紧的区域）
% 额外加密时间步，其余区间仍按 second_level_time_steps 均匀划分。
cfg.second_level_time_refine_start_t = 0.99;
cfg.second_level_time_refine_extra_steps = 10;
cfg.third_level_time_steps = 100;
cfg.third_level_time_refine_start_t = 0.99;
cfg.third_level_time_refine_extra_steps = 0;
cfg.t_min = 0.0;
cfg.rollout_t_max = 0.995;
cfg.random_seed = 7;
cfg.first_level_data_seed = cfg.random_seed + 1;
cfg.first_level_hyperparameter_seed = cfg.random_seed + 2;
cfg.first_level_fit_seed = cfg.random_seed + 3;
cfg.first_level_rollout_seed = cfg.random_seed + 4;
cfg.second_level_data_seed = cfg.random_seed + 5;
cfg.second_level_hyperparameter_seed = cfg.random_seed + 6;
cfg.second_level_fit_seed = cfg.random_seed + 7;
cfg.second_level_rollout_seed = cfg.random_seed + 8;
cfg.segment_points_per_segment = 5;
cfg.second_level_generation_samples = 1;
cfg.third_level_generation_samples = 1;
% 第二层实验: 继承第一层避障结果，加入 PTCLF 与障碍规定时间 CBF。
cfg.stop_after_first_level = false;
cfg.enable_third_level = true;
% 关掉可以跳过第二层的 no-obstacle baseline rollout（只用于画
% before/after 对比图），调参时不需要这张图可以关掉省时间。
cfg.second_level_run_no_obstacle_baseline = false;
% 同上，关掉可以跳过第三层的 no-obstacle baseline rollout。
cfg.third_level_run_no_obstacle_baseline = false;
cfg.third_level_window_stride = cfg.segment_points_per_segment - 1;

%% Animation
cfg.animation.enabled = true;
cfg.animation.trajectory_nr = 1;
cfg.animation.frame_stride = 2;
cfg.animation.delay_time = 0.12;
cfg.animation.third_level_diagnostics_enabled = true;
% 0: 自动选择三级控制链中峰值最大的 segment；1..16: 固定诊断该 segment。
cfg.animation.third_level_diagnostic_segment = 0;

%% Output
cfg.output.enabled = true;

%% Cache
cfg.cache.first_level_model_path = fullfile('outputs', 'LoG_GP_FirstLevel_Model.mat');
cfg.cache.second_level_model_path = fullfile('outputs', 'LoG_GP_SecondLevel_Model.mat');
cfg.cache.third_level_model_path = fullfile('outputs', 'LoG_GP_ThirdLevel_Model.mat');
cfg.cache.first_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_FirstLevel_Rollout_A_ObstacleOnly_NearP3_5Curves_Codex1.mat');
cfg.cache.second_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_SecondLevel_Rollout_C_TunedHbar10_ObstaclePTCBF_VarianceHOCBF_PTCLF_5Curves_Codex1.mat');
cfg.cache.second_level_no_obstacle_rollout_path = fullfile('outputs', ...
    'LoG_GP_SecondLevel_Rollout_C_TunedHbar10_VarianceHOCBF_PTCLF_NoObstacle_5Curves_Codex1.mat');
cfg.cache.third_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_ThirdLevel_Rollout_G_FirstBlockWeight40_1Curve_Codex1.mat');
cfg.cache.third_level_no_obstacle_rollout_path = fullfile('outputs', ...
    'LoG_GP_ThirdLevel_Rollout_C_TunedHbar10_VarianceHOCBF_EndpointPTCLF_NoObstacle_5Curves_Codex1.mat');
cfg.cache.first_level_hyperparameter_path = fullfile('outputs', 'LoG_GP_FirstLevel_Hyperparameter.mat');
cfg.cache.second_level_hyperparameter_path = fullfile('outputs', 'LoG_GP_SecondLevel_Hyperparameter.mat');
cfg.cache.third_level_hyperparameter_path = fullfile('outputs', 'LoG_GP_ThirdLevel_Hyperparameter.mat');

%% LoG-GP Parameters100
cfg.gp.first_level_n_pretrain = 450;
cfg.gp.second_level_n_pretrain = 500;
cfg.gp.second_level_length_scale_time_varying = true;
cfg.gp.second_level_length_scale_time_scale_start = 1.0;
cfg.gp.second_level_length_scale_time_scale_end = 0.5;

cfg.gp.third_level_n_pretrain = 800;
cfg.gp.max_local_data_quantity = 200;
cfg.gp.max_local_gp_quantity = ceil(2.0 * cfg.n_train * cfg.n_time_slices / ...
    cfg.gp.max_local_data_quantity);
% local GP 树相邻叶子之间的重叠区宽度 = (数据范围) * o_ratio。
% 越大，一次查询越容易同时激活多个 local GP（软分裂）；默认是 1/10。
cfg.gp.o_ratio = 0.001;
cfg.gp.aggregation_method = 'GPOE';
cfg.gp.first_level_training_accuracy_threshold = 0.8;
cfg.gp.second_level_training_accuracy_threshold = 2.0;
cfg.gp.third_level_training_accuracy_threshold = 1.5;
%% Obstacle (SafeFlow 避障, 物理坐标; 列 = 障碍)
% 总开关: 关掉 = 完全退回原三层生成
cfg.obstacle.enabled   = true;
cfg.obstacle.centers   = [0.485; 0.505];   % near the first-level generated waypoint-3 cluster, shifted right
cfg.obstacle.semi_axes = [0.08; 0.12];     % doubled ellipse width and height
cfg.obstacle.phi0       = 2.0;            % h>=0 时的普通 CBF 系数
cfg.obstacle.phi1_omega = 4.0;            % h<0 时 blow-up phi1=omega/(1-t)^2, 需 >2
cfg.obstacle.slack_enabled = false;      
cfg.obstacle.slack_weight  = 100;

%% Variance Constraint
cfg.variance_constraint.grad_tol = 1e-6;
% 障碍 CBF 每层开关 + 作用点(均受 cfg.obstacle.enabled 门控)。
% L1 管全部 5 骨架点; L2/L3 管每段内部 3 点(端点由上层继承已安全)。
cfg.variance_constraint.first_level_obstacle_enabled  = true;
cfg.variance_constraint.first_level_obstacle_points   = [1 2 3 4 5];
cfg.variance_constraint.second_level_obstacle_enabled = true;
cfg.variance_constraint.second_level_obstacle_points  = [2 3 4];
% 对照实验结论(2026-07-25): 关掉第三层避障后 P5 误差只从 0.041 降到
% 0.034(17%)，且最差段仍是同一段 row 8。说明末点误差主要不是避障造成
% 的，而是 sequential QP 只给末点 u5(杠杆 0.01) 而非完整 u(杠杆 0.24)。
cfg.variance_constraint.third_level_obstacle_enabled  = true;
cfg.variance_constraint.third_level_obstacle_points   = [2 3 4];
cfg.variance_constraint.first_level_integral_uncertainty_budget = 10;
% Experiment C: 障碍 CBF + variance HOCBF；不启用终端方差 PTCBF。
cfg.variance_constraint.first_level_hocbf_enabled = true;
cfg.variance_constraint.first_level_hocbf_alpha2 = 3.0;
cfg.variance_constraint.first_level_hocbf_relaxation_bound = 5;
cfg.variance_constraint.first_level_psi1_margin = 2;
cfg.variance_constraint.first_level_diagnostics = true;
cfg.variance_constraint.first_level_ptcbf_enabled = false;
cfg.variance_constraint.first_level_terminal_variance_beta_final = 50;
cfg.variance_constraint.first_level_terminal_variance_ptzf_initial_margin = 5;
cfg.variance_constraint.first_level_terminal_variance_ptzf_gamma = 0.03;
cfg.variance_constraint.first_level_terminal_variance_alpha = 2;
cfg.variance_constraint.first_level_ptclf_enabled = false;
cfg.variance_constraint.first_level_hocbf_slack_enabled = false;
cfg.variance_constraint.first_level_terminal_variance_slack_enabled = false;
% 前期(t<switch) variance 硬、避障软；后期(t>=switch) variance 软、避障硬
% (导师方案，实验③)。hocbf 前硬后软，obstacle 前软后硬。
cfg.variance_constraint.first_level_slack_switch_time = 0.65;
% Second-level experiment C: obstacle PTCBF + variance HOCBF + PTCLF.
cfg.variance_constraint.second_level_integral_uncertainty_budget = 0.5;
cfg.variance_constraint.second_level_hocbf_enabled = true;
cfg.variance_constraint.second_level_hocbf_alpha2 = 0.5;
cfg.variance_constraint.second_level_hocbf_relaxation_bound = 8;
cfg.variance_constraint.second_level_psi1_margin = 80;
cfg.variance_constraint.second_level_diagnostics = true;
cfg.variance_constraint.second_level_ptcbf_enabled = false;
cfg.variance_constraint.second_level_terminal_variance_beta_final = 5.0;
cfg.variance_constraint.second_level_terminal_variance_ptzf_initial_margin = 1;
cfg.variance_constraint.second_level_terminal_variance_ptzf_gamma = 0.2;
cfg.variance_constraint.second_level_terminal_variance_alpha = 2.0;
cfg.variance_constraint.second_level_ptclf_enabled = true;
cfg.variance_constraint.second_level_anchor_clf_ptzf_enabled = true;
cfg.variance_constraint.second_level_anchor_clf_ptzf_cg = 2.0;
cfg.variance_constraint.second_level_anchor_clf_cpt = 150;
cfg.variance_constraint.second_level_anchor_clf_ptzf_initial_margin = 4;
cfg.variance_constraint.second_level_slack_enabled = true;
% hocbf_slack_enabled=false: t < switch 硬，t >= switch 才靠
% hocbf_slack_late_start_time(=slack_switch_time) 打开 slack，实现
% "前硬后软"。
cfg.variance_constraint.second_level_hocbf_slack_enabled = false;
cfg.variance_constraint.second_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.second_level_anchor_clf_slack_enabled = true;
% Before 0.55: variance HOCBF hard, obstacle PTCBF/PTCLF soft.
% From 0.55: variance HOCBF may use slack, obstacle PTCBF/PTCLF become hard.
cfg.variance_constraint.second_level_slack_switch_time = 0.65;
cfg.variance_constraint.second_level_anchor_clf_slack_hard_after_time = 0.65;
cfg.variance_constraint.second_level_obstacle_slack_enabled = true;
cfg.variance_constraint.second_level_obstacle_activation_time = 0.30;
cfg.variance_constraint.second_level_obstacle_slack_hard_after_time = 0.65;
cfg.variance_constraint.second_level_hocbf_slack_weight = 66.8;
cfg.variance_constraint.second_level_terminal_variance_slack_weight = 66.8;
cfg.variance_constraint.second_level_anchor_clf_slack_weight = 30;
cfg.variance_constraint.second_level_anchor_snap_flow_steps = 5;
cfg.variance_constraint.second_level_anchor_snap_position_only = false;
cfg.variance_constraint.second_level_anchor_clf_position_only = false;
cfg.variance_constraint.third_level_integral_uncertainty_budget = 5;
% 第三层单独调小梯度退化门槛，减少 HOCBF 那一行在尾段被整行跳过的机会；
% 不写时一二层仍用共用的 cfg.variance_constraint.grad_tol(=1e-6)。
cfg.variance_constraint.third_level_grad_tol = 1e-3;
cfg.variance_constraint.third_level_hocbf_enabled = true;
cfg.variance_constraint.third_level_hocbf_alpha2 = 0.001;
cfg.variance_constraint.third_level_hocbf_relaxation_bound = 5;
cfg.variance_constraint.third_level_psi1_margin = 2;
cfg.variance_constraint.third_level_diagnostics = true;
cfg.variance_constraint.third_level_ptcbf_enabled = true;
cfg.variance_constraint.third_level_terminal_variance_ptcbf_end_time = 0.85;
cfg.variance_constraint.third_level_terminal_variance_beta_final = 5;
cfg.variance_constraint.third_level_terminal_variance_ptzf_initial_margin = 4.0;
cfg.variance_constraint.third_level_terminal_variance_ptzf_gamma = 0.3;
cfg.variance_constraint.third_level_terminal_variance_alpha = 9.0;
% Continuous endpoint tracking: PTCLF first distributes its correction over
% all increment blocks. The internal obstacle PTCBF then uses that control
% as its reference without imposing the P5 PTCLF row a second time.
cfg.variance_constraint.third_level_ptclf_enabled = true;
cfg.variance_constraint.third_level_sequential_increment_qp_enabled = true;
cfg.variance_constraint.third_level_sequential_ptclf_reference_enabled = true;
cfg.variance_constraint.third_level_first_block_control_weight = 40;
cfg.variance_constraint.third_level_sequential_ptclf_reference_impl_version = 6;
cfg.variance_constraint.third_level_hocbf_filter_end_time = 0.85;
cfg.variance_constraint.third_level_anchor_clf_ptzf_enabled = true;
cfg.variance_constraint.third_level_anchor_clf_ptzf_cg = 1.3;
cfg.variance_constraint.third_level_anchor_clf_cpt = 280;
cfg.variance_constraint.third_level_anchor_clf_endpoint_cpt = [80; 80];
cfg.variance_constraint.third_level_anchor_clf_endpoint_ptzf_cg = [5.0; 5.0];
% Per-endpoint PTZF bound: gbar0 = (1 + ratio) * g0 + absolute margin.
% A relative margin keeps P1/P5 and all segments on the same error scale.
cfg.variance_constraint.third_level_anchor_clf_endpoint_ptzf_initial_margin_ratios = [0.10; 0.10];
cfg.variance_constraint.third_level_anchor_clf_endpoint_ptzf_initial_margins = [0.175; 0.175];
cfg.variance_constraint.third_level_anchor_clf_ptzf_initial_margin = 0.35;
cfg.variance_constraint.third_level_slack_enabled = false;
cfg.variance_constraint.third_level_hocbf_slack_enabled = false;
cfg.variance_constraint.third_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.third_level_anchor_clf_slack_enabled = false;
cfg.variance_constraint.third_level_slack_switch_time = inf;
cfg.variance_constraint.third_level_anchor_clf_slack_hard_after_time = 0.85;
cfg.variance_constraint.third_level_obstacle_slack_enabled = false;
cfg.variance_constraint.third_level_obstacle_activation_time = 0.00;
cfg.variance_constraint.third_level_obstacle_slack_hard_after_time = 0.85;
cfg.variance_constraint.third_level_obstacle_slack_weight = 100;
cfg.variance_constraint.third_level_obstacle_phi1_omega = 0.2;
% 障碍内 h<0 时采用分段增益：前段先弱拉回，等 PTCLF 基本收敛后
% 再切换到 omega/(1-t_eff)^2 的 blow-up 增益。
cfg.variance_constraint.third_level_obstacle_phi1_early_gain = 0.05;
cfg.variance_constraint.third_level_obstacle_phi1_switch_time = 0.40;
cfg.variance_constraint.third_level_obstacle_phi0 = 5.0;
cfg.variance_constraint.third_level_hocbf_slack_weight = 0.001;
cfg.variance_constraint.third_level_terminal_variance_slack_weight = 1;
cfg.variance_constraint.third_level_anchor_clf_slack_weight = 5000;
cfg.variance_constraint.third_level_anchor_clf_first_slack_weight = 5000;
cfg.variance_constraint.third_level_anchor_clf_last_slack_weight = 1e8;
cfg.variance_constraint.third_level_anchor_snap_flow_steps = 5;
cfg.variance_constraint.third_level_anchor_snap_position_only = false;
cfg.variance_constraint.third_level_anchor_snap_hold_impl_version = 3;
cfg.variance_constraint.third_level_post_endpoint_overwrite_enabled = false;
end
