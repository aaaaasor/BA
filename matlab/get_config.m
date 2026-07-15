function cfg = get_config()
%% Training Data
cfg.n_train = 30;
cfg.first_level_generation_samples = 5;
cfg.first_level_use_tangent_features = true;
cfg.n_time_slices = 15;
cfg.first_level_time_steps = 100;
cfg.second_level_time_steps = 100;
% 在 [0.99, rollout_t_max] 这段尾部（PTZF 包络急剧收紧的区域）
% 额外加密时间步，其余区间仍按 second_level_time_steps 均匀划分。
cfg.second_level_time_refine_start_t = 0.99;
cfg.second_level_time_refine_extra_steps = 10;
cfg.third_level_time_steps = 100;
% 第三层尾段加密: gbar 在 t->1 陡降,加密后
% 每步收缩量更小,让 PTCLF 的 g 能贴着降下去,减轻末端违反、提高端点精度。
cfg.third_level_time_refine_start_t = 0.99;
% cg 提高后 gbar(t) 在 t->1 附近收缩更陡，加密尾段时间步避免数值滞后。
cfg.third_level_time_refine_extra_steps = 30;
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
cfg.enable_third_level = false;
cfg.third_level_window_stride = cfg.segment_points_per_segment - 1;

%% Animation
cfg.animation.enabled = true;
cfg.animation.trajectory_nr = 1;
cfg.animation.frame_stride = 2;
cfg.animation.delay_time = 0.12;

%% Output
cfg.output.enabled = true;

%% Cache
cfg.cache.first_level_model_path = fullfile('outputs', 'LoG_GP_FirstLevel_Model.mat');
cfg.cache.second_level_model_path = fullfile('outputs', 'LoG_GP_SecondLevel_Model.mat');
cfg.cache.third_level_model_path = fullfile('outputs', 'LoG_GP_ThirdLevel_Model.mat');
cfg.cache.first_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_FirstLevel_Rollout_C_ObstaclePlusVarianceHOCBF_NearP3.mat');
cfg.cache.second_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_SecondLevel_Rollout_ObstaclePTCBF_HOCBF_PTCLF_Switch0p65.mat');
cfg.cache.third_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_ThirdLevel_Rollout_WithUFrom0.mat');
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
% 第三层超参数: 第一次运行自动 fitrgp 并缓存到 mat（SigmaN 加载后 /50，
% 与第二层一致），之后每次直接加载缓存。
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
% 总开关: 关掉 = 完全退回原三层生成(无障碍 CBF / 无终端滤波 / 无 L3 端点硬覆盖)。
cfg.obstacle.enabled   = true;
cfg.obstacle.centers   = [0.485; 0.505];   % near the first-level generated waypoint-3 cluster, shifted right
cfg.obstacle.semi_axes = [0.04; 0.06];     % visible local ellipse for the 5-point first-level curve
cfg.obstacle.phi0       = 2.0;            % h>=0 时的普通 CBF 系数
cfg.obstacle.phi1_omega = 4.0;            % h<0 时 blow-up phi1=omega/(1-t)^2, 需 >2
% 前软后硬(实验②，导师方案): t<switch 软、t>=switch 硬。此处必须是
% true 才能让 *_slack_hard_after_time 的切换生效(false 会短路成全程硬)。
cfg.obstacle.slack_enabled = true;        % Experiment B: obstacle soft early, hard after switch
cfg.obstacle.slack_weight  = 1e4;         % 大权重 -> 近似硬约束

%% Variance Constraint
cfg.variance_constraint.grad_tol = 1e-6;
% 障碍 CBF 每层开关 + 作用点(均受 cfg.obstacle.enabled 门控)。
% L1 管全部 5 骨架点; L2/L3 管每段内部 3 点(端点由上层继承已安全)。
cfg.variance_constraint.first_level_obstacle_enabled  = true;
cfg.variance_constraint.first_level_obstacle_points   = [1 2 3 4 5];
cfg.variance_constraint.second_level_obstacle_enabled = true;
cfg.variance_constraint.second_level_obstacle_points  = [1 2 3 4 5];
cfg.variance_constraint.third_level_obstacle_enabled  = false;
cfg.variance_constraint.third_level_obstacle_points   = [1 2 3 4 5];
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
cfg.variance_constraint.second_level_integral_uncertainty_budget = 3;
cfg.variance_constraint.second_level_hocbf_enabled = true;
cfg.variance_constraint.second_level_hocbf_alpha2 = 0.5;
cfg.variance_constraint.second_level_hocbf_relaxation_bound = 5;
cfg.variance_constraint.second_level_psi1_margin = 25;
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
cfg.variance_constraint.second_level_hocbf_slack_enabled = false;
cfg.variance_constraint.second_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.second_level_anchor_clf_slack_enabled = true;
% 统一切换时刻: t < switch 时 PTCLF 带 slack、HOCBF/PTCBF 硬，之后互换。
cfg.variance_constraint.second_level_slack_switch_time = 0.65;
cfg.variance_constraint.second_level_hocbf_slack_weight = 66.8;
cfg.variance_constraint.second_level_terminal_variance_slack_weight = 66.8;
cfg.variance_constraint.second_level_anchor_clf_slack_weight = 31.4;
cfg.variance_constraint.second_level_anchor_snap_flow_steps = 0;
cfg.variance_constraint.second_level_anchor_snap_position_only = false;
cfg.variance_constraint.second_level_anchor_clf_position_only = false;
cfg.variance_constraint.third_level_integral_uncertainty_budget = 5;
cfg.variance_constraint.third_level_hocbf_enabled = true;
cfg.variance_constraint.third_level_hocbf_alpha2 = 2.0;
cfg.variance_constraint.third_level_hocbf_relaxation_bound = 5.0;
cfg.variance_constraint.third_level_psi1_margin = 5.0;
cfg.variance_constraint.third_level_diagnostics = true;
cfg.variance_constraint.third_level_ptcbf_enabled = true;
cfg.variance_constraint.third_level_terminal_variance_beta_final = 8;
cfg.variance_constraint.third_level_terminal_variance_ptzf_initial_margin = 4.0;
cfg.variance_constraint.third_level_terminal_variance_ptzf_gamma = 0.3;
cfg.variance_constraint.third_level_terminal_variance_alpha = 0.5;
cfg.variance_constraint.third_level_ptclf_enabled = true;
cfg.variance_constraint.third_level_anchor_clf_ptzf_enabled = true;
cfg.variance_constraint.third_level_anchor_clf_ptzf_cg = 0.8;
cfg.variance_constraint.third_level_anchor_clf_cpt = 230;
cfg.variance_constraint.third_level_anchor_clf_ptzf_initial_margin = 5;
cfg.variance_constraint.third_level_slack_enabled = true;
cfg.variance_constraint.third_level_hocbf_slack_enabled = false;
cfg.variance_constraint.third_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.third_level_anchor_clf_slack_enabled = true;
% t < 0.55: PTCLF 带 slack、HOCBF/PTCBF 硬；t >= 0.55: 互换
% (HOCBF/PTCBF 变软、PTCLF 变硬)。
cfg.variance_constraint.third_level_slack_switch_time = 0.55;
cfg.variance_constraint.third_level_hocbf_slack_weight = 10;
cfg.variance_constraint.third_level_terminal_variance_slack_weight = 30;
cfg.variance_constraint.third_level_anchor_clf_slack_weight = 80;
% 倒数 10 步把段端点(矩阵型 CLF)用最小范数投影钉到目标值,之后继续
% rollout(段内其余自由度不受影响)。仅在 ptclf_enabled 时生效。
cfg.variance_constraint.third_level_anchor_snap_flow_steps = 10;
cfg.variance_constraint.third_level_anchor_snap_position_only = false;
end
