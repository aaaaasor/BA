function cfg = get_config()
%% Scenario
% 'obstacle' - 原三层 demo：单位方块内的合成轨迹 + 椭圆障碍。
% 'racing'   - 赛道段：Nuerburgring GP-Strecke s=250..1150 m 的几何路径数据集，
%              由 build_track_dataset 生成，训练点从同一个 65 点数组按 stride 抽。
% 该开关只切换训练数据来源和障碍设置，三层结构和求解器路径完全共用。
cfg.scenario = 'racing';
cfg.track_dataset_path = fullfile('trajectory_data', 'track_dataset_arena.mat');

%% Training Data
cfg.n_train = 30;
cfg.first_level_generation_samples = 100;
cfg.first_level_use_tangent_features = true;
cfg.n_time_slices = 15;
cfg.first_level_time_steps = 100;
cfg.second_level_time_steps = 100;
% The rollout now ends at t=0.99 with no PTZF time shift.  No zero-length
% tail-refinement interval is added at that endpoint.
cfg.second_level_time_refine_start_t = 0.99;
cfg.second_level_time_refine_extra_steps = 0;
cfg.third_level_time_steps = 100;
cfg.third_level_time_refine_start_t = 0.99;
cfg.third_level_time_refine_extra_steps = 0;
cfg.t_min = 0.0;
cfg.rollout_t_max = 0.996;
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
cfg.stop_after_first_level = 0;
cfg.enable_third_level = false;
% Run a matched first-level baseline with the same initial samples and
% variance constraints, changing only obstacle_enabled=false.
cfg.first_level_run_no_obstacle_baseline = false;
% 关掉可以跳过第二层的 no-obstacle baseline rollout（只用于画
% before/after 对比图），调参时不需要这张图可以关掉省时间。
cfg.second_level_run_no_obstacle_baseline = false;
% 同上，关掉可以跳过第三层的 no-obstacle baseline rollout。
cfg.third_level_run_no_obstacle_baseline = false;
cfg.third_level_window_stride = cfg.segment_points_per_segment - 1;

%% Parallel Execution
% RK4 rollout 的 sample 循环（每个 sample = 一条轨迹）可以串行或并行执行。
% 默认串行(enabled=false)，行为与之前完全一致；打开后每个 worker(核)一次
% 只领一条轨迹（SubrangeSize=1），算完再领下一条。并行需要 Parallel
% Computing Toolbox；缺少时按 fallback_to_serial 决定是退回串行还是报错。
cfg.parallel.enabled = 1;
% 0 = 用默认 parallel profile 的核数；>0 = 固定使用指定核数，不按当前层的
% 轨迹条数截断；现有池大小不符时会自动重建。
cfg.parallel.num_workers = 6;
cfg.parallel.fallback_to_serial = true;

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
% Plot the first-level joint obstacle/boundary soft-min safe set.
cfg.output.plot_first_level_joint_softmin_safe_set = true;
% Live view of the curve committed at each first-level RK4 step.  Plotting
% every sample would create 100 figures, so the default follows sample 1.
cfg.output.live_first_level_rk4_trajectory_enabled = 0;
cfg.output.live_first_level_rk4_trajectory_sample_indices = ...
    1:cfg.first_level_generation_samples;
cfg.output.live_first_level_rk4_trajectory_stride = 1;
cfg.output.live_first_level_rk4_trajectory_delay = 0.01;
% Reuse one figure for all samples; only the dynamic trajectory objects are
% replaced, while the red/green h_soft background remains in place.
cfg.output.live_first_level_rk4_trajectory_close_after_save = false;
% Rollout figure marker switch:
% true  = show colored generated-point circles and square/diamond anchors;
% false = draw trajectory lines only.
cfg.output.rollout_markers_enabled = false;
% true = rollout 面板只画"整条折线都不穿障碍"的轨迹, 弦穿越的整条隐藏,
% 标题和控制台会报 保留数/总数。判据用 cfg.obstacle 的物理几何并在每条弦上
% 稠密采样, 所以"点安全但连线切角"的那些会被剔除。
cfg.output.plot_only_obstacle_free_curves = false;
% Skip second-level rollout uncertainty evaluation and its uncertainty
% plots without changing the second-level rollout itself.
cfg.output.second_level_uncertainty_enabled = 1;
% Temporary speed switch: skip third-level rollout uncertainty evaluation
% and its uncertainty plots. This does not change the rollout itself.
cfg.output.third_level_uncertainty_enabled = false;
% Level-specific interpretation: stage 2 checks generated points only;
% stage 3 densely checks every chord of the delivered polyline.
% 边界样条保真度检查图: 把 CBF 实际使用的 pchip 左右曲线和原始
% Nuerburgring.csv 折线叠在一起, 并给出逐点偏差。只在需要核对赛道几何时
% 打开(每次约几秒), 平时关闭。
cfg.output.plot_pchip_vs_raw_track = true;

%% Cache
cfg.cache.first_level_model_path = fullfile('outputs', 'LoG_GP_FirstLevel_Model.mat');
cfg.cache.second_level_model_path = fullfile('outputs', 'LoG_GP_SecondLevel_Model.mat');
cfg.cache.third_level_model_path = fullfile('outputs', 'LoG_GP_ThirdLevel_Model.mat');
cfg.cache.first_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_FirstLevel_Rollout_AllConstraints_NoPTCLF_5Curves.mat');
cfg.cache.first_level_no_obstacle_rollout_path = fullfile('outputs', ...
    'LoG_GP_FirstLevel_Rollout_NoObstacle_5Curves_Codex1.mat');
cfg.cache.second_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_SecondLevel_Rollout_AllConstraints_Snap5.mat');
cfg.cache.second_level_no_obstacle_rollout_path = fullfile('outputs', ...
    'LoG_GP_SecondLevel_Rollout_VarianceHOCBF_VariancePTCBF_PTCLF_NoObstacle_5Curves_Codex1.mat');
cfg.cache.third_level_rollout_path = fullfile('outputs', ...
    'LoG_GP_ThirdLevel_Rollout_AllConstraints_Snap5.mat');
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
cfg.gp.second_level_training_accuracy_threshold = 3.0;
cfg.gp.third_level_training_accuracy_threshold = 1.5;
%% Obstacle (SafeFlow 避障, 物理坐标; 列 = 障碍)
% 总开关: 关掉 = 完全退回原三层生成
cfg.obstacle.enabled   = 1;
cfg.obstacle.centers   = [0.485; 0.505];   % near the first-level generated waypoint-3 cluster, shifted right
cfg.obstacle.semi_axes = [0.08; 0.12];     % doubled ellipse width and height
cfg.obstacle.phi0       = 2.0;            % h>=0 时的普通 CBF 系数
cfg.obstacle.phi1_omega = 4.0;            % h<0 时 blow-up phi1=omega/(1-t)^2, 需 >2
cfg.obstacle.slack_enabled = false;
cfg.obstacle.slack_weight  = 100;
% 作用点落到椭圆正中心时 grad = 2*Q*(p-c) 退化为 0，CBF 行没有一阶方向。
% 此时改用一条"沿短轴逃逸"的替代行(见 obstacle_cbf_info)，这里给它的
% 速度。留空 = 自适应取该障碍的短半轴 min(a,b)，即一个时间单位走出椭圆。
% Racing obstacle geometry. Each shape can be enabled and adjusted
% independently; cfg.obstacle.enabled remains the sole master switch.
cfg.obstacle.square.enabled = 1;
cfg.obstacle.square.track_fraction = 0.72;
cfg.obstacle.square.half_size_ratio = 2.1;
cfg.obstacle.square.center_x_offset = 0.015;
cfg.obstacle.square.center_y_offset = 0.020;
cfg.obstacle.square.length_ratio = 1.5;
cfg.obstacle.square.global_angle = 0.0;  % horizontal long diagonal
cfg.obstacle.square.exponent = 1.2;     % mildly smoothed elongated diamond
cfg.obstacle.square.track_inside_ratio = 0.50;
% constraint_inflation 直接加在半轴上(见 make_level_variance_constraint 的
% inflated_semi_axes), 所以"膨胀后的自由缝 ≈ 物理自由缝 − inflation"。它要
% 覆盖的是弦相对真实路径的偏离(这里量到约 0.001~0.002: 曲率矢高 d^2/(8R)
% ≈ 0.0003, 绕障横向摊分 ~0.002), 取物理最小缝的 1/4~1/3 即可。取过大会把
% 走廊整个堵死, 硬约束下 QP 直接不可行, 反而连物理障碍都不再被尊重。
% square 需要比另外两个大: exponent = 1.2 接近菱形, 尖角处膨胀边界的局部曲率
% 半径只有约 0.050, 两个贴在膨胀边界上的点之间, 弦会下凹 d^2/(8R)。实测弦长
% 0.0582 时下凹 0.0084 > 膨胀 0.006, 所以弦穿进物理障碍 0.0029。要求下凹不
% 超过膨胀量: d^2/(8(R+delta)) <= delta  =>  delta >= 0.0073, 取 0.010 留余量。
% 代价: 物理右缝 0.0263 -> 膨胀后 0.0163 (38% 走廊宽), 通道仍然宽敞。
cfg.obstacle.square.constraint_inflation = 0.010;

cfg.obstacle.ellipse.enabled = 1;
cfg.obstacle.ellipse.track_fraction = 0.52;
cfg.obstacle.ellipse.center_x = 0.30;
% Place the ellipse on the upper/outside edge, leaving 40% clear below.
cfg.obstacle.ellipse.center_y = 0.96;
% Semi-major/minor axes relative to the local corridor half-width.
cfg.obstacle.ellipse.semi_major_ratio = 4.5;
% 2.0 时半短轴 0.0408 = 走廊全宽 0.0417 的 98%, 几乎横跨整条走廊, 全靠贴在
% 右墙上才留出左边一条 0.0109 的缝, 扣掉 1.25 倍膨胀和边界 margin 后只剩
% 0.0038 —— 三个障碍里最紧的, 而且膨胀已经加不动(上限 0.0039)。
% 缩到 1.8 后物理缝 0.0149, 可承受膨胀升到 0.0071, 外观差别很小。
cfg.obstacle.ellipse.semi_minor_ratio = 1.8;
% Zero aligns the major axis with the local track tangent.
cfg.obstacle.ellipse.relative_angle = 0.0;
cfg.obstacle.ellipse.constraint_inflation = 0.006;

cfg.obstacle.superellipse.enabled = 1;
cfg.obstacle.superellipse.track_fraction = 0.31;
cfg.obstacle.superellipse.center_x_offset = -0.006;
cfg.obstacle.superellipse.center_y_offset = -0.004;
cfg.obstacle.superellipse.semi_major_ratio = 2.4;
cfg.obstacle.superellipse.semi_minor_ratio = 0.35;
cfg.obstacle.superellipse.relative_angle = 4 * pi / 180;
cfg.obstacle.superellipse.exponent = 4;
cfg.obstacle.superellipse.constraint_inflation = 0.005;

%% Track Boundary Constraint
cfg.track_boundary.enabled = 1;
% Two fixed global implicit fields h1(x,y), h2(x,y).  They are built once
% from the two rails and queried directly during rollout; no centerline
% location, longitudinal phase, or run-time nearest cross-section is used.
cfg.track_boundary.constraint_method = 'global_implicit_fields';
% Bump when the boundary equation changes so cached rollouts are rebuilt.
% 22: 场的 h 改为到轨折线的精确距离(原来是到采样点, 贴墙处高估 0.07~0.28 m),
%     梯度改为对双线性 h 解析求导(原来插值另存的 FD 梯度网格, 与 h 不自洽)。
% 23: level-2 h1/h2 may be combined by one conservative soft minimum.
% 24: retuned the complete level-2 hard/soft activation schedule.
cfg.track_boundary.implementation_version = 25;
cfg.track_boundary.n_spline_points = 400;
% Prescribed-time boundary CBF.  Safe points use phi0; unsafe points use
% the early gain until the late blow-up branch takes over.
cfg.track_boundary.phi0 = 1.0;
% cfg.track_boundary.phi1_early_gain = 100.0;
cfg.track_boundary.phi1_switch_time = 0.85;
cfg.track_boundary.phi1_omega = 0.1;
% No numerical ceiling: retain the original prescribed-time blow-up.
cfg.track_boundary.phi1_max = inf;
% SafeFlow-style terminal safety filter is kept available but temporarily
% disabled. Set this switch to true to project an unsafe terminal state back
% into the joint boundary/obstacle safe set after a successful RK4 rollout.
cfg.variance_constraint.terminal_safety_filter_enabled = false;
cfg.variance_constraint.terminal_safety_filter_tolerance = 1e-8;
cfg.variance_constraint.terminal_safety_filter_max_iterations = 200;
% Numerical regularization used only when the boundary rows are soft.  A
% hard boundary deliberately uses the uncapped prescribed-time gain.
cfg.track_boundary.phi1_tau_min = 0.01;
% margin 进 h 的定义 (h = n'*(p-q) - margin), 等于把走廊两侧各收窄这么多。
cfg.track_boundary.margin = 0.003;
cfg.track_boundary.activation_time = 0.85;
% 边界/中心线样条类型。必须是 'spline'(C2): track_boundary_cbf_info 的梯度
% 链式项要用 c'', 而 pchip 只有 C1, 其 c'' 在结点处跳变(最大 696), 会让
% ds*/dp 的分母在走廊内变号(最小 -2.479)。改成 'pchip' 会让链式项在最坏点
% 的梯度误差大一个量级(1.27e-2 vs 5.13e-3)。两者对原始赛道折线的最大偏离
% 只差 1 cm, 所以用 spline 没有保真度代价。
cfg.track_boundary.spline_type = 'spline';
% Track containment is a safety condition and therefore remains hard.
% Performance rows (in particular the second-level anchor CLF) retain
% slack so that they cannot make the safety QP infeasible.
cfg.track_boundary.slack_enabled = false;
cfg.track_boundary.slack_weight = 1e5;

%% Variance Constraint
cfg.variance_constraint.grad_tol = 1e-6;
% Independent prescribed-time endpoints.  Variance/PTCLF theoretical time
% is tau=t/T_ptzf, so the final RK4 k4 stage reaches tau=1 exactly instead
% of crossing it through an additive shift.  Constraint activation and
% hard/soft switch times remain independent physical-time parameters.
cfg.variance_constraint.terminal_variance_ptzf_terminal_time = ...
    cfg.rollout_t_max;
cfg.variance_constraint.anchor_clf_ptzf_terminal_time = ...
    cfg.rollout_t_max;
% Safe-side rows remain permissive enough to coexist with the hard obstacle
% rows. Unsafe points use a separate, stronger early recovery gain below.
cfg.variance_constraint.third_level_track_boundary_phi0 = 20.0;
% OOD is evaluated in a dimensionless GP scale:
% sqrt(mean_i(sigma_i^2 / SigmaF_i^2)).  A rollout is declared OOD when
% the configured quantile exceeds this threshold.  Using a quantile rather
% than the absolute maximum avoids classifying the complete rollout from a
% few random initial-state outliers.
cfg.variance_constraint.ood_normalized_sigma_threshold = 0.8;
cfg.variance_constraint.ood_quantile = 0.95;
cfg.variance_constraint.first_level_obstacle_enabled  = true;
cfg.variance_constraint.first_level_obstacle_points   = [1 2 3 4 5];
cfg.variance_constraint.first_level_obstacle_constraint_inflation_scale = 1.0;
cfg.variance_constraint.second_level_obstacle_enabled = true;
cfg.variance_constraint.second_level_obstacle_points  = [2 3 4];
% Reproduce the earlier second-level setting: do not apply the configured
% obstacle inflation to the second-level PTCBF geometry.
cfg.variance_constraint.second_level_obstacle_constraint_inflation_scale = 0.0;
cfg.variance_constraint.third_level_obstacle_enabled  = true;
cfg.variance_constraint.third_level_obstacle_points   = [2 3 4];
cfg.variance_constraint.third_level_obstacle_constraint_inflation_scale = 1.0;
% Per-level boundary switches and affected trajectory points. The master
% cfg.track_boundary.enabled must also be true.
cfg.variance_constraint.first_level_track_boundary_enabled = true;
cfg.variance_constraint.first_level_track_boundary_points = [1 2 3 4 5];
% Smooth the left/right rail pair before it is combined with the obstacle
% fields.  Using the same kappa as the joint soft minimum makes the nested
% construction exactly equivalent to one soft minimum over all components.
cfg.variance_constraint.first_level_track_boundary_combine_method = ...
    'softmin';
cfg.variance_constraint.first_level_track_boundary_softmin_kappa = 2000.0;
% Let the hard variance constraints form a track-like sample first.  The
% boundary PTCBF then joins softly before taking over terminal safety.
cfg.variance_constraint.first_level_track_boundary_activation_time = 0.50;
cfg.variance_constraint.second_level_track_boundary_enabled = true;
cfg.variance_constraint.second_level_track_boundary_points = [2 3 4];
% Stage-2 late handoff: obstacle and boundary PTCBFs enter at 0.90 with
% slack, then become hard together at 0.95.
cfg.variance_constraint.second_level_track_boundary_activation_time = 0.90;
cfg.variance_constraint.second_level_track_boundary_phi0 = 5.0;
% Combine the two global boundary fields into one conservative smooth
% minimum at level 2.  This replaces two potentially opposing hard rows by
% one row governed mainly by the more dangerous boundary.  kappa=500 adds
% at most log(2)/500 = 1.386e-3 of extra inward conservatism.
cfg.variance_constraint.second_level_track_boundary_combine_method = ...
    'softmin';
cfg.variance_constraint.second_level_track_boundary_softmin_kappa = 2000.0;
cfg.variance_constraint.third_level_track_boundary_enabled = true;
% P1/P5 are snapped exactly to the safe level-2 anchors at the end.  The
% third-level boundary filter owns only the newly generated interior points.
cfg.variance_constraint.third_level_track_boundary_points = [2 3 4];
cfg.variance_constraint.third_level_track_boundary_margin = 0.003;
cfg.variance_constraint.first_level_track_boundary_phi1_omega = 0.1/100;
% Once the delayed first-level boundary filter is activated, unsafe points
% use the prescribed-time blow-up immediately; safe points still use phi0.
cfg.variance_constraint.first_level_track_boundary_phi1_switch_time = 0.0;
cfg.variance_constraint.second_level_track_boundary_phi1_omega = 0.1/100;
cfg.variance_constraint.second_level_track_boundary_phi1_max = inf;
% phi1_early_gain 只从 cfg.track_boundary 读, 没有分层覆盖 —— 下面这行本身
% 不生效; 真正起作用的是 cfg.track_boundary.phi1_early_gain。
cfg.variance_constraint.second_level_track_boundary_phi1_early_gain = 30.0;
% Independent boundary safety stays finite; joint soft-min supplies the
% active endpoint prescribed-time blow-up.
cfg.variance_constraint.second_level_track_boundary_phi1_switch_time = 1.0;
cfg.variance_constraint.third_level_track_boundary_phi1_omega = 0.02;
cfg.variance_constraint.third_level_track_boundary_phi1_early_gain = 50.0;
cfg.variance_constraint.third_level_track_boundary_phi1_switch_time = 0.99;
% Keep the prescribed-time gain inside the fixed-step RK4 stability region:
% phi_max*dt = 200*0.00995 = 1.99 < 2.785.
cfg.variance_constraint.third_level_track_boundary_phi1_max = inf;
cfg.variance_constraint.third_level_track_boundary_activation_time = 0.7;
% All three levels use a soft-to-hard track-boundary schedule.  In the
% third-level cascade the boundary joins the obstacle rows in one safety QP.
cfg.variance_constraint.first_level_track_boundary_slack_enabled = true;
cfg.variance_constraint.first_level_track_boundary_slack_hard_after_time = 0.90;
cfg.variance_constraint.second_level_track_boundary_slack_enabled = true;
cfg.variance_constraint.second_level_track_boundary_slack_hard_after_time = 0.95;
cfg.variance_constraint.third_level_track_boundary_slack_enabled = false;
cfg.variance_constraint.third_level_track_boundary_slack_hard_after_time = 0.0;
cfg.variance_constraint.first_level_integral_uncertainty_budget = 10;
% First level: all safety/variance constraints enabled; PTCLF remains off.
cfg.variance_constraint.first_level_hocbf_enabled = 1;
cfg.variance_constraint.first_level_hocbf_alpha2 = 3.0;
cfg.variance_constraint.first_level_hocbf_relaxation_bound = 5;
cfg.variance_constraint.first_level_psi1_margin = 2;
cfg.variance_constraint.first_level_diagnostics = false;
cfg.variance_constraint.first_level_ptcbf_enabled = true;
% 收紧末端方差上界: 12 -> 9。beta(t) <= beta_final 是硬 PTCBF 目标，
% 降低它直接要求一层 rollout 末端停在更低的 GP 方差上。
cfg.variance_constraint.first_level_terminal_variance_beta_final = 0.1;
cfg.variance_constraint.first_level_terminal_variance_ptzf_initial_margin = 3;
cfg.variance_constraint.first_level_terminal_variance_ptzf_gamma = 0.3;
cfg.variance_constraint.first_level_terminal_variance_alpha = 8.0;
cfg.variance_constraint.first_level_ptclf_enabled = false;
% Solver backend by level: the first two levels retain MATLAB quadprog,
% while the third-level sequential cascade uses the small-row closed-form
% active-set solver whenever that path is available.
cfg.variance_constraint.first_level_closed_form_solver_enabled = false;
cfg.variance_constraint.first_level_hocbf_slack_enabled = false;
cfg.variance_constraint.first_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.first_level_slack_switch_time = 0.90;
% 前期(t<switch) variance 硬、避障软；后期(t>=switch) variance 软、避障硬
% (导师方案，实验③)。hocbf 前硬后软，obstacle 前软后硬。
% First-level obstacle PTCBF: wait until the random source begins to form a
% track-shaped trajectory, then use a soft-to-hard obstacle correction.
cfg.variance_constraint.first_level_obstacle_activation_time = 0.50;
cfg.variance_constraint.first_level_obstacle_phi0 = 2.0;
cfg.variance_constraint.first_level_obstacle_phi1_omega = 0.8/100;
cfg.variance_constraint.first_level_obstacle_phi1_switch_time = 0.0;
cfg.variance_constraint.first_level_obstacle_slack_enabled = true;
cfg.variance_constraint.first_level_obstacle_slack_weight = 10;
cfg.variance_constraint.first_level_obstacle_slack_hard_after_time = 0.97;
% One conservative PTCBF row per controlled point, formed from all active
% first-level obstacle functions and both track-boundary functions.
cfg.variance_constraint.first_level_joint_safety_softmin_enabled = true;
cfg.variance_constraint.first_level_joint_safety_softmin_kappa = 2000.0;
cfg.variance_constraint.first_level_joint_safety_phi1_omega = 0.8/100;
cfg.variance_constraint.second_level_grad_tol = 1e-6;
cfg.variance_constraint.second_level_integral_uncertainty_budget = 30;
cfg.variance_constraint.second_level_hocbf_enabled = 1;
cfg.variance_constraint.second_level_hocbf_alpha2 = 0.5;
cfg.variance_constraint.second_level_hocbf_relaxation_bound = 8;
cfg.variance_constraint.second_level_psi1_margin = 55;
cfg.variance_constraint.second_level_diagnostics = false;
cfg.variance_constraint.second_level_ptcbf_enabled = true;
cfg.variance_constraint.second_level_terminal_variance_beta_final = 10;
cfg.variance_constraint.second_level_terminal_variance_ptzf_initial_margin = 1;
cfg.variance_constraint.second_level_terminal_variance_ptzf_gamma = 0.5;
cfg.variance_constraint.second_level_terminal_variance_alpha = 1.0;
cfg.variance_constraint.second_level_ptclf_enabled = true;
cfg.variance_constraint.second_level_closed_form_solver_enabled = false;
% 第二层保持原来的包络式 PTCLF: Vdot <= cpt*(Vbar - V) + Vbar_dot,
% Vbar(t) = Vbar0*exp(-cg*t/(1-t))。(第三层才换成 SafeFlow 的 FMBF 形式。)
cfg.variance_constraint.second_level_anchor_clf_form = 'envelope';
cfg.variance_constraint.second_level_anchor_clf_ptzf_enabled = true;
cfg.variance_constraint.second_level_anchor_clf_ptzf_cg = 2.5;
cfg.variance_constraint.second_level_anchor_clf_cpt = 50;
cfg.variance_constraint.second_level_anchor_clf_ptzf_initial_margin = 4;
cfg.variance_constraint.second_level_slack_enabled = true;
cfg.variance_constraint.second_level_hocbf_slack_enabled = false;
cfg.variance_constraint.second_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.second_level_anchor_clf_slack_enabled = true;
% Until 0.95 both variance constraints stay hard and PTCLF stays soft.
% At 0.95 the priorities swap: HOCBF/variance PTCBF become soft while
% PTCLF and geometric safety become hard for the terminal handoff.
cfg.variance_constraint.second_level_slack_switch_time = 0.95;
cfg.variance_constraint.second_level_anchor_clf_slack_hard_after_time = 0.95;
cfg.variance_constraint.second_level_obstacle_slack_enabled = true;
cfg.variance_constraint.second_level_obstacle_activation_time = 0.90;
cfg.variance_constraint.second_level_obstacle_activation_times = ...
    [0.9, 0.9, 0.90]; % square, ellipse, right superellipse
% 第二层 obstacle PTCBF 的独立 blow-up 增益：仅作用于第二层 h<0 时的
% phi1=omega/(1-t_eff)^2；不再需要修改三层共用的 cfg.obstacle.phi1_omega。
cfg.variance_constraint.second_level_obstacle_phi0 = 5.0;
cfg.variance_constraint.second_level_obstacle_phi1_omega = 0.1/100;
cfg.variance_constraint.second_level_obstacle_phi1_early_gain = 20.0;
% Independent obstacle safety stays finite; joint soft-min supplies the
% active endpoint prescribed-time blow-up.
cfg.variance_constraint.second_level_obstacle_phi1_switch_time = 1.0;
% One conservative PTCBF per controlled point, formed from all three
% obstacle functions and both global track-boundary functions.
cfg.variance_constraint.second_level_joint_safety_softmin_enabled = true;
cfg.variance_constraint.second_level_joint_safety_softmin_kappa = 2000.0;
cfg.variance_constraint.second_level_joint_safety_phi0 = 15.0;
% Joint safety uses uncapped prescribed-time gain with t_eff=t.
cfg.variance_constraint.second_level_joint_safety_phi1_early_gain = 100.0;
cfg.variance_constraint.second_level_joint_safety_phi1_switch_time = 0.0;
cfg.variance_constraint.second_level_joint_safety_phi1_omega = 0.05/3;
cfg.variance_constraint.second_level_joint_safety_recovery_margin = 0.00;
% [回到 20:58 基线时注释掉, 该值当时未记录] cfg.variance_constraint.second_level_obstacle_phi1_early_gain = 50.0;
% [回到 20:58 基线时注释掉, 该值当时未记录] cfg.variance_constraint.second_level_obstacle_phi1_switch_time = 0.99;
cfg.variance_constraint.second_level_hocbf_slack_weight = 10;
cfg.variance_constraint.second_level_obstacle_slack_weight = 1e5;
cfg.variance_constraint.second_level_obstacle_slack_hard_after_time = 0.95;
cfg.variance_constraint.second_level_terminal_variance_slack_weight = 10;
cfg.variance_constraint.second_level_anchor_clf_slack_weight = 1000;
cfg.variance_constraint.second_level_anchor_snap_flow_steps = 5;
cfg.variance_constraint.second_level_anchor_snap_position_only = false;
cfg.variance_constraint.second_level_anchor_clf_position_only = false;
cfg.variance_constraint.third_level_integral_uncertainty_budget = 5;
% 第三层单独调小梯度退化门槛，减少 HOCBF 那一行在尾段被整行跳过的机会；
% 不写时一二层仍用共用的 cfg.variance_constraint.grad_tol(=1e-6)。
cfg.variance_constraint.third_level_grad_tol = 1e-6;
cfg.variance_constraint.third_level_hocbf_enabled = true;
cfg.variance_constraint.third_level_hocbf_alpha2 = 1;
cfg.variance_constraint.third_level_hocbf_relaxation_bound = 2;
cfg.variance_constraint.third_level_psi1_margin = 0.5;
% Full RK4 sub-stage trace switch. false skips the expensive 4-per-step
% diagnostic aggregation and all plots/animations that require those traces;
% the rollout path and final-state post-processing are still produced.
cfg.variance_constraint.third_level_diagnostics = false;
% Produce three lightweight CBF time-trace figures after the main third-level
% rollout.  Only the selected sample is re-run with full diagnostics, so the
% 1600-sample rollout does not need to retain the expensive RK4-stage trace.
cfg.third_level_cbf_trace_plot_enabled = true;
cfg.third_level_cbf_trace_sample_idx = 4;
% Lightweight trace for the advisor's u plot. This records only mu, v, u
% and the staged control decomposition, without the full HOCBF diagnostics.
cfg.variance_constraint.third_level_control_trace_enabled = false;
cfg.variance_constraint.third_level_ptcbf_enabled = true;
cfg.variance_constraint.third_level_terminal_variance_ptcbf_end_time = 0.8;
cfg.variance_constraint.third_level_terminal_variance_beta_final = 6;
cfg.variance_constraint.third_level_terminal_variance_ptzf_initial_margin = 1.0;
cfg.variance_constraint.third_level_terminal_variance_ptzf_gamma = 0.6;
cfg.variance_constraint.third_level_terminal_variance_alpha = 9.0;
% Continuous endpoint tracking: PTCLF distributes its correction over all
% increment blocks before the internal obstacle PTCBF stage.
cfg.variance_constraint.third_level_ptclf_enabled = true;
% Use quadprog for the third-level weighted minimum-norm QPs. The former
% closed-form backend enumerated all 2^n active sets and dominated rollout
% time once obstacle and track-boundary rows were enabled.
% The third-level cascade contains only a few rows per stage.  Use the
% exact small active-set enumeration instead of quadprog for those weighted
% minimum-norm halfspace problems.
cfg.variance_constraint.third_level_closed_form_solver_enabled = true;
cfg.variance_constraint.third_level_qp_warm_start_enabled = false;
% 串行(增量级联) vs 并行(所有约束进同一个 QP 一次解)。
cfg.variance_constraint.third_level_sequential_increment_qp_enabled = true;
cfg.variance_constraint.third_level_sequential_ptclf_reference_enabled = true;
cfg.variance_constraint.third_level_first_block_control_weight = 40;
% 实验: 给避障 PTCBF 也加首块权重
cfg.variance_constraint.third_level_obstacle_first_block_control_weight = 1;
cfg.variance_constraint.third_level_sequential_ptclf_reference_impl_version = 7;
cfg.variance_constraint.third_level_hocbf_filter_end_time = 0.8;
cfg.variance_constraint.third_level_anchor_clf_form = 'safeflow';
cfg.variance_constraint.third_level_anchor_clf_ptzf_enabled = true;
cfg.variance_constraint.third_level_anchor_clf_phi1_omega = 2.0;
cfg.variance_constraint.third_level_anchor_clf_phi0 = 280;
% 分端点 [首点; 末点]，只在 sequential increment QP 里生效。
cfg.variance_constraint.third_level_anchor_clf_endpoint_phi0 = [30; 30];
cfg.variance_constraint.third_level_anchor_clf_endpoint_phi1_omega = [2.0; 2.0];
cfg.variance_constraint.third_level_slack_enabled = false;
cfg.variance_constraint.third_level_hocbf_slack_enabled = false;
cfg.variance_constraint.third_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.third_level_anchor_clf_slack_enabled = false;
cfg.variance_constraint.third_level_slack_switch_time = inf;
cfg.variance_constraint.third_level_anchor_clf_slack_hard_after_time = 0.0;
cfg.variance_constraint.third_level_obstacle_slack_enabled = false;
cfg.variance_constraint.third_level_obstacle_slack_hard_after_time = 0.0;
% Start with soft obstacle guidance before restoring hard terminal safety.
cfg.variance_constraint.third_level_obstacle_activation_time = 0.5;
cfg.variance_constraint.third_level_obstacle_activation_times = ...
	[0.0, 0.0, 0.0]; % square, ellipse, right superellipse
cfg.variance_constraint.third_level_obstacle_slack_weight = 1e4;
cfg.variance_constraint.third_level_obstacle_phi1_omega = 0.3;
% 障碍内 h<0 时采用分段增益：前段先弱拉回，等 PTCLF 基本收敛后
% 再切换到 omega/(1-t_eff)^2 的 blow-up 增益。
cfg.variance_constraint.third_level_obstacle_phi1_early_gain = 0.2;
cfg.variance_constraint.third_level_obstacle_phi1_switch_time = 0.85;
cfg.variance_constraint.third_level_obstacle_phi0 = 80;
% For each generated interior point, combine all currently active obstacle
% functions and both global track-boundary functions into one conservative
% smooth minimum.  The sequential safety stage therefore solves only three
% hard halfspaces (P2/P3/P4), which are handled by the closed-form backend.
cfg.variance_constraint.third_level_joint_safety_softmin_enabled = true;
cfg.variance_constraint.third_level_joint_safety_softmin_kappa = 2000.0;
cfg.variance_constraint.third_level_joint_safety_phi0 = 20.0;
cfg.variance_constraint.third_level_joint_safety_phi1_early_gain = 15;
cfg.variance_constraint.third_level_joint_safety_phi1_switch_time = 0.0;
cfg.variance_constraint.third_level_joint_safety_phi1_omega =0.2;
cfg.variance_constraint.third_level_joint_safety_phi1_max = inf;
cfg.variance_constraint.third_level_joint_safety_recovery_margin = 0.0;
cfg.variance_constraint.third_level_hocbf_slack_weight = 0.001;
cfg.variance_constraint.third_level_terminal_variance_slack_weight = 1;
cfg.variance_constraint.third_level_anchor_clf_slack_weight = 5000;
cfg.variance_constraint.third_level_anchor_clf_first_slack_weight = 5000;
cfg.variance_constraint.third_level_anchor_clf_last_slack_weight = 1e8;
cfg.variance_constraint.third_level_anchor_snap_flow_steps = 5;
cfg.variance_constraint.third_level_anchor_snap_position_only = false;
cfg.variance_constraint.third_level_anchor_snap_hold_impl_version = 4;
cfg.variance_constraint.third_level_post_endpoint_overwrite_enabled = false;
% The former terminal local-cell projection depended on the same
% phase-search window. Keep it disabled now that boundary classification
% uses global implicit fields with no longitudinal search window.
cfg.variance_constraint.third_level_terminal_feasibility_qp_enabled = false;
cfg.variance_constraint.third_level_terminal_feasibility_qp_version = 2;

%% Scenario Overrides
% Racing uses track-relative obstacle geometry rather than the fixed
% unit-square obstacle above. cfg.obstacle.enabled (defined once above) is
% the sole master switch for both the obstacle and racing scenarios.
if strcmp(cfg.scenario, 'racing')
    % Geometry is populated after the track segment is loaded. This avoids
    % hard-coded unit-square coordinates and keeps obstacle sizes tied to
    % the local racing corridor width.
    cfg.cache.first_level_model_path = fullfile('outputs', ...
        'Racing_FirstLevel_Model.mat');
    cfg.cache.second_level_model_path = fullfile('outputs', ...
        'Racing_SecondLevel_Model.mat');
    cfg.cache.third_level_model_path = fullfile('outputs', ...
        'Racing_ThirdLevel_Model.mat');
    cfg.cache.first_level_hyperparameter_path = fullfile('outputs', ...
        'Racing_FirstLevel_Hyperparameter.mat');
    cfg.cache.second_level_hyperparameter_path = fullfile('outputs', ...
        'Racing_SecondLevel_Hyperparameter.mat');
    cfg.cache.third_level_hyperparameter_path = fullfile('outputs', ...
        'Racing_ThirdLevel_Hyperparameter.mat');
    cfg.cache.first_level_rollout_path = fullfile('outputs', ...
        'Racing_FirstLevel_Rollout_SerialTest.mat');
    cfg.cache.second_level_rollout_path = fullfile('outputs', ...
        'Racing_SecondLevel_Rollout_SerialTest.mat');
    cfg.cache.third_level_rollout_path = fullfile('outputs', ...
        'Racing_ThirdLevel_Rollout_SerialTest.mat');
end
end
