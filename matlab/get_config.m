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
cfg.first_level_generation_samples = 50;
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
cfg.parallel.enabled = true;
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
% Rollout figure marker switch:
% true  = show colored generated-point circles and square/diamond anchors;
% false = draw trajectory lines only.
cfg.output.rollout_markers_enabled = true;

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
cfg.obstacle.enabled   = true;
cfg.obstacle.centers   = [0.485; 0.505];   % near the first-level generated waypoint-3 cluster, shifted right
cfg.obstacle.semi_axes = [0.08; 0.12];     % doubled ellipse width and height
cfg.obstacle.phi0       = 2.0;            % h>=0 时的普通 CBF 系数
cfg.obstacle.phi1_omega = 4.0;            % h<0 时 blow-up phi1=omega/(1-t)^2, 需 >2
cfg.obstacle.slack_enabled = false;
cfg.obstacle.slack_weight  = 100;
% 作用点落到椭圆正中心时 grad = 2*Q*(p-c) 退化为 0，CBF 行没有一阶方向。
% 此时改用一条"沿短轴逃逸"的替代行(见 obstacle_cbf_info)，这里给它的
% 速度。留空 = 自适应取该障碍的短半轴 min(a,b)，即一个时间单位走出椭圆。
cfg.obstacle.escape_speed = [];

% Racing obstacle geometry. Each shape can be enabled and adjusted
% independently; cfg.obstacle.enabled remains the sole master switch.
cfg.obstacle.square.enabled = true;
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

cfg.obstacle.ellipse.enabled = true;
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

cfg.obstacle.superellipse.enabled = true;
cfg.obstacle.superellipse.track_fraction = 0.31;
cfg.obstacle.superellipse.center_x_offset = -0.0025;
cfg.obstacle.superellipse.center_y_offset = -0.006;
cfg.obstacle.superellipse.semi_major_ratio = 2.4;
cfg.obstacle.superellipse.semi_minor_ratio = 0.35;
cfg.obstacle.superellipse.relative_angle = 4 * pi / 180;
cfg.obstacle.superellipse.exponent = 4;
cfg.obstacle.superellipse.constraint_inflation = 0.007;

%% Track Boundary Constraint
% Master switch. Each left/right boundary is represented by a parametric
% pchip spline through all 400 resampled boundary points (= 399 intervals).
% Sparse controls are insufficient for this folded track and make the fitted
% boundary cut across the corridor. The inward
% side is selected automatically from the track centerline, so the same CBF
% works on horizontal, vertical, and folded portions of the racing track.
cfg.track_boundary.enabled = true;
% Bump when the boundary equation changes so cached rollouts are rebuilt.
cfg.track_boundary.implementation_version = 13;
cfg.track_boundary.n_spline_points = 400;
% Prescribed-time boundary CBF.  Safe points use phi0; unsafe points use
% the early gain until the late blow-up branch takes over.
cfg.track_boundary.phi0 = 5.0;
cfg.track_boundary.phi1_early_gain = 5.0;
cfg.track_boundary.phi1_switch_time = 0.85;
cfg.track_boundary.phi1_omega = 0.1;
% Numerical regularization used only when the boundary rows are soft.  A
% hard boundary deliberately uses the uncapped prescribed-time gain.
cfg.track_boundary.phi1_tau_min = 0.01;
% margin 进 h 的定义 (h = n'*(p-q) - margin), 等于把走廊两侧各收窄这么多。
cfg.track_boundary.margin = 0.0033;
cfg.track_boundary.activation_time = 0.0;
% The closest-point search is pure distance minimisation inside a lookup
% window centred on the phase inherited from the parent segment.  The window
% is only there to keep a folded track's other branches out of the search;
% its ends are not physical boundaries and carry no penalty.
cfg.track_boundary.phase_search_half_steps = 0.5;
% Track containment is a safety condition and therefore remains hard.
% Performance rows (in particular the second-level anchor CLF) retain
% slack so that they cannot make the safety QP infeasible.
cfg.track_boundary.slack_enabled = false;
cfg.track_boundary.slack_weight = 1e6;

%% Variance Constraint
cfg.variance_constraint.grad_tol = 1e-6;
% The third-level hard boundary remains exact.  A larger safe-side class-K
% gain only permits faster motion while h >= 0, so the obstacle recovery
% rows are not needlessly opposed by an overly conservative corridor row.
% The prescribed-time recovery gain for h < 0 remains uncapped.
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
% 三层用同一个膨胀量(scale = 1.0)。膨胀保护的是最终交付折线的弦, 而最终折线
% 的点距由第三层决定(赛道长/64 = 0.045); 第一、二层产生的点也全都出现在最终
% 折线里(它们是第三层窗口的端点, 被 snap 硬钉、且不在 third_level_obstacle_
% points = [2 3 4] 之内, 第三层的障碍约束够不着), 所以它们的安全要求与第三层
% 内部点相同。不要按本层弦长去放大: 第一层弦长 0.721, 曲率矢高 d^2/(8R)
% ≈ 0.070, 比走廊全宽 0.046 还大, 那样膨胀会把整片区域堵死, 而且没有意义
% —— 第一层 5 点之间的直线是骨架, 不是轨迹, 后两层会把它填成真正的路径。
cfg.variance_constraint.first_level_obstacle_constraint_inflation_scale = 1.0;
cfg.variance_constraint.second_level_obstacle_enabled = true;
cfg.variance_constraint.second_level_obstacle_points  = [2 3 4];
% 第二层比第三层多膨胀一档: 第二层的点就是第三层窗口的端点, 被 snap 硬钉住
% 且不在 third_level_obstacle_points = [2 3 4] 里, 第三层的障碍约束够不着它们。
% 让它们比第三层的要求多退一点, 接点处的弦才有回旋余地。
% 不要再往上加: scale 1.5 时 square 右缝只剩 0.0063, 小于实测的 square 弦下凹
% 量 0.0084; 而且通道压窄会让第二层自己的修正变大、输出更抖 —— 段边界折角本
% 来就是第三层折角的主因(段边界 0.152 vs 段内 0.041, 3.69 倍)。
cfg.variance_constraint.second_level_obstacle_constraint_inflation_scale = 1.25;
cfg.variance_constraint.third_level_obstacle_enabled  = true;
cfg.variance_constraint.third_level_obstacle_points   = [2 3 4];
cfg.variance_constraint.third_level_obstacle_constraint_inflation_scale = 1.0;
% Per-level boundary switches and affected trajectory points. The master
% cfg.track_boundary.enabled must also be true.
cfg.variance_constraint.first_level_track_boundary_enabled = true;
cfg.variance_constraint.first_level_track_boundary_points = [1 2 3 4 5];
cfg.variance_constraint.second_level_track_boundary_enabled = true;
cfg.variance_constraint.second_level_track_boundary_points = [2 3 4];
cfg.variance_constraint.third_level_track_boundary_enabled = true;
% P1/P5 are snapped exactly to the safe level-2 anchors at the end.  The
% third-level boundary filter owns only the newly generated interior points.
cfg.variance_constraint.third_level_track_boundary_points = [2 3 4];
% 这个窗口名义上只防折叠赛道选错分支, 但实测这一段赛道任意两处相隔 >= 1 个
% 点间距的位置空间距离都 >= 2 倍走廊半宽, 根本不折返 —— 窗口在这里的实际作用
% 是纵向锚定: 点会膨胀到填满给它的任何窗口。3.5 时实测纵向漂移最大 3.2~3.7 个
% 点间距, 弦长被拉到名义 0.0451 的 2.36 倍 (最长 0.1063)。
% 发卡弯的几何极限是 d_max = sqrt(8*R_inner*W) = sqrt(8*0.0144*0.0588) = 0.0823:
% 弦长超过它, 无论端点放哪儿弦都不可能留在走廊内 (所需退让 d^2/(8*R_inner) 已
% 经超过整条走廊宽度), 加 margin 或约束中点都无解。3.5 时正好有一条 0.0832 的
% 弦越过这条线。收到 1.5 把漂移压回 ~1.5 个点间距, 弦长随之回落 (3.5 时实测
% 最大弦长/最大漂移 ≈ 0.64, 按此外推 1.5 下应明显低于 0.0823, 需跑一次确认)。
% 代价: 窗口重新开始夹住 s_near, h 有失真 —— 但 margin 已从 0.0033 提到 0.008,
% 覆盖全赛道 97% 的弦所需退让, 比原来靠失真凑数的状态可靠。
% 1.1 vs 1.5 对照实验(均在 margin=0.008 下): 1.1 边界出界 1 点/2 弦, 1.5 全 0;
% 两者的障碍穿越完全相同(同样的 seg21 P3-P4 和 seg28 P3-P4, square 穿透深度
% -0.00909 到小数点后五位一致) —— 避障约束不读这个窗口。1.1 在 margin=0.0033
% 时代显得更好, 是因为窗口失真在补 margin 的缺口; margin 提到 0.008 后这份
% 失真变成净负担(发卡弯里保守, 其余 43% 的点是乐观方向)。
cfg.variance_constraint.third_level_track_boundary_phase_search_half_steps = 1.1;
cfg.variance_constraint.third_level_track_boundary_activation_time = 0.7;
% All three levels use a soft-to-hard track-boundary schedule.  In the
% third-level cascade the boundary joins the obstacle rows in one safety QP.
cfg.variance_constraint.first_level_track_boundary_slack_enabled = true;
cfg.variance_constraint.first_level_track_boundary_slack_hard_after_time = 0.50;
cfg.variance_constraint.second_level_track_boundary_slack_enabled = true;
cfg.variance_constraint.second_level_track_boundary_slack_hard_after_time = 0.50;
% 第三层配合上面的 activation_time = 0.93: 一激活就是硬约束, 没有软阶段。
% 注意 track_boundary_cbf_info 里 tau 的分支也跟着这个开关走 —— slack 关闭
% 时用 tau = max(1-t_eff, eps), 最后一步 t_eff 精确等于 1, phi1 = omega/eps^2
% 约 2e30。只要那一刻还有 h < 0, 该行右端就是 -2e30, 硬约束下必然不可行。
cfg.variance_constraint.third_level_track_boundary_slack_enabled = false;
cfg.variance_constraint.third_level_track_boundary_slack_hard_after_time = 0.0;
cfg.variance_constraint.first_level_integral_uncertainty_budget = 10;
% First level: all safety/variance constraints enabled; PTCLF remains off.
cfg.variance_constraint.first_level_hocbf_enabled = true;
cfg.variance_constraint.first_level_hocbf_alpha2 = 3.0;
cfg.variance_constraint.first_level_hocbf_relaxation_bound = 5;
cfg.variance_constraint.first_level_psi1_margin = 2;
cfg.variance_constraint.first_level_diagnostics = false;
cfg.variance_constraint.first_level_ptcbf_enabled = true;
cfg.variance_constraint.first_level_terminal_variance_beta_final = 12;
cfg.variance_constraint.first_level_terminal_variance_ptzf_initial_margin = 5;
cfg.variance_constraint.first_level_terminal_variance_ptzf_gamma = 0.3;
cfg.variance_constraint.first_level_terminal_variance_alpha = 8.0;
cfg.variance_constraint.first_level_ptclf_enabled = false;
% Solver backend by level: the first two levels retain MATLAB quadprog,
% while the third-level sequential cascade uses the small-row closed-form
% active-set solver whenever that path is available.
cfg.variance_constraint.first_level_closed_form_solver_enabled = false;
cfg.variance_constraint.first_level_hocbf_slack_enabled = false;
cfg.variance_constraint.first_level_terminal_variance_slack_enabled = false;
% 前期(t<switch) variance 硬、避障软；后期(t>=switch) variance 软、避障硬
% (导师方案，实验③)。hocbf 前硬后软，obstacle 前软后硬。
cfg.variance_constraint.first_level_slack_switch_time = 0.65;
% First-level obstacle PTCBF: wait until the random source begins to form a
% track-shaped trajectory, then use a soft-to-hard obstacle correction.
cfg.variance_constraint.first_level_obstacle_activation_time = 0.80;
cfg.variance_constraint.first_level_obstacle_phi0 = 2.0;
cfg.variance_constraint.first_level_obstacle_phi1_omega = 0.8;
cfg.variance_constraint.first_level_obstacle_slack_enabled = true;
cfg.variance_constraint.first_level_obstacle_slack_weight = 10;
cfg.variance_constraint.first_level_obstacle_slack_hard_after_time = 0.75;
% Second level: obstacle, variance HOCBF/PTCBF, and endpoint PTCLF enabled.
cfg.variance_constraint.second_level_integral_uncertainty_budget = 3;
cfg.variance_constraint.second_level_hocbf_enabled = true;
cfg.variance_constraint.second_level_hocbf_alpha2 = 0.5;
cfg.variance_constraint.second_level_hocbf_relaxation_bound = 8;
cfg.variance_constraint.second_level_psi1_margin = 55;
cfg.variance_constraint.second_level_diagnostics = false;
cfg.variance_constraint.second_level_ptcbf_enabled = true;
cfg.variance_constraint.second_level_terminal_variance_beta_final = 3.0;
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
% hocbf_slack_enabled=false: t < switch 硬，t >= switch 才靠
% hocbf_slack_late_start_time(=slack_switch_time) 打开 slack，实现
% "前硬后软"。
cfg.variance_constraint.second_level_hocbf_slack_enabled = false;
cfg.variance_constraint.second_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.second_level_anchor_clf_slack_enabled = true;
% At 0.65 the variance HOCBF may use slack.  The obstacle PTCBF remains
% soft until 0.85, and the endpoint tracking CLF remains soft throughout.
cfg.variance_constraint.second_level_slack_switch_time = 0.65;
% Keep the tracking CLF soft; hard obstacle and track-boundary rows own the
% late-stage feasible set.
cfg.variance_constraint.second_level_anchor_clf_slack_hard_after_time = inf;
cfg.variance_constraint.second_level_obstacle_slack_enabled = true;
% Let the random flow select a branch before applying late obstacle
% guidance. The bounded soft window avoids a sudden infeasible intersection
% with hard track-boundary rows, then restores a hard terminal safety row.
cfg.variance_constraint.second_level_obstacle_activation_time = 0.93;
cfg.variance_constraint.second_level_obstacle_activation_times = ...
    [0.80, 0.93, 0.93]; % square, ellipse, right superellipse
% 第二层 obstacle PTCBF 的独立 blow-up 增益：仅作用于第二层 h<0 时的
% phi1=omega/(1-t_eff)^2；不再需要修改三层共用的 cfg.obstacle.phi1_omega。
cfg.variance_constraint.second_level_obstacle_phi0 = 8.0;
cfg.variance_constraint.second_level_obstacle_phi1_omega = 0.1;
cfg.variance_constraint.second_level_obstacle_slack_hard_after_time = 0.97;
cfg.variance_constraint.second_level_hocbf_slack_weight = 10;
cfg.variance_constraint.second_level_obstacle_slack_weight = 10;
cfg.variance_constraint.second_level_terminal_variance_slack_weight = 10;
cfg.variance_constraint.second_level_anchor_clf_slack_weight = 30;
cfg.variance_constraint.second_level_anchor_snap_flow_steps = 5;
cfg.variance_constraint.second_level_anchor_snap_position_only = false;
cfg.variance_constraint.second_level_anchor_clf_position_only = false;
cfg.variance_constraint.third_level_integral_uncertainty_budget = 0.5;
% 第三层单独调小梯度退化门槛，减少 HOCBF 那一行在尾段被整行跳过的机会；
% 不写时一二层仍用共用的 cfg.variance_constraint.grad_tol(=1e-6)。
cfg.variance_constraint.third_level_grad_tol = 1e-3;
cfg.variance_constraint.third_level_hocbf_enabled = true;
cfg.variance_constraint.third_level_hocbf_alpha2 = 1;
cfg.variance_constraint.third_level_hocbf_relaxation_bound = 0.5;
cfg.variance_constraint.third_level_psi1_margin = 0.5;
% Full RK4 sub-stage trace switch. false skips the expensive 4-per-step
% diagnostic aggregation and all plots/animations that require those traces;
% the rollout path and final-state post-processing are still produced.
cfg.variance_constraint.third_level_diagnostics = false;
% Lightweight trace for the advisor's u plot. This records only mu, v, u
% and the staged control decomposition, without the full HOCBF diagnostics.
cfg.variance_constraint.third_level_control_trace_enabled = false;
cfg.variance_constraint.third_level_ptcbf_enabled = true;
cfg.variance_constraint.third_level_terminal_variance_ptcbf_end_time = 0.4;
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
cfg.variance_constraint.third_level_closed_form_solver_enabled = false;
cfg.variance_constraint.third_level_qp_warm_start_enabled = false;
% 串行(增量级联) vs 并行(所有约束进同一个 QP 一次解)。
cfg.variance_constraint.third_level_sequential_increment_qp_enabled = true;
cfg.variance_constraint.third_level_sequential_ptclf_reference_enabled = true;
cfg.variance_constraint.third_level_first_block_control_weight = 40;
% 实验: 给避障 PTCBF 也加首块权重
cfg.variance_constraint.third_level_obstacle_first_block_control_weight = 1;
cfg.variance_constraint.third_level_sequential_ptclf_reference_impl_version = 7;
cfg.variance_constraint.third_level_hocbf_filter_end_time = 0.4;
% 第三层 PTCLF 用 SafeFlow 的 FMBF 构造: Vdot <= -phi(t,V)*V，
% phi = phi0 (V<=0, 保持) / omega*(1-t_eff)^-2 (V>0, blow-up)。
% 没有包络，因此不再有 gbar0 / initial_margin。
% ptzf_enabled=false 时全程用 phi0，退化为普通 CLF: Vdot <= -phi0*V。
%
% omega 取 2 有两个独立的理由，而且它们指向同一个值:
%
% 1) 峰值 |u| 最小。约束取等号时 rdot = -phi*r/2, |u| = m + phi*r/2
%    (r=||e||, m 是名义 flow 沿 e 的漂移)。phi = omega*(1+s)^2 按 s 多项式
%    增长、r 按 exp(-omega*s/2) 指数衰减，乘积峰值在 s* = 4/omega - 1，
%    峰值 ~ (8*r0/omega)*exp(omega/2-2)，对 omega 求极值得 argmin = 2。
%    实测(r0=1, m=0.3): omega=1 -> r_end/r0=1.1e-4, peak|u|=2.09;
%                       omega=2 -> r_end/r0=3.7e-6, peak|u|=1.78。
%    即 omega 从 1 调到 2 收敛好 29 倍、峰值 u 反而小 15%。omega 太小时
%    误差被拖到 phi 爆破时仍很大，u 反而更大。
%
% 2) 离散稳定性上界也正好是 2。rollout 积分的是状态 e 而不是 V，
%    对应 rdot = -(phi/2)*r，RK4 要求 dt*phi/2 <~ 2.78，取最后一个 PTCLF
%    生效步(snap 区之前，t=0.935, tau=0.0597, dt=0.00995) -> omega <= 2.0。
%    再大就越过稳定域，V 振荡发散(实测 QP 的 bound 冲到 -1.7e7, exitflag=-2)。
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
cfg.variance_constraint.third_level_obstacle_slack_enabled = true;
% Start with soft obstacle guidance before restoring hard terminal safety.
cfg.variance_constraint.third_level_obstacle_activation_time = 0.5;
% 试过把 superellipse 提前到 0.70(和边界同时), 结果更差: 切入从 1 条变 3 条,
% 间隙 0.00004 -> 0.00002, 第三层耗时 27.5 -> 44.9 s。原因是障碍和边界性质不同:
% 边界是全程性的(每个点每一步都受它管), 早开始是让它有时间把整条轨迹拉进走廊,
% 收益累积; 障碍是局部的, 早开始只是在还离得很远时就施加修正, 把轨迹推向别处,
% 等真正靠近时反而没余量调整, 而且和边界长时间争同一批自由度。所以保持晚开。
cfg.variance_constraint.third_level_obstacle_activation_times = ...
    [0.85, 0.90, 0.93]; % square, ellipse, right superellipse
cfg.variance_constraint.third_level_obstacle_slack_hard_after_time = 0.97;
cfg.variance_constraint.third_level_obstacle_slack_weight = 1e4;
cfg.variance_constraint.third_level_obstacle_phi1_omega = 0.1;
% 障碍内 h<0 时采用分段增益：前段先弱拉回，等 PTCLF 基本收敛后
% 再切换到 omega/(1-t_eff)^2 的 blow-up 增益。
cfg.variance_constraint.third_level_obstacle_phi1_early_gain = 0.05;
cfg.variance_constraint.third_level_obstacle_phi1_switch_time = 0.70;
cfg.variance_constraint.third_level_obstacle_phi0 = 8;
cfg.variance_constraint.third_level_hocbf_slack_weight = 0.001;
cfg.variance_constraint.third_level_terminal_variance_slack_weight = 1;
cfg.variance_constraint.third_level_anchor_clf_slack_weight = 5000;
cfg.variance_constraint.third_level_anchor_clf_first_slack_weight = 5000;
cfg.variance_constraint.third_level_anchor_clf_last_slack_weight = 1e8;
cfg.variance_constraint.third_level_anchor_snap_flow_steps = 5;
cfg.variance_constraint.third_level_anchor_snap_position_only = false;
cfg.variance_constraint.third_level_anchor_snap_hold_impl_version = 4;
cfg.variance_constraint.third_level_post_endpoint_overwrite_enabled = false;
% Terminal feasibility QP: keep snapped P1/P5 fixed and minimally move only
% P2/P3/P4 into their assigned local track cells while retaining all hard
% obstacle inequalities.  No geometric safety margin is added.
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
