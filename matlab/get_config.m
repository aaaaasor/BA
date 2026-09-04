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
% Third-level flow-matching training grid: 100 uniform time slices over
% the complete interval, matching the nominal rollout resolution.
cfg.third_level_n_time_slices = 100;
cfg.third_level_training_time_refine_start_t = inf;
cfg.third_level_training_time_refine_dt = 0.01;
cfg.first_level_time_steps = 100;
cfg.second_level_time_steps = 100;
% The rollout now ends at t=0.99 with no PTZF time shift.  No zero-length
% tail-refinement interval is added at that endpoint.
cfg.second_level_time_refine_start_t = 0.99;
cfg.second_level_time_refine_extra_steps = 00;
cfg.third_level_time_steps = 100;
cfg.third_level_time_refine_start_t = 0;
cfg.third_level_time_refine_extra_steps = 0;
cfg.t_min = 0.0;
% First/second-level physical rollout endpoint (kept unchanged so their
% existing rollout caches remain valid).
cfg.rollout_t_max = 0.996;
% Third level alone stops at 0.995.  Its variance PTZF receives the
% level-specific +0.005 clock shift configured below.
cfg.third_level_rollout_t_max = 0.999;
cfg.random_seed = 7;
cfg.first_level_data_seed = cfg.random_seed + 1;
cfg.first_level_hyperparameter_seed = cfg.random_seed + 2;
cfg.first_level_fit_seed = cfg.random_seed + 3;
cfg.first_level_rollout_seed = cfg.random_seed + 4;
cfg.second_level_data_seed = cfg.random_seed + 5;
cfg.second_level_hyperparameter_seed = cfg.random_seed + 6;
cfg.second_level_fit_seed = cfg.random_seed + 7;
% Dedicated seed for the third-level FM source noise. The archived baseline
% previously inherited the ambient global RNG state at this point.
cfg.third_level_data_seed = 73;
% Alternate only the second-level rollout source noise. The first-level
% targets and third-level rollout randomness remain unchanged.
cfg.second_level_rollout_seed = 215;
cfg.third_level_rollout_seed = 115;
% First-level trajectories use the same deterministic rejection sampler as
% the second level: run the requested complete candidates in parallel, reject a whole
% trajectory only when a hard variance/HOCBF condition fails, then retry the
% rejected trajectory index with its next reproducible seed until the
% requested accepted count is available. Geometric safety is not part of
% seed selection because it is measured independently as an evaluation metric.
cfg.first_level_seed_filter.enabled = 1;
% v4 forces one fresh first-level rollout for this retry-policy revision.
cfg.first_level_seed_filter.implementation_version = 4;
cfg.first_level_seed_filter.base_seed = cfg.first_level_rollout_seed;
cfg.first_level_seed_filter.target_trajectory_count = ...
    cfg.first_level_generation_samples;
cfg.first_level_seed_filter.max_attempts_per_trajectory = 100;
cfg.first_level_seed_filter.tolerance = 1e-8;
% Seed selection is based only on variance constraints.  Geometric safety
% is evaluated afterwards as a reported metric, so it must not bias sampling.
cfg.first_level_seed_filter.final_track_point_filter_enabled = false;
% Retained only as diagnostic metadata while the point filter is disabled.
cfg.first_level_seed_filter.final_track_point_include_margin = true;
cfg.first_level_seed_filter.final_track_point_tolerance = 1e-8;
cfg.first_level_seed_filter.final_obstacle_point_filter_enabled = false;
cfg.first_level_seed_filter.final_obstacle_point_tolerance = 1e-8;
cfg.first_level_seed_filter.final_geometry_internal_points_only = false;
% Build the common first-level PTZF envelope from the actual deterministic
% first-attempt seed population, not the extremely loose GP prior bound.
% The factor leaves headroom for retry seeds while keeping the prescribed-
% time cap tight enough to prevent trajectories drifting deep out of data.
cfg.first_level_seed_filter.initial_beta_envelope_factor = 1.25;
cfg.first_level_seed_filter.parallel_trajectory_batch_size = ...
    cfg.first_level_seed_filter.target_trajectory_count;
% Every second-level segment has its own seed. Freeze accepted segments and advance only
% the seed of a segment that fails the active hard variance conditions.
cfg.second_level_seed_filter.enabled = 1;
cfg.second_level_seed_filter.seed_scope = 'segment';
cfg.second_level_seed_filter.retry_scope = 'segment';
% v14: 20 local attempts, then regenerate the owning first-level parent.
cfg.second_level_seed_filter.implementation_version = 14;
cfg.second_level_seed_filter.base_seed = cfg.second_level_rollout_seed;
cfg.second_level_seed_filter.target_trajectory_count = ...
    cfg.first_level_generation_samples;
cfg.second_level_seed_filter.max_attempts_per_trajectory = 20;
cfg.second_level_seed_filter.max_upstream_retries = 10;
cfg.second_level_seed_filter.upstream_retry_enabled = true;
% In segment-retry mode, the above limit applies independently to each segment.
cfg.second_level_seed_filter.tolerance = 1e-8;
cfg.second_level_seed_filter.final_track_point_filter_enabled = false;
cfg.second_level_seed_filter.final_track_point_tolerance = 1e-8;
cfg.second_level_seed_filter.final_obstacle_point_filter_enabled = false;
cfg.second_level_seed_filter.final_obstacle_point_tolerance = 1e-8;
cfg.second_level_seed_filter.final_geometry_internal_points_only = true;
cfg.second_level_seed_filter.parallel_trajectory_batch_size = ...
    cfg.second_level_seed_filter.target_trajectory_count;
% Each third-level segment has its own seed. Freeze accepted segments and
% retry only the segment that fails the active hard variance conditions.
% Keep the existing filter switch: this behavior applies when enabled.
cfg.third_level_seed_filter.enabled = 1;
cfg.third_level_seed_filter.implementation_version = 9;
cfg.third_level_seed_filter.seed_scope = 'segment';
cfg.third_level_seed_filter.retry_scope = 'segment';
cfg.third_level_seed_filter.base_seed = cfg.third_level_rollout_seed;
cfg.third_level_seed_filter.target_trajectory_count = ...
    cfg.first_level_generation_samples;
cfg.third_level_seed_filter.max_attempts_per_trajectory = 20;
cfg.third_level_seed_filter.max_upstream_retries = 10;
cfg.third_level_seed_filter.upstream_retry_enabled = true;
% In segment-retry mode, the above limit applies independently to each segment.
cfg.third_level_seed_filter.tolerance = 1e-8;
cfg.third_level_seed_filter.final_track_point_filter_enabled = false;
% Retained only as diagnostic metadata while the point filter is disabled.
cfg.third_level_seed_filter.final_track_point_tolerance = 5e-4;
cfg.third_level_seed_filter.final_obstacle_point_filter_enabled = false;
cfg.third_level_seed_filter.final_obstacle_point_tolerance = 1e-8;
cfg.third_level_seed_filter.final_geometry_internal_points_only = true;
cfg.third_level_seed_filter.final_geometry_state_mode = 'increment';
% Line checks are disabled for seed selection; pointwise Safety is evaluated
% after generation and does not inspect connecting edges.
cfg.third_level_seed_filter.final_track_line_filter_enabled = false;
cfg.third_level_seed_filter.final_obstacle_line_filter_enabled = false;
cfg.third_level_seed_filter.final_track_line_tolerance = 1e-8;
cfg.third_level_seed_filter.final_obstacle_line_tolerance = 1e-8;
cfg.third_level_seed_filter.initial_beta_envelope_factor = 1.25;
cfg.third_level_seed_filter.parallel_trajectory_batch_size = ...
    cfg.third_level_seed_filter.target_trajectory_count;
cfg.segment_points_per_segment = 5;
cfg.second_level_generation_samples = 1;
cfg.third_level_generation_samples = 1;
% 第二层实验: 继承第一层避障结果，加入 PTCLF 与障碍规定时间 CBF。
cfg.stop_after_first_level = 0;
cfg.enable_third_level = 1;
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
cfg.animation.enabled = false;
% Explicitly animate the delivered third-level 16-segment rollout instead
% of relying on automatic selection or showing the first-level live view.
cfg.animation.level = 'third';
cfg.animation.trajectory_nr = 1;
cfg.animation.frame_stride = 2;
cfg.animation.delay_time = 0.12;
cfg.animation.third_level_diagnostics_enabled = true;
% 0: 自动选择三级控制链中峰值最大的 segment；1..16: 固定诊断该 segment。
cfg.animation.third_level_diagnostic_segment = 0;

%% Output
cfg.output.enabled = true;
% One master switch for comparison and reproducibility export. When enabled,
% main_demo computes Safety/KL/CS/AS/Time and writes raw third-level variance,
% the applied QP correction u, all three seed layouts, the full configuration,
% and a manifest for hyperparameter/model/rollout caches to one MAT file.
% The u trace is deliberately minimal (u + indices only).
cfg.safe_flow_evaluation.enabled = true;
cfg.safe_flow_evaluation.output_path = fullfile('outputs', ...
    'Racing_SafeFlow_Metrics_Variance_U.mat');
cfg.safe_flow_evaluation.kl_grid_size = 128;
% Build the endpoint-KL KDE reference from the dataset only. Every method
% must reuse this file so that bandwidth, grid, and P are identical.
cfg.safe_flow_evaluation.kl_reference_path = fullfile('outputs', ...
    'Racing_KL_Reference.mat');
cfg.safe_flow_evaluation.kl_reference_rebuild = false;
cfg.safe_flow_evaluation.safety_tolerance = 1e-8;
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
% Replay the most abnormal complete second-level parent trajectories first.
% Each parent owns four segment samples; their final segments accumulate in
% one live figure and one numbered MP4 per parent.
cfg.output.live_second_level_rk4_trajectory_enabled = 0;
cfg.output.live_second_level_rk4_trajectory_parent_samples = ...
    [56, 44];
cfg.output.live_second_level_rk4_trajectory_targets_first = true;
cfg.output.live_second_level_rk4_trajectory_stride = 2;
cfg.output.live_second_level_rk4_trajectory_delay = 0.01;
cfg.output.live_second_level_rk4_trajectory_close_after_save = false;
cfg.output.live_second_level_rk4_video_enabled = true;
cfg.output.live_second_level_rk4_video_frame_rate = 20;
cfg.output.live_second_level_rk4_video_quality = 95;
% Live third-level generation: animate every parent trajectory in order.
% Each parent owns 16 child segments; completed segments accumulate until
% that parent is complete, then the figure is cleared for the next parent.
cfg.output.live_third_level_rk4_trajectory_enabled = 0;
% Only animate these abnormal third-level parent trajectories. Their 16
% child segments are evaluated first, so the requested videos are produced
% without waiting for all preceding parent samples.
cfg.output.live_third_level_rk4_trajectory_parent_samples = ...
    [1, 40, 48, 72, 93, 37, 99, 98, 22, 46];
cfg.output.live_third_level_rk4_trajectory_targets_first = true;
cfg.output.live_third_level_rk4_alternating_segment_colors = false;
cfg.output.live_third_level_rk4_trajectory_stride = 2;
cfg.output.live_third_level_rk4_trajectory_delay = 0.01;
cfg.output.live_third_level_rk4_trajectory_close_after_save = false;
cfg.output.live_third_level_rk4_video_enabled = true;
cfg.output.live_third_level_rk4_video_frame_rate = 20;
cfg.output.live_third_level_rk4_video_quality = 95;
% Rollout figure marker switch:
% true  = show colored generated-point circles and square/diamond anchors;
% false = draw trajectory lines only.
cfg.output.rollout_markers_enabled = 0;
% true = rollout 面板只画"整条折线都不穿障碍"的轨迹, 弦穿越的整条隐藏,
% 标题和控制台会报 保留数/总数。判据用 cfg.obstacle 的物理几何并在每条弦上
% 稠密采样, 所以"点安全但连线切角"的那些会被剔除。
cfg.output.plot_only_obstacle_free_curves = false;
% Evaluate first-level posterior variance along the saved rollout and export
% the normalized/raw uncertainty traces.  When seed filtering already
% produced the same uncertainty array it is reused; otherwise it is queried
% from the fitted first-level GP without rerunning the rollout.
cfg.output.first_level_uncertainty_enabled = 0;
% Skip second-level rollout uncertainty evaluation and its uncertainty
% plots without changing the second-level rollout itself.
cfg.output.second_level_uncertainty_enabled = 1;
% Evaluate and plot the third-level GP predictive mean along every saved
% rollout state, with one panel per output dimension.
cfg.output.third_level_gp_mean_per_output_enabled = false;
% Temporary speed switch: skip third-level rollout uncertainty evaluation
% and its uncertainty plots. This does not change the rollout itself.
cfg.output.third_level_uncertainty_enabled = 0;
% Plot one max-normalized variance panel per GP output dimension.  Each
% panel is divided by its own maximum over all plotted samples and times.
cfg.output.per_output_variance_plot_enabled = false;
% Evaluate and plot GP variance on every third-level training trajectory.
% false skips both the uncertainty calculation and the plot completely.
cfg.output.third_level_training_variance_plot_all = false;
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
% Third level only: independently estimate a full ARD length-scale vector
% per output from short windows near each endpoint, then use
% ell_d(t)=ell_d,1+(ell_d,0-ell_d,1)(1-t)^p inside one GP.
% This does not alter the second-level scalar length-scale schedule.
cfg.gp.third_level_endpoint_length_scale_enabled = true;
cfg.gp.third_level_endpoint_length_scale_window = 0.1;
% Third level only: ell(t)=ell_1+(ell_0-ell_1)(1-t)^p.  With p=2,
% ell_dot(0)=2(ell_1-ell_0) and ell_dot(1)=0.  The second-level scalar
% length-scale schedule remains unchanged.
cfg.gp.third_level_endpoint_length_scale_power = 1.0;
% Absolute cap applied independently to the raw start/end window ARD
% components before the endpoint-wide scale factors below.  The start
% fit has a clean gap: q90=6.73, while 33/420 components exceed 20 and
% reach 8.28e6, so 10 clips the pathological cluster without compressing
% the normal bulk.
cfg.gp.third_level_endpoint_length_scale_cap = 10.0;
cfg.gp.third_level_endpoint_length_scale_floor = 2.0;
% Restore the x2 start enlargement for source-state coverage, while using a
% milder x0.9 end shrinkage than the previous x0.75 experiment.  This keeps
% ell_0 broad without creating as large an endpoint contraction in ell(t).
cfg.gp.third_level_start_length_scale_scale = 2.0;
cfg.gp.third_level_end_length_scale_scale = 0.9;
% Disable the alternative second-level-style global scalar schedule while
% the endpoint-window ARD experiment is active.
cfg.gp.third_level_length_scale_time_varying = false;
cfg.gp.third_level_length_scale_time_scale_start = 1.0;
cfg.gp.third_level_length_scale_time_scale_end = 0.5;
cfg.gp.max_local_data_quantity = 200;
cfg.gp.max_local_gp_quantity = ceil(2.0 * cfg.n_train * cfg.n_time_slices / ...
    cfg.gp.max_local_data_quantity);
% local GP 树相邻叶子之间的重叠区宽度 = (数据范围) * o_ratio。
% 第三层继续使用公共值；第一、第二层在各自建模时使用专用覆盖值。
cfg.gp.o_ratio = 0.05;
cfg.gp.first_level_o_ratio = 0.05;
cfg.gp.second_level_o_ratio = 0.05;
cfg.gp.aggregation_method = 'GPOE';
cfg.gp.first_level_training_accuracy_threshold = 0.8;
cfg.gp.second_level_training_accuracy_threshold = 3.0;
cfg.gp.third_level_training_accuracy_threshold = 0.20;
% Advisor-specified observation-noise standard deviation for every
% third-level GP output (in the standardized training-data coordinates).
cfg.gp.third_level_noise_std = 0.01;
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
cfg.obstacle.square.constraint_inflation = 0.0;

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
cfg.obstacle.ellipse.constraint_inflation = 0.0;

cfg.obstacle.superellipse.enabled = 1;
cfg.obstacle.superellipse.track_fraction = 0.214;
cfg.obstacle.superellipse.center_x_offset = 0.0065500010352428362;
cfg.obstacle.superellipse.center_y_offset = 0.00091350316206417759;
cfg.obstacle.superellipse.semi_major_ratio = 1.0853956186099165;
cfg.obstacle.superellipse.semi_minor_ratio = 0.32774691228613168;
cfg.obstacle.superellipse.relative_angle = -0.0024493419584541432;
cfg.obstacle.superellipse.exponent = 4;
cfg.obstacle.superellipse.constraint_inflation = 0.0;

%% Track Boundary Constraint
cfg.track_boundary.enabled = 1;
% Two fixed global implicit fields h1(x,y), h2(x,y).  They are built once
% from the two rails and queried directly during rollout; no centerline
% location, longitudinal phase, or run-time nearest cross-section is used.
cfg.track_boundary.constraint_method = 'global_implicit_fields';
% Bump when the boundary equation changes so cached rollouts are rebuilt.
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
cfg.track_boundary.margin = 0.0;
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
% Preserve the original first/second-level variance clock.  Third level
% overrides both its theoretical endpoint and shift independently below.
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
cfg.variance_constraint.second_level_obstacle_enabled = 1;
cfg.variance_constraint.second_level_obstacle_points  = [2 3 4];
% Reproduce the earlier second-level setting: do not apply the configured
% obstacle inflation to the second-level PTCBF geometry.
cfg.variance_constraint.second_level_obstacle_constraint_inflation_scale = 1.0;
cfg.variance_constraint.third_level_obstacle_enabled  = 1;
cfg.variance_constraint.third_level_obstacle_points   = [2 3 4];
cfg.variance_constraint.third_level_obstacle_constraint_inflation_scale = 1.1;
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
cfg.variance_constraint.second_level_track_boundary_enabled = 1;
cfg.variance_constraint.second_level_track_boundary_points = [2 3 4];
% Stage-2 late handoff: obstacle and boundary PTCBFs enter at 0.90 with
% slack, then become hard together at 0.95.
cfg.variance_constraint.second_level_track_boundary_activation_time = 0.90;
cfg.variance_constraint.second_level_track_boundary_phi0 = 2.0;
% Combine the two global boundary fields into one conservative smooth
% minimum at level 2.  This replaces two potentially opposing hard rows by
% one row governed mainly by the more dangerous boundary.  kappa=500 adds
% at most log(2)/500 = 1.386e-3 of extra inward conservatism.
cfg.variance_constraint.second_level_track_boundary_combine_method = ...
    'softmin';
cfg.variance_constraint.second_level_track_boundary_softmin_kappa = 2000.0;
cfg.variance_constraint.third_level_track_boundary_enabled = 1;
% P1/P5 are snapped exactly to the safe level-2 anchors at the end.  The
% third-level boundary filter owns only the newly generated interior points.
cfg.variance_constraint.third_level_track_boundary_points = [2 3 4];
cfg.variance_constraint.third_level_track_boundary_margin = 0.0;
cfg.variance_constraint.first_level_track_boundary_phi1_omega = 5/100;
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
cfg.variance_constraint.third_level_track_boundary_phi1_omega = 0.01;
cfg.variance_constraint.third_level_track_boundary_phi1_early_gain = 50.0;
cfg.variance_constraint.third_level_track_boundary_phi1_switch_time = 0.99;
% Keep the prescribed-time gain inside the fixed-step RK4 stability region:
% phi_max*dt = 200*0.00995 = 1.99 < 2.785.
cfg.variance_constraint.third_level_track_boundary_phi1_max = inf;
% Advisor configuration: track-boundary PTCBF is active for the complete
% third-level rollout; the later variance stage supplies the final priority
% while the variance filters remain active.
cfg.variance_constraint.third_level_track_boundary_activation_time = 0.0;
% All three levels use a soft-to-hard track-boundary schedule.  In the
% third-level cascade the boundary joins the obstacle rows in one safety QP.
cfg.variance_constraint.first_level_track_boundary_slack_enabled = true;
cfg.variance_constraint.first_level_track_boundary_slack_hard_after_time = 0.75;
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
cfg.variance_constraint.first_level_terminal_variance_ptzf_initial_margin = 0.1;
cfg.variance_constraint.first_level_terminal_variance_ptzf_gamma = 0.3;
cfg.variance_constraint.first_level_terminal_variance_alpha = 8.0;
cfg.variance_constraint.first_level_ptclf_enabled = false;
% Solver backend by level: the first two levels retain MATLAB quadprog,
% while the third-level sequential cascade uses the small-row closed-form
% active-set solver whenever that path is available.
cfg.variance_constraint.first_level_closed_form_solver_enabled = false;
cfg.variance_constraint.first_level_hocbf_slack_enabled = false;
cfg.variance_constraint.first_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.first_level_slack_switch_time = 0.75;
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
cfg.variance_constraint.first_level_obstacle_slack_hard_after_time = 0.75;
% One conservative PTCBF row per controlled point, formed from all active
% first-level obstacle functions and both track-boundary functions.
cfg.variance_constraint.first_level_joint_safety_softmin_enabled = true;
cfg.variance_constraint.first_level_joint_safety_softmin_kappa = 2000.0;
cfg.variance_constraint.first_level_joint_safety_phi1_omega = 0.5;
cfg.variance_constraint.second_level_grad_tol = 1e-6;
cfg.variance_constraint.second_level_integral_uncertainty_budget = 30;
cfg.variance_constraint.second_level_hocbf_enabled = 1;
cfg.variance_constraint.second_level_hocbf_alpha2 = 0.5;
cfg.variance_constraint.second_level_hocbf_relaxation_bound = 8;
cfg.variance_constraint.second_level_psi1_margin = 55;
cfg.variance_constraint.second_level_diagnostics = false;
cfg.variance_constraint.second_level_ptcbf_enabled = 1;
% 障碍平滑目标(第二层): cost += 0.5*||G*u + e||^2, e/G 来自
% ||p_{i-1} - 2 p_i + p_{i+1}|| 在一小步预测流之后的值。目的是压住"点 6
% 被障碍 CBF 推向自己车道之外"造成的折角——第三层的锚点就是第二层的点，
% 第三层无法再修正它。
% 标定: 实测第二层物理二阶差分中位 0.0908 (第三层是 0.0118, 差 7.7 倍),
% 所以 length_scale 取 0.05 而不是第三层的 0.01, 使 0.5*||e||^2 与最小
% 范数项 0.5*||u||^2 同量级(约 1.7 vs 0.005~0.5)。
% 起效时刻由空间门控决定, 不是 activation_time: 实测第二层的段要到
% t~0.6-0.76 才靠近该障碍(padding=0.025 时中位 0.613, padding=0.010 时
% 0.757), 之前流场还在从噪声收敛, 点根本不在障碍附近。所以时间门设成 0,
% 让空间门自己决定; 而 vicinity_padding 取 0.020 而不是三层用的 0.010,
% 是为了让平滑项早约 0.15 个流时介入, 在折角成形的过程中就起作用, 而不是
% 等形状定了再掰。
% <<< 开关: 第二层要不要加这个 cost。false = 完全回到没有该项的行为
% (level_constraint 里不写字段, 旧的二层 rollout 缓存继续有效)。
cfg.variance_constraint.second_level_obstacle_smoothness_enabled = 0;
cfg.variance_constraint.second_level_obstacle_smoothness = struct( ...
    'enabled', ...
    cfg.variance_constraint.second_level_obstacle_smoothness_enabled, ...
    'obstacle_index', 3, 'weight', 1.0, ...
    'activation_time', 0.0, 'prediction_horizon', 0.05, ...
    'length_scale', 0.05, 'vicinity_padding', 0.020, ...
    'implementation_version', 1);
cfg.variance_constraint.second_level_terminal_variance_beta_final = 3.5;
cfg.variance_constraint.second_level_terminal_variance_ptzf_initial_margin = 0;
cfg.variance_constraint.second_level_terminal_variance_ptzf_gamma = 0.6;
cfg.variance_constraint.second_level_terminal_variance_alpha = 1;
cfg.variance_constraint.second_level_ptclf_enabled = 1;
cfg.variance_constraint.second_level_closed_form_solver_enabled = false;
% 第二层保持原来的包络式 PTCLF: Vdot <= cpt*(Vbar - V) + Vbar_dot,
% Vbar(t) = Vbar0*exp(-cg*t/(1-t))。(第三层才换成 SafeFlow 的 FMBF 形式。)
cfg.variance_constraint.second_level_anchor_clf_form = 'envelope';
cfg.variance_constraint.second_level_anchor_clf_ptzf_enabled = true;
cfg.variance_constraint.second_level_anchor_clf_ptzf_cg = 30;
cfg.variance_constraint.second_level_anchor_clf_cpt = 80;
cfg.variance_constraint.second_level_anchor_clf_ptzf_initial_margin = 1;
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
    [0.9, 0.9, 0.3]; % square, ellipse, right superellipse
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
cfg.variance_constraint.second_level_joint_safety_phi1_early_gain = 2.0;
cfg.variance_constraint.second_level_joint_safety_phi1_switch_time = 0.9;
cfg.variance_constraint.second_level_joint_safety_phi1_omega = 0.05/3;
cfg.variance_constraint.second_level_joint_safety_recovery_margin = 0.00;
% cfg.variance_constraint.second_level_obstacle_phi1_early_gain = 50.0;
% cfg.variance_constraint.second_level_obstacle_phi1_switch_time = 0.99;
cfg.variance_constraint.second_level_hocbf_slack_weight = 10;
cfg.variance_constraint.second_level_obstacle_slack_weight = 1e5;
cfg.variance_constraint.second_level_obstacle_slack_hard_after_time = 0.95;
cfg.variance_constraint.second_level_terminal_variance_slack_weight = 10;
cfg.variance_constraint.second_level_anchor_clf_slack_weight = 1000;
cfg.variance_constraint.second_level_anchor_snap_flow_steps = 4;
cfg.variance_constraint.second_level_anchor_snap_position_only = false;
cfg.variance_constraint.second_level_anchor_clf_position_only = false;
cfg.variance_constraint.third_level_integral_uncertainty_budget = 5;
% 第三层单独调小梯度退化门槛，减少 HOCBF 那一行在尾段被整行跳过的机会；
% 不写时一二层仍用共用的 cfg.variance_constraint.grad_tol(=1e-6)。
cfg.variance_constraint.third_level_grad_tol = 1e-6;
cfg.variance_constraint.third_level_hocbf_enabled = false;
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
cfg.third_level_cbf_trace_plot_enabled = false;
cfg.third_level_cbf_trace_sample_idx = 4;
% After the lightweight all-sample uncertainty evaluation, automatically
% re-run the sample with the largest beta-cap violation using full RK4
% diagnostics and export the mechanism traces requested by the advisor.
cfg.third_level_variance_violation_diagnostic_enabled = true;
% Lightweight trace for the advisor's u plot. This records only mu, v, u
% and the staged control decomposition, without the full HOCBF diagnostics.
cfg.variance_constraint.third_level_control_trace_enabled = false;
cfg.variance_constraint.third_level_ptcbf_enabled = 1;
cfg.variance_constraint.third_level_terminal_variance_ptcbf_end_time = 0.9;
cfg.variance_constraint.third_level_terminal_variance_beta_final = 3.0;
% Impose the terminal variance PTCBF on each GP output separately rather
% than on the summed sigma^2.  Measured motivation: near t=1 the summed
% gradient is ~1.6x shorter than the sum of the per-output gradient
% norms, so aggregating throws away control authority exactly where the
% constraint is hardest to meet.  The scalar budget is split across
% outputs in proportion to their prior variance.
cfg.variance_constraint.third_level_terminal_variance_per_dimension = 0;
cfg.variance_constraint.third_level_terminal_variance_ptzf_initial_margin = 1.0;
cfg.variance_constraint.third_level_terminal_variance_ptzf_gamma = 1.0;
% Third-level variance clock: physical t=0.995 maps to theoretical t=1.
cfg.variance_constraint.third_level_terminal_variance_ptzf_terminal_time = 1.0;
cfg.variance_constraint.third_level_terminal_variance_ptzf_time_shift = 0.005;
% Third-level-only time-varying class-K term.  A first-order blow-up is
% deliberately used here: the paper's second-order form was numerically too
% stiff for the hard, slack-free discrete RK4 rollout.
cfg.variance_constraint.third_level_terminal_variance_alpha = 1.0;
cfg.variance_constraint.third_level_terminal_variance_time_varying_alpha = false;
cfg.variance_constraint.third_level_terminal_variance_time_varying_alpha_power = 2.0;
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
% 障碍/边界 safety filter 共用的首块控制权重。
cfg.variance_constraint.third_level_safety_first_block_control_weight = 40;
% Experimental local bending COST (no added hard rows or slack). Applies
% only near physical obstacle 3 in the third-level safety QP. Position maps
% are physical XY; length_scale normalizes the squared second difference.
% Disable, or set weight=0, for the unchanged legacy safety objective.
% 与第二层同一套标定逻辑: length_scale 取该层二阶差分的量级(第三层实测
% 中位 0.0118), weight=1 时 0.5*||e||^2 约 0.07, 与最小范数项可比。
% 原先的 weight=100 / length_scale=0.01 使该项达到 69~1861, 比最小范数
% 项大 2~3 个数量级, gate 区域内等于只优化弯曲、完全忽略贴合 GP 流场。
% vicinity_padding 从 0.025 降到 0.010: 0.025 会让 gate 的横向半径变成
% 0.033, 是局部走廊半宽 0.0227 的 1.5 倍, 横向门控完全失效。
% <<< 开关: 第三层要不要加这个 cost。语义同第二层。
cfg.variance_constraint.third_level_obstacle_smoothness_enabled = 0;
cfg.variance_constraint.third_level_obstacle_smoothness = struct( ...
    'enabled', ...
    cfg.variance_constraint.third_level_obstacle_smoothness_enabled, ...
    'obstacle_index', 3, 'weight', 1000.0, ...
    'activation_time', 0.0, 'prediction_horizon', 0.05, ...
    'length_scale', 0.02, 'vicinity_padding', 0.010, ...
    'implementation_version', 2);
% Variance HOCBF/PTCBF post-filter: penalize correction on the shared S0
% block so uncertainty reduction does not primarily translate the first
% physical point and the whole cumulative segment.
cfg.variance_constraint.third_level_variance_first_block_control_weight = 3;
cfg.variance_constraint.third_level_sequential_ptclf_reference_impl_version = 12;
cfg.variance_constraint.third_level_hocbf_filter_end_time = 0.85;
cfg.variance_constraint.third_level_anchor_clf_form = 'safeflow';
cfg.variance_constraint.third_level_anchor_clf_ptzf_enabled = true;
cfg.variance_constraint.third_level_anchor_clf_phi1_omega = 1.0;
cfg.variance_constraint.third_level_anchor_clf_phi1_early_gain = 30;
cfg.variance_constraint.third_level_anchor_clf_phi1_switch_time = 0.8;
cfg.variance_constraint.third_level_anchor_clf_phi1_max = inf;
cfg.variance_constraint.third_level_anchor_clf_phi0 = 280;
% 分端点 [首点; 末点]，只在 sequential increment QP 里生效。
cfg.variance_constraint.third_level_anchor_clf_endpoint_phi0 = [30; 30];
cfg.variance_constraint.third_level_anchor_clf_endpoint_phi1_omega = [2; 2];
cfg.variance_constraint.third_level_anchor_clf_endpoint_phi1_early_gain = ...
    [80;80];
cfg.variance_constraint.third_level_anchor_clf_endpoint_phi1_max = ...
    [inf; inf];
cfg.variance_constraint.third_level_slack_enabled = false;
cfg.variance_constraint.third_level_hocbf_slack_enabled = false;
cfg.variance_constraint.third_level_terminal_variance_slack_enabled = false;
cfg.variance_constraint.third_level_anchor_clf_slack_enabled = false;
cfg.variance_constraint.third_level_slack_switch_time = inf;
cfg.variance_constraint.third_level_anchor_clf_slack_hard_after_time = 0.0;
cfg.variance_constraint.third_level_obstacle_slack_enabled = false;
cfg.variance_constraint.third_level_obstacle_slack_hard_after_time = 0.0;
% Activate every obstacle from the beginning.  The per-obstacle vector must
% match the scalar time or it overrides it.
cfg.variance_constraint.third_level_obstacle_activation_time = 0.0;
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
cfg.variance_constraint.third_level_joint_safety_phi0 = 2.0;
cfg.variance_constraint.third_level_joint_safety_phi1_early_gain = 0.5;
cfg.variance_constraint.third_level_joint_safety_phi1_switch_time = 0.0;
cfg.variance_constraint.third_level_joint_safety_phi1_omega = 0.53;
cfg.variance_constraint.third_level_joint_safety_phi1_max = inf;
cfg.variance_constraint.third_level_joint_safety_recovery_margin = 0.0;
cfg.variance_constraint.third_level_hocbf_slack_weight = 0.001;
cfg.variance_constraint.third_level_terminal_variance_slack_weight = 1;
cfg.variance_constraint.third_level_anchor_clf_slack_weight = 5000;
cfg.variance_constraint.third_level_anchor_clf_first_slack_weight = 5000;
cfg.variance_constraint.third_level_anchor_clf_last_slack_weight = 1e8;
% Four snap steps start at t=0.95616, comfortably after both independent
% variance filters have switched off at t=0.8.
cfg.variance_constraint.third_level_anchor_snap_flow_steps = 5;
cfg.variance_constraint.third_level_anchor_snap_position_only = false;
cfg.variance_constraint.third_level_anchor_snap_hold_impl_version = 6;
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

