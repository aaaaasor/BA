# 当前方法与实验设置总结

> **配置快照日期：2026-08-30**
> 本文只描述当前代码实际生效的设置。参数来源以 `matlab/get_config.m` 及其构造出的逐层 constraint 为准；历史扫描值和已关闭的试验项不算入当前方法。

## 1. 任务与总体思路

任务是在 Nürburgring GP-Strecke 的二维赛道片段上学习轨迹分布，并生成满足低不确定性、赛道边界和障碍物要求的轨迹。方法采用三层 coarse-to-fine flow matching：每一层都由局部专家 GP（LoG-GP）预测生成流场，并在 rollout 时通过 QP 添加修正量 `u`。

生成动力学写为

\[
\dot x=\mu_{\mathrm{GP}}(x,t)+u(x,t),
\]

其中 `mu_GP` 是 GP 预测的 flow-matching 速度，`u` 是 QP 在**标准化轨迹空间**中的速度修正量。`u` 不是车辆的转向角、加速度或其他物理控制输入。

当前三层细化结构为：

| 层级 | 表示与作用 | 每条最终轨迹的点数 | 每个父轨迹的 rollout 单元 |
|---|---|---:|---:|
| 第一层 | 直接生成 5 个粗粒度锚点；每点特征为 `[x,y,dx/ds,dy/ds]` | 5 | 1 条完整粗轨迹 |
| 第二层 | 在第一层相邻锚点间各生成 5 点绝对坐标段，相邻段共享端点 | 17 | 4 段 |
| 第三层 | 将第二层每个间隔再细分；每段用 `[S0,S1-S0,...,S4-S3]` 增量表示 | 65 | 16 段 |

最终每条轨迹包含 65 个生成点。默认生成 100 条最终轨迹。

## 2. 数据、时间离散与复现设置

| 参数 | 当前值 |
|---|---:|
| 场景 | `racing` |
| 数据文件 | `trajectory_data/track_dataset_arena.mat` |
| 训练轨迹数 `n_train` | 30 |
| 第一、二层训练时间切片 | 15 |
| 第三层训练时间切片 | 100，均匀覆盖 `[0,1]` |
| 第一、二、三层 RK4 步数 | 均为 100 |
| 实际 rollout 区间 | `[0,0.996]` |
| 最终生成轨迹数 | 100 |
| 每段点数 | 5 |
| 第三层窗口 stride | 4（非重叠窗口，共 16 段） |
| 全局随机种子 | 7 |
| 并行计算 | 开启，6 workers；不可用时退回串行 |

分层随机种子由全局种子确定：第一层 data/hyperparameter/fit/rollout 分别为 8/9/10/11；第二层分别为 12/13/14/15；第三层 rollout 基准 seed 为 115。

## 3. LoG-GP 流场模型

所有层采用局部 GP 专家并通过 GPOE 聚合。公共参数为：每个局部专家最多使用 200 条数据，专家重叠比例 `o_ratio=0.05`，聚合方式为 `GPOE`。

| 参数 | 第一层 | 第二层 | 第三层 |
|---|---:|---:|---:|
| 预训练样本数 | 450 | 500 | 800 |
| 训练精度阈值 | 0.8 | 3.0 | 0.20 |
| overlap ratio | 0.05 | 0.05 | 0.05 |
| length scale | 普通拟合 | 随时间标量缩放 `1.0 -> 0.5` | 端点窗口 ARD 插值 |
| observation noise std | 模型默认 | 模型默认 | 0.01（标准化坐标） |

第三层分别使用靠近 `t=0` 和 `t=1` 的长度为 0.1 的时间窗口拟合完整 ARD 向量 `ell_0`、`ell_1`，然后在同一个 GP 内插值：

\[
\ell_d(t)=\ell_{d,1}+(\ell_{d,0}-\ell_{d,1})(1-t)^p,
\qquad p=1.
\]

原始 ARD 分量截断到 `[2,10]`，随后起点向量乘 2.0、终点向量乘 0.9。第三层普通的标量 length-scale schedule 已关闭。

## 4. QP 的基本形式

QP 的默认目标是最小化对 GP 流场的修正：

\[
\min_u\ \frac12\lVert Wu\rVert_2^2,
\]

并同时满足当前时刻启用的积分方差 HOCBF、末端方差 PTCBF、锚点 PTCLF、障碍物 PTCBF 和赛道边界 PTCBF 线性不等式。第一、二层使用 MATLAB `quadprog`；第三层使用顺序增量 QP 和小规模 active-set 半空间求解器。

### 4.1 积分方差 HOCBF

令 GP 总预测方差为 `beta(t)`，累计方差为 `I(t)=int_0^t beta(tau)dtau`，预算为 `B`。代码使用

\[
\psi_0=Bt-I(t)+\bar h,
\qquad
\psi_1=\dot\psi_0+\alpha_1\psi_0,
\qquad
\dot\psi_1+\alpha_2\psi_1\ge 0.
\]

`alpha_1` 不是固定超参数，而是对每个样本根据初始方差自动计算，使 `psi_1(0)` 至少达到配置的 `psi1_margin`。

### 4.2 末端方差 PTCBF

目标是

\[
\beta(t)\le \beta_{\mathrm{final}}+\bar h_T(t),
\]

其中

\[
\bar h_T(t)=\bar h_0
\exp\!\left[-\gamma\frac{t_{\mathrm{eff}}}{1-t_{\mathrm{eff}}}\right].
\]

第一、二层采用 `t_eff=t/0.996`，因此实际 `t=0.996` 对应方差理论时钟 1。第三层采用

\[
t_{\mathrm{eff}}=(t+0.005)/1.0,
\]

但该层末端方差 PTCBF 只启用到物理时间 `t<0.85`，之后释放。

### 4.3 锚点 PTCLF

第二层使用包络形式：

\[
V=\lVert e\rVert^2,\qquad
\dot V\le c_{pt}(\bar V-V)+\dot{\bar V},
\]

\[
\bar V(t)=\bar V_0\exp\!\left[-c_g\frac{t_{\mathrm{eff}}}{1-t_{\mathrm{eff}}}\right].
\]

第三层使用 SafeFlow/FMBF 形式，对首末端误差施加

\[
\dot V\le-\phi(t)V,
\]

并在后段使用 `phi(t)=omega/(1-t)^2`。第三层首先根据 P1/P5 的 PTCLF 求参考修正，再在该参考上求 P2/P3/P4 的障碍与边界安全修正，最后处理全局方差行。

## 5. 三层方差与锚点参数

| 参数 | 第一层 | 第二层 | 第三层 |
|---|---:|---:|---:|
| 积分方差预算 `B` | 10 | 30 | 5（HOCBF 当前关闭） |
| HOCBF | 开 | 开 | **关** |
| `alpha2` | 3.0 | 0.5 | 1（不生效） |
| relaxation bound | 5 | 8 | 2（不生效） |
| `psi1_margin` | 2 | 55 | 0.5（不生效） |
| 末端方差 PTCBF | 开 | 开 | 开，仅 `t<0.85` |
| `beta_final` | 0.1 | 3.5 | 3.0 |
| PTZF initial margin | 3 | 0.1 | 1.0 |
| PTZF `gamma` | 0.3 | 0.3 | 1.0 |
| terminal class-K `alpha` | 8.0 | 0.3 | 1.0 |
| terminal variance per dimension | 关 | 关 | 关（对总方差使用一行） |
| 锚点 PTCLF | 关 | 开，envelope | 开，SafeFlow |

第二层锚点 PTCLF 参数：`c_g=15.5`、`c_pt=350`、initial margin=1；PTCLF 在 `t<0.95` 可使用 slack，`t>=0.95` 后变硬。第二层最后 4 个 flow steps 执行 anchor snap。

第三层锚点 PTCLF 参数：全局 `phi0=280`、`omega=1`、early gain=30、switch time=0.8；顺序 QP 对 P1/P5 分别覆盖为 `phi0=[30,30]`、`omega=[2,2]`、early gain=`[80,80]`，均不设上限。首个增量块在 PTCLF 和 safety 两级中的控制权重均为 40，方差后处理中的首块权重为 3。最后 5 个 flow steps执行 endpoint snap-and-hold。第三层所有约束均不使用 slack。

方差约束的软硬时序：第一层在 `t>=0.90` 后允许方差 HOCBF/PTCBF slack；第二层在 `t>=0.95` 后允许方差 HOCBF/PTCBF slack；第三层无 slack，但末端方差 PTCBF 在 `t=0.85` 关闭。

## 6. 障碍物与赛道边界

### 6.1 障碍物几何

障碍物采用可旋转超椭圆。令 `q=R^T(p-c)`，半轴为 `a_1,a_2`，指数为 `n`。安全函数满足 `h>=0` 为障碍外部：

\[
h(p)=
\begin{cases}
\sum_i|q_i/a_i|^n-1,&n\le2,\\
\left(\sum_i|q_i/a_i|^n\right)^{1/n}-1,&n>2.
\end{cases}
\]

当前启用三个障碍物，位置和尺寸在加载赛道后由赛道局部宽度解析：

| 障碍物 | 当前配置 |
|---|---|
| 1：平滑菱形/方形 | track fraction 0.72；half-size ratio 2.1；length ratio 1.5；offset `(0.015,0.020)`；global angle 0；exponent 1.2；inside ratio 0.50；inflation 0.010 |
| 2：椭圆 | track fraction 0.52；中心覆盖为 `(0.30,0.96)`；semi-major ratio 4.5；semi-minor ratio 1.8；relative angle 0；inflation 0.006 |
| 3：右侧超椭圆 | track fraction 0.214；offset `(0.0065500010,0.0009135032)`；semi-major ratio 1.085395619；semi-minor ratio 0.327746912；relative angle -0.002449342 rad；exponent 4；inflation 0.005 |

第三个障碍物解析后的当前物理尺寸约为全长 0.0616、全宽 0.0186，中心约为 `(0.5748,0.4067)`；代码中的赛道相对参数是复现时的权威值。

障碍 CBF 采用

\[
\dot h+\phi(t,h)h\ge0.
\]

安全侧使用有限 `phi0`；不安全侧先使用 early gain，再在 switch time 后使用

\[
\phi_1(t)=\frac{\omega}{(1-t)^2}.
\]

几何 PTCBF 使用原始物理时间，实际只运行到 `t=0.996`，没有像前两层方差时钟那样归一化到 1。

### 6.2 赛道边界

赛道边界使用左右轨建立的两个全局隐式场 `h_left(x,y)`、`h_right(x,y)`，方法为 `global_implicit_fields`，样条为 C2 `spline`，构造点数 400。控制器内部使用

\[
h_{\mathrm{ctrl}}=h_{\mathrm{raw}}-m,
\]

公共 margin 为 0.003；第三层单独使用 0.001。终端 safety projection 当前关闭。

### 6.3 分层几何约束参数

| 设置 | 第一层 | 第二层 | 第三层 |
|---|---|---|---|
| 受控点 | P1–P5 | P2–P4 | P2–P4 |
| 边界 activation | 0.50 | 0.90 | 0.0 |
| 边界 `phi0` | 1（继承公共值） | 2 | 20 |
| 边界 `omega` | 0.001 | 0.001 | 0.01 |
| 边界 early/blow-up switch | 0.0 | 1.0（独立行不进入 blow-up） | 0.99 |
| 边界 slack | `t<0.90` | `t<0.95` | 无 |
| 障碍 activation | 0.50 | `[0.90,0.90,0.30]` | `[0,0,0]` |
| 障碍 `phi0` | 2 | 5 | 80 |
| 障碍 `omega` | 0.008 | 0.001 | 0.3 |
| 障碍 early gain / switch | 直接 blow-up / 0.0 | 20 / 1.0 | 0.2 / 0.85 |
| 障碍 slack | `t<0.97` | `t<0.95` | 无 |

每个受控点将所有启用的障碍物和两侧赛道边界通过 soft minimum 合并为一条 joint safety 约束：

\[
h_{\mathrm{soft}}=-\frac1\kappa\log\sum_j\exp(-\kappa h_j),
\qquad \kappa=2000.
\]

分层 joint safety 参数为：第一层 `omega=0.008`；第二层 `phi0=15`、early gain=2、switch=0.9、`omega=0.05/3`；第三层 `phi0=20`、early gain=2、switch=0.93、`omega=0.01`。所有 `phi1_max=Inf`。

**实际进入 QP 时需要注意：**当前三层都打开了 joint soft-min。`joint_safety_softmin_info` 用 joint row 替换单独的 obstacle 和 boundary rows，并把 joint row 作为 `obstacle` 类型交给求解器。因此 joint row 的软硬切换采用本层的 **obstacle slack schedule**；表中独立 boundary slack schedule 不再单独产生一组 boundary QP rows。边界 activation 仍决定边界分量何时加入 joint soft-min。

## 7. 各层实际 QP、约束时序与求解器

### 7.1 软约束与硬约束的统一定义

将当前 QP 行分成硬约束集合 `H` 和软约束集合 `S`。代码实际求解

\[
\begin{aligned}
\min_{u,\delta}\quad &
\frac12\lVert Wu\rVert_2^2+
\frac12\sum_{i\in S}w_i\delta_i^2,\\
\text{s.t.}\quad &A_i u\le b_i, &&i\in H,\\
&A_i u-\delta_i\le b_i,\quad \delta_i\ge0, &&i\in S.
\end{aligned}
\]

本文中的“硬”表示该行没有 slack；“软”表示允许非负 slack，但违反会按相应权重进入 cost。QP 每一行先按其梯度范数归一化，slack 权重再乘回行范数平方，以保持原约束单位下的惩罚尺度。

若所有约束在 `u=0, delta=0` 时已经满足，代码直接返回零修正，不调用数值求解器。这是零解可行的解析快捷路径，不代表该层整体使用闭式求解。

### 7.2 第一层：单个联合 QP

第一层的每个 RK4 子阶段把所有当前启用的约束放进同一个 QP。状态是 5 个粗锚点的绝对标准化特征，控制权重 `W=I`，PTCLF 关闭。

| 物理时间 | 实际进入 QP 的约束 | 软/硬 | slack 权重 |
|---|---|---|---:|
| `0 <= t < 0.50` | 积分方差 HOCBF；末端方差 PTCBF | 硬；硬 | 无 |
| `0.50 <= t < 0.90` | 上述两条；P1–P5 的 joint obstacle+boundary PTCBF | 方差两条硬；joint safety 软 | joint 10 |
| `0.90 <= t < 0.97` | 同上 | 方差 HOCBF/PTCBF 软；joint safety 软 | 方差各 10000；joint 10 |
| `0.97 <= t <= 0.996` | 同上 | 方差 HOCBF/PTCBF 软；joint safety 硬 | 方差各 10000 |

具体说明：

- 积分方差 HOCBF 从 `t=0` 开始，整个 rollout 都生成该行；`t=0.90` 起允许 slack。
- 末端方差 PTCBF 从 `t=0` 开始，整个 rollout 都生成该行；`t=0.90` 起允许 slack。其理论方差时钟在物理 `t=0.996` 到达 1，但末段是软约束。
- 三个障碍和赛道边界都在 `t=0.50` 激活，并通过 `kappa=2000` 的 joint soft-min 对 P1–P5 各形成一条 safety row。
- joint safety 的不安全侧从激活时刻起直接使用 blow-up 分支，`omega=0.008`；`t<0.97` 为软约束，之后为硬约束。
- 第一层没有 anchor PTCLF、没有 snap-and-hold、没有平滑 cost。

**QP 求解器：**`first_level_closed_form_solver_enabled=false`，因此一般使用 MATLAB `quadprog`；若 interior-point 失败，代码先用 `linprog` 为硬约束寻找可行点，再用 active-set `quadprog` 重求。除零解快捷路径外，第一层不是闭式解。

### 7.3 第二层：每段一个联合 QP

第二层对第一层 4 个相邻锚点区间分别生成 5 点绝对坐标段。每个 RK4 子阶段仍是一个联合 QP，控制权重 `W=I`。P2–P4 受几何约束；P1/P5 由 anchor PTCLF 拉向第一层相邻锚点。

| 物理时间 | 实际进入 QP 的约束 | 软/硬 | slack 权重 |
|---|---|---|---:|
| `0 <= t < 0.30` | 积分方差 HOCBF；末端方差 PTCBF；P1/P5 anchor PTCLF | 方差两条硬；PTCLF 软 | PTCLF 1000 |
| `0.30 <= t < 0.90` | 上述约束；仅障碍物 3 加入 P2–P4 joint safety | 方差两条硬；PTCLF 软；joint safety 软 | PTCLF 1000；joint 100000 |
| `0.90 <= t < 0.95` | 三个障碍物和两侧边界全部加入 joint safety | 方差两条硬；PTCLF 软；joint safety 软 | PTCLF 1000；joint 100000 |
| `0.95 <= t < 0.95616` | 同上 | 方差 HOCBF/PTCBF 软；PTCLF 硬；joint safety 硬 | 方差各 10 |
| `0.95616 <= t <= 0.996` | 方差 HOCBF/PTCBF；joint safety；P1/P5 endpoint hold | 方差两条软；joint safety 硬；endpoint hold 为硬等式；PTCLF 关闭 | 方差各 10 |

具体说明：

- 积分方差 HOCBF 和末端方差 PTCBF 都从 `t=0` 开始；`t=0.95` 起变为软约束。
- 末端方差的理论时钟在物理 `t=0.996` 到达 1，但该时刻允许 variance slack。
- anchor PTCLF 使用 envelope 形式，`c_g=15.5`、`c_pt=350`，在 `t<0.95` 为软约束，`0.95<=t<0.95616` 为硬约束。
- 障碍 3 在 `t=0.30` 激活；障碍 1、2 与赛道边界在 `t=0.90` 激活。当前 joint safety row 采用 obstacle schedule，因此在 `t<0.95` 软、之后硬。
- 最后 4 个 RK4 flow steps 从 `t=0.95616` 开始：P1/P5 被投影到第一层 anchor 并保持，PTCLF 关闭，endpoint hold 作为硬等式进入 QP。
- 第二层局部平滑 cost 当前关闭。

**QP 求解器：**`second_level_closed_form_solver_enabled=false`，一般使用 MATLAB `quadprog`，失败回退逻辑与第一层相同。除零解快捷路径外，第二层不是闭式解。

### 7.4 第三层：三级顺序 QP cascade

第三层采用增量状态 `[S0,S1-S0,...,S4-S3]`。每个 RK4 子阶段不把所有目标一次性放入一个 QP，而是按顺序构造完整控制

\[
u=u_{\mathrm{PTCLF}}+\Delta u_{\mathrm{safety}}+
\Delta u_{\mathrm{variance}}.
\]

#### Stage A：端点 PTCLF reference

对 P1/P5 的两个 SafeFlow PTCLF halfspaces 求

\[
u_{\mathrm{PTCLF}}=
\arg\min_u\frac12\lVert W_Au\rVert^2
\quad\text{s.t.}\quad
A_{\mathrm{PTCLF}}u\le b_{\mathrm{PTCLF}}.
\]

首个增量块权重为 40，其余为 1。PTCLF 从 `t=0` 开启，到 endpoint snap 区开始时关闭；全程无 slack。

#### Stage B：joint obstacle+boundary safety filter

以 Stage A 的完整控制为 reference，只求满足 P2/P3/P4 joint safety rows 所需的最小附加修正：

\[
\Delta u_{\mathrm{safety}}=
\arg\min_{\Delta u}\frac12\lVert W_B\Delta u\rVert^2
\quad\text{s.t.}\quad
A_{\mathrm{safety}}(u_{\mathrm{PTCLF}}+\Delta u)\le b_{\mathrm{safety}}.
\]

三个障碍物和两侧边界都从 `t=0` 开启，joint safety 始终为硬约束。首块权重为 40，其余为 1。平滑 cost 当前关闭，因此 Stage B 的 cost 里没有二阶差分项。

#### Stage C：末端方差 post-filter

以 Stage B 的结果为 reference，再求

\[
\Delta u_{\mathrm{variance}}=
\arg\min_{\Delta u}\frac12\lVert W_C\Delta u\rVert^2
\quad\text{s.t.}\quad
A_{\mathrm{var}}(u_{\mathrm{PTCLF}}+
\Delta u_{\mathrm{safety}}+\Delta u)\le b_{\mathrm{var}}.
\]

当前第三层积分方差 HOCBF 关闭，因此 `A_var` 只含总预测方差的 terminal PTCBF 一行。该行在 `0<=t<0.85` 启用、为硬约束；`t>=0.85` 后整行移除，而不是改成软约束。Stage C 的首块权重为 3，其余为 1。

#### 第三层完整时间表

| 物理时间 | Stage A：PTCLF | Stage B：joint safety | Stage C：variance | 端点处理 |
|---|---|---|---|---|
| `0 <= t < 0.80` | 硬；early gain 80（P1/P5） | 硬；early gain 2 | terminal variance PTCBF 硬 | 无 |
| `0.80 <= t < 0.85` | 硬；`omega=2` blow-up | 硬；early gain 2 | terminal variance PTCBF 硬 | 无 |
| `0.85 <= t < 0.93` | 硬；blow-up | 硬；early gain 2 | **关闭** | 无 |
| `0.93 <= t < 0.94620` | 硬；blow-up | 硬；`omega=0.01` blow-up | 关闭 | 无 |
| `0.94620 <= t <= 0.996` | **关闭** | 硬；blow-up | 关闭 | P1/P5 投影并施加硬速度等式 |

注：`0.94620` 来自 100 步、终点 0.996、最后 5 个 flow steps 的当前时间网格。RK4 的 k2/k3/k4 也按实际 substage time 切换，避免一个步内混用两套约束。

**第三层求解器：**`third_level_closed_form_solver_enabled=true`。当前每级行数很少且无 slack，通常由 `solve_weighted_min_norm_halfspaces` 精确求解。该实现枚举不超过 12 行的 active sets，并对每个候选 KKT 系统用线性代数求解；项目代码将其称为“closed-form active-set solver”。严格说它不是一条预先写死的符号解析公式，而是有限 active-set 枚举得到的精确小规模 QP 解。若行数超过 12 或该求解失败，才回退到 `quadprog`。

在 endpoint snap 区，P1 先由硬速度等式固定，P2–P4 仍由闭式 safety QP 处理，最后一个增量块 `u5` 再通过线性方程直接求解以保持 P5。

## 8. 平滑度 cost 的当前状态

代码中保留了障碍物附近的局部二阶差分代价：

\[
J_{\mathrm{smooth}}=
\frac{w}{2\ell^2}\sum_i
\lVert p_{i-1}-2p_i+p_{i+1}\rVert^2.
\]

它用于惩罚相邻连线向量的变化，从而减少明显折角和轨迹换侧。但**当前第二层和第三层开关都为 0，因此该项现在完全不参与 QP cost**。

保留但未生效的参数如下：

| 层级 | obstacle | weight | length scale | activation | prediction horizon | vicinity padding |
|---|---:|---:|---:|---:|---:|---:|
| 第二层 | 3 | 1 | 0.05 | 0 | 0.05 | 0.020 |
| 第三层 | 3 | 1000 | 0.02 | 0 | 0.05 | 0.010 |

论文或汇报中不应把该平滑项描述为当前方法的一部分，除非重新打开相应开关并重新生成结果。

## 9. Seed filter 与 Safety 的关系

当前三层 seed filter **只按方差条件筛选**：

- 保留：当前处于硬约束阶段的积分方差 HOCBF、末端方差 PTCBF，以及有限数值检查；
- 关闭：赛道点、障碍点、赛道连线和障碍连线的 seed 拒绝条件；
- 第一层以完整 5 点轨迹为重试单元，每条最多尝试 100 个 seed；第二、三层以 segment 为重试单元，每段最多尝试 20 个 seed，已经通过的兄弟段冻结保留；
- 第二层某段 20 次仍不通过时，更换其所属第一层轨迹的 seed，并重新生成该父轨迹及其全部 4 个第二层段；最多进行 10 轮这种上溯；
- 第三层某段 20 次仍不通过时，先重新生成其所属第二层段，再重新生成该段的 4 个第三层子段。如果所属第二层段本身 20 次仍不通过，或连续 10 个替代第二层段仍无法产生全部合格子段，则继续上溯，更换所属第一层轨迹的 seed，并重新生成该父轨迹下的 4 个第二层段与 16 个第三层段；第一层上溯最多 10 轮；
- 只有被上溯影响的父轨迹及后代会被替换，其他已经接受的轨迹和段保持不变；所有替换后的 seed、acceptance diagnostics 和 rollout cache 会同步更新；
- implementation version：第一/二/三层分别为 4/14/9，旧的第二、三层 rollout cache 因配置版本变化而失效。

因此 Safety 不再被 seed filter 预先筛成 100%。避障和边界约束仍在 rollout 的 QP 中运行；这里只是不根据最终几何结果更换 seed。

## 10. 比较指标及其当前定义

### 10.1 Safety（越大越好）

只检查最终 65 个第三层生成点：每个点必须位于**原始赛道边界**内并处于所有物理障碍物外。Safety 不减控制器 margin，也不检查相邻生成点之间的直线是否穿障碍。第一、第二层的原始几何 $h$ 容差保持 $10^{-8}$；第三层最终判定使用 $5\times10^{-4}$ 的物理法向距离容差。赛道隐式场本身为有符号距离；障碍超椭圆使用一阶法向距离 $h/\lVert\nabla h\rVert$。轨迹的全部生成点都安全时，该轨迹记为安全：

\[
\mathrm{Safety}=\frac{\#\{\text{所有生成点均安全的轨迹}\}}{N}.
\]

### 10.2 KL divergence（越小越好）

比较数据集终点分布 `P` 与生成终点分布 `Q`：

\[
D_{KL}(P\Vert Q)=\int P(z)\log\frac{P(z)}{Q(z)}\,dz.
\]

两者使用二维 Gaussian KDE。带宽只由数据集终点确定，采用 `std(data)*N^(-1/6)`；网格范围为数据终点范围向每侧扩展 6 个带宽，分辨率为 `128 x 128`。所有方法必须复用 `outputs/Racing_KL_Reference.mat` 中完全相同的带宽、网格和参考密度。

### 10.3 CS 与 AS（越小越好）

令相邻段向量为 `w_k=p_k-p_{k-1}`：

\[
\mathrm{CS}=\frac1{H-1}\sum_k
\left(1-\frac{w_k^Tw_{k+1}}{\lVert w_k\rVert\lVert w_{k+1}\rVert}\right),
\]

\[
\mathrm{AS}=\frac1{H-1}\sum_k
\lVert p_{k+1}-2p_k+p_{k-1}\rVert_2.
\]

指标在最终去除共享段端点重复后的 65 点轨迹上计算。

### 10.4 Time（越小越好）

当前 Time 为三层 rollout 总时间除以最终轨迹数：

\[
\mathrm{Time}=
\frac{T_{L1\ rollout}+T_{L2\ rollout}+T_{L3\ rollout}}{N}.
\]

它不包含超参数优化、GP 训练、cache I/O、绘图和指标计算。跨方法比较时必须使用相同硬件、worker 数、样本数和 cache 状态。

### 10.5 Variance（越小越好）

当前代码保存第三层每个 rollout 时间、每个 segment sample、每个 GP 输出的原始预测方差 `variance_values`。用于方法比较时建议固定同一个评价 GP，并报告：

\[
\mathrm{MeanVar}=\frac1{NTD}\sum_{n,t,d}\sigma_d^2(x_{n,t},t),
\qquad
\mathrm{TerminalVar}=\frac1{ND}\sum_{n,d}\sigma_d^2(x_{n,T},T).
\]

当前 `safe_flow_metrics` 同时保存 `mean_predictive_variance` 和 `terminal_predictive_variance`：前者对 `time x segment sample x GP output` 全部元素求平均，后者对最终 rollout 时间切片的 `segment sample x GP output` 求平均。推荐主表使用 `MeanVar`，并将 `TerminalVar` 作为补充。

## 11. 输出与缓存

总评估开关为 `cfg.safe_flow_evaluation.enabled=true`。输出文件：

- `outputs/Racing_SafeFlow_Metrics_Variance_U.mat`：Safety、KL、CS、AS、Time，raw variance、接受样本的 `u`、三层 seeds、完整 `cfg` 和 cache manifest；
- `outputs/Racing_KL_Reference.mat`：所有方法共享的 KL 参考网格/KDE；
- 三层 hyperparameter、model、rollout cache 和 accepted seed MAT/CSV 文件。

其中 `u` 的单位和含义是标准化轨迹空间中的 flow correction，只用于分析生成器修正强度，不能解释为车辆控制量。

## 12. 当前需要在实验报告中明确的边界

1. Safety 是生成点级指标，不检查连接线；这是当前有意采用的较宽松定义。
2. Seed filter 只筛方差，不筛 Safety；因此需要同时报告 Safety 和方差筛选的 acceptance/attempt 统计。
3. 几何 PTCBF 的理论奇点在 `t=1`，数值 rollout 停在 `0.996`；第一、二层方差 PTZF 则把 `0.996` 映射为其理论终点 1。
4. 第三层末端方差约束在 `t=0.85` 后释放，terminal safety projection 关闭。
5. 平滑度 cost 当前关闭，不能用它解释现有结果。
6. `u` 不是赛车的物理控制输入，因此当前不报告依赖真实加速度/转向角的 Cost、AR 或 SR-A。

## 13. 主要代码入口

- 总配置：`matlab/get_config.m`
- 主流程：`matlab/main_demo.m`
- 分层约束构造：`matlab/make_level_variance_constraint.m`
- 方差 HOCBF/QP 入口：`matlab/apply_hocbf_integral.m`
- 末端方差 PTCBF：`matlab/terminal_variance_ptcbf.m`
- 第三层顺序 QP：`matlab/solve_sequential_increment_qp.m`
- 障碍函数：`matlab/obstacle_level_and_gradient.m`
- 赛道边界：`matlab/track_boundary_cbf_info.m`
- 指标与导出：`matlab/evaluate_and_save_safeflow_metrics.m`
