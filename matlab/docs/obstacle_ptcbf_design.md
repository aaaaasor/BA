# 轨迹避障（障碍 PTCBF）最终设计方案

> 目标：把 SafeFlow（arXiv:2504.08661）的规定时间避障 CBF 加进本项目的
> LoG-GP 流匹配 rollout，并做「有 / 无 OOD 控制」A/B 对照，观察无 OOD 时
> 轨迹被障碍推离训练流形后发散的现象。
>
> 结论先行：
> 1. **整条最终曲线要真正避障，必须三层（L1/L2/L3）都加障碍 PTCBF**，因为
>    层级自由度自上而下冻结，每层只能管自己拥有的点。
> 2. **「OOD 控制」= 两条基于 GP 方差 β 的约束一起**：`integral` HOCBF +
>    末端方差 `terminal` PTCBF。「无 OOD」= 这两个都关（不是只关 HOCBF）。
> 3. 障碍 CBF（安全）和 PTCLF（端点/形状）、方差约束（OOD）互相独立。

---

## 1. 术语：OOD 控制的精确定义

作用在 `β = Σσ²ᵢ`（LoG-GP 总预测方差 = 离训练流形程度）上的约束有两条，
合起来才是「OOD 控制」：

| 约束 | 开关 | 作用 | 属于 OOD 控制 |
|---|---|---|---|
| `integral` HOCBF | `*_hocbf_enabled` | 累计方差积分 ≤ 预算 B | ✅（路径分量） |
| `terminal` 方差 PTCBF | `*_ptcbf_enabled` | 末端方差 ≤ β_final | ✅（末端分量） |
| `anchor` PTCLF | `*_ptclf_enabled` | 端点钉到上层锚点 | ❌（形状/端点） |
| **障碍 CBF（新增）** | `*_obstacle_enabled` | 避障 | ❌（安全） |

⚠️ 命名坑：已有的 `ptcbf_enabled` 是**末端方差 PTCBF**（OOD），与新加的
**障碍 PTCBF**（`obstacle_enabled`，避障）是两码事。

---

## 2. 论文机制（避障那部分）

SafeFlow 避障 = 规定时间 CBF（FMBF；论文式 12–16 单障碍、29–31 复合）：

```
每航点 s_k、障碍 j：  h_j(s_k) = (s_k − c_j)ᵀ Q_j (s_k − c_j) − 1 ≥ 0     (式 38)
QP：  min ‖u‖²   s.t.   b_{j,k}ᵀ u_k ≥ − a_{j,k}
      b_{j,k} = ∂h_j/∂s_k
      a_{j,k} = b_{j,k}ᵀ E_k v_θ + φ_j(t, h_j)·h_j
φ 规定时间：  h ≥ 0 → φ0（普通 CBF）
             h < 0 → φ1(t) blow-up，t → 1⁻ 时 → ∞                     (式 8/42)
终端安全滤波：  T₁ = argmin ‖T − T_{1−}‖  s.t.  h_j(E_k T) ≥ 0          (式 17/32)
```

φ1 的 blow-up 把「初始就在障碍内的点」在 t=1 前逼出安全集；终端滤波补偿
数值误差、给出精确安全。本项目里 `v_θ` 用 LoG-GP 的均值速度 μ。

---

## 3. 为什么整条曲线避障必须三层都加

层级自由度自上而下冻结：每层端点被 PTCLF 钉住、并在最后被覆盖成上一层的点。
所以每层只**拥有**一部分点、且无法修复上层传下来的点：

```
最终曲线的点  =  L1 的 5 个骨架点            ← 只有 L1 能动
              ∪  L2 的 12 个内部点(锚点之间)  ← 只有 L2 能动
              ∪  L3 每段的 3 个内部点         ← 只有 L3 能动
```

- **只在 L3**：L3 每段端点被覆盖成 L2 的 17 点；若 L2 穿过椭圆，端点被覆盖回
  椭圆内，L3 救不了。
- **只在 L1**：5 个骨架点绕开了，但相邻锚点若分居椭圆两侧，L2/L3 在它们之间
  细化出的弦照样切进椭圆——那些中间点不归 L1 管。

**安全必须在最终轨迹的分辨率上满足，而这些点分散在三层里，故每层都要在
「自己拥有的点」上强制避障**（多分辨率下的 composite 安全）。

作用点分配：

| 层 | 拥有的点 | 障碍 CBF 作用点 |
|---|---|---|
| L1 | 全部 5 骨架点 | `[1 2 3 4 5]` + 终端安全滤波 |
| L2 | 每段内部 3 点 | `[2 3 4]`（端点由 L1 继承已安全） |
| L3 | 每段内部 3 点 | `[2 3 4]`（端点由 L2 继承已安全） |

---

## 4. 障碍几何（放大到明显）

物理坐标 ~ 单位方块内，训练轨迹是对角 S 形 `(0.1,0.1)→(0.5,0.5)→(0.9,0.9)`，
中点切向接近竖直。取**轴对齐椭圆**压在中点，明显尺寸：

```
c = [0.5; 0.5]           中心（正压均值路径中点）
a = 0.20  (x 半轴)        Q = diag(1/a², 1/b²)
b = 0.14  (y 半轴)
```

比初版的 `a=b=0.12` 明显更大，未加控制的曲线会清楚地穿过，避障绕行一目了然。
（多障碍时 `centers/semi_axes` 各加一列即可，QP 每点每障碍加一行。）

### 三层用同一个真实椭圆，不做层间充气

论文在每个航点直接强制 `h(s_k) ≥ 0`，没有层间充气 margin。本方案里最终
65/80 点的曲线中，**每个点都被某一层的障碍 CBF 直接约束**（L1 管 5 骨架点、
L2 管每段内部 3 点、L3 管每段内部 3 点），不存在"没被约束、只能靠上层充气
兜"的点，故充气多余。**三层一律用真实半轴 `(a, b)`。**（离散航点之间连续
曲线的微小内凹属离散化误差，与论文同源，可靠加密时间步/点数缓解，不靠充气。）

---

## 5. 三层通用的障碍 PTCBF 数学

### 5.1 点 → 物理位置的线性映射（三层唯一的差别）

设某层状态为 `x`（L1/L2 是绝对归一化坐标，L3 是局部增量 z）。第 m 点物理
位置 `p_m ∈ R²` 都能写成 `x` 的**线性函数** `p_m = M_m x + o_m`（2×N 矩阵）：

- **L1 / L2（绝对，选择子型）**：位置在下标 `idx_m = (m−1)*4 + [1,2]`
  ```
  M_m = 选择 idx_m 两列，各乘 std(idx_m)
  o_m = mean(idx_m)
  ```
  （和 `anchor_clf_info.m` 的 indices 形式一致。data_transform：L1 用
  `data_transform`，L2 用 `segment_data_transform`。）

- **L3（局部增量，累加矩阵型）**：`v = std⊙z + mean`，
  `v = [S0; S1−S0; …; S4−S3]`（只有位置取增量）。第 m 点位置 =
  前 m 个 block 位置增量累加：
  ```
  M_m：对每个 block i ≤ m，把列 (i−1)*4+[1,2] 置成 diag(std 对应两项)
  o_m：Σ_{i≤m} mean 对应两项
  ```
  这是 `build_increment_endpoint_clf_targets.m` 里 `A`（首/末点）的推广到
  全部 5 点。data_transform 用 `third_segment_data_transform`。

> 三层用同一个 `obstacle_cbf_info.m`，只是传入的 `{(M_m,o_m)}` 不同——
> 完全复用 `anchor_clf_info.m` 里 indices / matrix 的双形式思路。

### 5.2 单点单障碍的 QP 行

```
h_m    = (p_m − c)ᵀ Q (p_m − c) − 1                 Q = diag(1/a², 1/b²)  (三层同一真实椭圆)
grad_m = ∂h_m/∂x = 2 (Q (p_m − c))ᵀ M_m             (1×N)

规定时间条件:  ḣ_m + φ(t, h_m)·h_m ≥ 0,   ḣ_m = grad_m·(μ + u)

⇒ 项目的  A·u ≤ bound  形式:
     A_row = − grad_m
     bound =   grad_m·μ  +  φ(t, h_m)·h_m
```

μ 是该层 GP 速度（L1/L2 绝对、L3 增量），和其它约束同坐标系，量纲自洽。

> 注意：L3 增量表示把各点耦合（`p_5` 依赖全部 block，`p_2` 只依赖前两个），
> 不同点的障碍行**耦合**，不能像论文式 16 逐点闭式解——统一丢进现有 QP 由
> `quadprog` 联合求解（更精确）。

### 5.3 φ blow-up（复用 ptzf_time_shift）

```
t_eff = t + constraint_cfg.ptzf_time_shift      (rk4_rollout 已把 t_max<1 → t_eff=1)
τ     = max(1 − t_eff, eps)
h ≥ 0:  φ = φ0            (常数, 建议 2.0)
h < 0:  φ = ω / τ²        (逆多项式, ω>2 保证 u 有界 [Prop.2]; 建议 ω=4)
```

---

## 6. 端点硬覆盖 + 终端安全滤波

### 6.1 各层端点的实际处理（对齐当前代码 + 本方案的新增）

- **L2：已有硬覆盖**（`main_demo.m:349`）：rollout 后
  `final_segment_data(:, anchor_clf_indices) = anchor_clf_targets_physical`，
  端点被精确替换成 L1 生成点。
- **L3：当前代码没有覆盖**（`main_demo.m:593-605` 只读端点算
  `third_level_anchor_error` 诊断残差）。**本方案新增 L3 硬覆盖**，与 L2 对齐：
  decode 出 `final_third_segment_data`（每段 5 点物理坐标）后，把每段的
  **点 1、点 5** 覆盖成 `third_anchor_clf_targets`（即已避障的 L2 点）。

L3 覆盖伪代码（增量表示，覆盖发生在 decode 之后的全局物理坐标上）：
```
for 每个 L3 segment 行 r:
    final_third_segment_data(r, 1:F)            = third_anchor_clf_targets(r, 1:F);      % 点1
    final_third_segment_data(r, 4F+(1:F))       = third_anchor_clf_targets(r, F+(1:F));  % 点5
    % F = segment_feature_dim(=4); 内部点 2,3,4 保持 rollout 结果不动
```

### 6.2 硬覆盖与避障的职责划分（重要）

硬覆盖只碰**端点**，不碰内部点，故与"内部点避障"正交：

- **障碍 CBF 只作用在内部点 `[2 3 4]`**，端点(1,5)不加 CBF（反正被覆盖，
  加了既没用又和 PTCLF 打架）。
- **端点安全 = 覆盖目标安全**：L3 端点覆盖成 L2 点 → 只要 **L2 已避障**
  （L2 开障碍 CBF）+ **L1 终端安全滤波**，覆盖进来的端点天然在障碍外。
- 因此 **L3 不需要再做终端安全滤波**；下层端点安全逐层继承。

### 6.3 终端安全滤波（只需 L1）

L1 rollout 后、把 5 点当作下层锚点之前，把落在椭圆内的点投影出椭圆
（论文式 17）。L1 安全 → L2 端点继承安全 → L3 端点继承安全。

**投影实现**（新 `terminal_safety_filter.m`）：对每点 p，若
`d = (p−c)ᵀQ(p−c) < 1`，沿中心射线推到边界外：
```
p ← c + (p − c) / sqrt(d) · (1 + ε)          (ε 小正数, 保证严格 > 0)
```
（射线投影，简单且安全；如需最近点投影可对轴对齐椭圆做几步牛顿，非必需。）

---

## 7. 文件改动清单

| 文件 | 改动 |
|---|---|
| **新** `build_obstacle_point_maps.m` | 由 data_transform + 模式(absolute/increment) 返回各点 `{(M_m,o_m)}` |
| **新** `obstacle_cbf_info.m` | 仿 `anchor_clf_info.m`：输入 stats/cfg/t，输出多行 `A(k×N)`、`bound(k×1)` 及诊断（每点 h、h_min、active 数） |
| **新** `terminal_safety_filter.m` | 把物理点投影出椭圆（L1 用） |
| **新** `run_obstacle_ood_experiment.m` | A/B 驱动 + 叠加图 |
| `active_qp_constraints.m` | 追加 obstacle 行，`constraint_types` 加 `"obstacle"` |
| `apply_hocbf_integral.m` | 调 `obstacle_cbf_info`、并入 QP、加 obstacle 诊断字段 |
| `solve_slack_qp.m`（及 `slack_enabled_for_constraints`/`slack_weights_for_constraints`） | 注册 `"obstacle"`：默认硬；仅可行性兜底开大权重 slack（对应论文 δ 松弛），不随 `slack_switch_time` 切换 |
| `make_level_variance_constraint.m` | 读入各层 `*_obstacle_*` 字段 + 真实椭圆几何；`obstacle_enabled` 需 `cfg.obstacle.enabled && 该层 *_obstacle_enabled` |
| `get_config.m` | 新增 §8 配置块 |
| `main_demo.m` | 三层各自建 point maps 挂到约束；L1 rollout 后 `terminal_safety_filter`；**新增 L3 端点硬覆盖**（decode 后把每段点1/点5 覆盖成 `third_anchor_clf_targets`，照抄 L2 第 349 行）；画椭圆；A/B 用独立 rollout 缓存路径 |

**缓存**：rollout 缓存按 `isequaln(saved_*_constraint, …)` 判定，新增字段会自动
让旧缓存失效；A/B 两变体各用独立 `*_rollout_path` 避免互相覆盖。

---

## 8. 新增配置（get_config.m）

```matlab
%% Obstacle（物理坐标；列 = 障碍）
% ★ 避障总开关：关掉 = 完全退回原三层生成（无障碍 CBF / 无终端滤波 /
%   无 L3 端点硬覆盖），行为与现在的 main_demo 完全一致。
cfg.obstacle.enabled   = true;

cfg.obstacle.centers   = [0.5; 0.5];      % c   (2 x n_obs)
cfg.obstacle.semi_axes = [0.20; 0.14];    % [a; b]  (2 x n_obs) —— 明显尺寸，三层同用
% blow-up
cfg.obstacle.phi0       = 2.0;
cfg.obstacle.phi1_omega = 4.0;            % >2
% slack（仅可行性兜底，近似硬；独立于 slack_switch_time）
cfg.obstacle.slack_enabled = true;
cfg.obstacle.slack_weight  = 1e4;

% 每层开关 + 作用点（均受 cfg.obstacle.enabled 总开关门控）
cfg.variance_constraint.first_level_obstacle_enabled  = true;
cfg.variance_constraint.first_level_obstacle_points   = [1 2 3 4 5];
cfg.variance_constraint.second_level_obstacle_enabled = true;
cfg.variance_constraint.second_level_obstacle_points  = [2 3 4];
cfg.variance_constraint.third_level_obstacle_enabled  = true;
cfg.variance_constraint.third_level_obstacle_points   = [2 3 4];
% 终端安全滤波（只需 L1）
cfg.variance_constraint.first_level_terminal_safety_filter_enabled = true;
```

**总开关语义**（`cfg.obstacle.enabled`）：

- `true`：三层障碍 CBF + L1 终端安全滤波 + L3 端点硬覆盖 全部生效。
- `false`：以上全部旁路，退回**原始三层生成**（L2 仍有其原有的第 349 行覆盖，
  但**不新增 L3 覆盖**、不加任何障碍相关逻辑）——即当前 `main_demo` 的行为。
  实现上：`make_level_variance_constraint` 里 `enabled && 总开关` 才置真；
  `main_demo` 里 L1 终端滤波、L3 端点覆盖两段都包在 `if cfg.obstacle.enabled`。

---

## 9. A/B 实验协议

三层始终开避障，保证整条曲线避障；**L1/L2 及其 OOD 控制两变体完全一致**
（保证骨架/锚点相同、变量干净），**只切 L3 的两条方差约束**：

```
变体 A（有 OOD）: L3 obstacle=on,  L3 hocbf=on,  L3 ptcbf=on
变体 B（无 OOD）: L3 obstacle=on,  L3 hocbf=off, L3 ptcbf=off
```

`run_obstacle_ood_experiment.m` 跑两次（各自独立 L3 rollout 缓存），叠加出图。

---

## 10. 诊断与画图

- **叠加图**：同一张 X-Y 图画椭圆（`a,b`）+ 变体 A 轨迹 + 变体 B 轨迹。
- 复用 `plot_sigma_vs_time`（σ vs t，看 B 是否冲高）。
- 新增 **h_min vs t**（每步所有作用点的最小障碍值，负 = 侵入）。
- 沿用已算的 `third_level_diverged_sample_count`（发散样本计数）。

---

## 11. 预期结果

- **变体 A（有 OOD）**：内部点被推出椭圆、方差 CBF 把增量压在流形上 →
  段在锚点间「鼓」开、有界收敛；h_min 抬到 ≥0；发散计数 ≈ 0。
- **变体 B（无 OOD）**：障碍 u（尤其 φ1 末端 blow-up）把 z 推进高方差增量区、
  μ 失真 → 段内点增量爆炸、σ 单调冲高、发散计数显著 > 0。
  **这就是「不加 OOD 会发散」的直接证据。**

---

## 12. 实现顺序（分步、可逐步验证）

1. §4 几何 + §8 配置 + `build_obstacle_point_maps.m`（含 absolute / increment 两模式）。
2. `obstacle_cbf_info.m`（先跑通单点单障碍的 h/grad/bound + 诊断）。
3. 接线：`active_qp_constraints.m` → `apply_hocbf_integral.m` →
   `solve_slack_qp.m` 注册 `"obstacle"` → `make_level_variance_constraint.m`。
4. **先只在 L1** 开避障 + 终端安全滤波，确认骨架 5 点清楚绕开椭圆。
5. 再开 L2、L3（三层同一真实椭圆，无充气），确认整条 65/80 点曲线整体在椭圆外。
   顺带验证 `cfg.obstacle.enabled=false` 时行为与原 `main_demo` 完全一致。
6. `run_obstacle_ood_experiment.m` + 叠加图 / h_min 图，跑出 A/B 对照。
```
