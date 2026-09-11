# SafeFlow 论文公式整理

> Dai, Yang, Yu, Liu, Sadeghian, Haddadin, Hirche, *SafeFlow: Safe Robot Motion Planning with Flow Matching via Control Barrier Functions*, arXiv:2504.08661v3.
> 编号与原文一致（式 (1)–(47)）。

## 0. 符号约定

| 符号 | 含义 |
|---|---|
| $\bm{s}\in\mathbb{S}\subseteq\mathbb{R}^{d_s}$ | 机器人状态，维数 $d_s$ |
| $\bm{\mathcal{T}}\in\mathbb{S}^{H+1}\subseteq\mathbb{R}^{d}$ | 轨迹，$\bm{\mathcal{T}}=[(\bm{s}^0)^\top,\dots,(\bm{s}^H)^\top]^\top$，$d=(H+1)d_s$ |
| $H$ | 规划步长（horizon） |
| $t\in[0,1]$ | **生成时间**（flow matching 的积分变量，不是物理时间） |
| $k=0,\dots,H$ | 轨迹内的时间步索引 |
| $\bm{v}(\cdot,\cdot):[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$ | 真实向量场；$\bm{v}^{\bm\theta}$ 为神经网络近似（参数 $\bm\theta\in\mathbb{R}^{d_\theta}$） |
| $\bm{\psi}(\cdot,\cdot)$ | 由 $\bm v$ 诱导的流映射，$\bm\psi(0,\bm{\mathcal{T}})=\bm{\mathcal{T}}$ |
| $\bm{\mathcal{T}}_t=\bm\psi(t,\bm{\mathcal{T}}_0)$ | $t$ 时刻中间轨迹，$=[(\bm s_t^0)^\top,\dots,(\bm s_t^H)^\top]^\top$ |
| $p_0,\;q,\;p_t$ | 先验分布 / 目标数据分布 / 中间分布 |
| $\mathbb{D}=\{\bm{\mathcal{T}}_0^{(\iota)},\bm{\mathcal{T}}^{(\iota)}\}_{\iota\in\mathbb{N}}$ | 专家演示数据集 |

---

## 一、预备知识（Section II）

### 1.1 Flow Matching 模型

**(1) 生成 ODE**

$$\frac{\mathrm{d}\bm{\psi}(t,\bm{\mathcal{T}}_0)}{\mathrm{d}t}=\bm{v}\big(t,\bm{\psi}(t,\bm{\mathcal{T}}_0)\big),\qquad t\in[0,1]$$

其中 $\bm{\mathcal{T}}_0\sim p_0$，最终生成轨迹 $\bm{\mathcal{T}}=\bm{\mathcal{T}}_1$。
训练目标：学习 $\bm{v}_t^{\bm\theta}=\bm{v}^{\bm\theta}(t,\cdot)$，使其诱导的流把 $p_0$ 输运到 $q$。

### 1.2 控制障碍函数（CBF）

**(2) 控制仿射系统**

$$\dot{\bm{x}}(t)=\bm{f}(\bm{x}(t))+\bm{g}(\bm{x}(t))\bm{u}(t)$$

$\bm f:\mathbb{X}\to\mathbb{R}^{d_x}$，$\bm g:\mathbb{X}\to\mathbb{R}^{d_x\times d_u}$，已知且局部 Lipschitz。
安全集：$\mathbb{C}:=\{\bm{x}\in\mathbb{X}\mid h(\bm{x})\ge 0\}$。

**(3) CBF 条件（Definition 1）**

$$\sup_{\bm{u}\in\mathbb{R}^{m}}\Big\{L_{\bm f}h(\bm x_t)+L_{\bm g}h(\bm x_t)\bm u+\alpha\big(h(\bm x_t)\big)\Big\}\ge 0,\qquad \forall t\ge 0$$

$\alpha(\cdot)$ 为 class-$\mathcal{K}$ 函数。
**Lemma 1**：若 $\bm x(0)\in\mathbb{C}$ 且 $h$ 是 CBF，则 $\mathbb{C}$ 前向不变（系统安全）。

> 关键缺陷：FM/扩散的初值 $\bm{\mathcal{T}}_0$ 采自无界高斯，$\bm x(0)\in\mathbb{C}$ **不成立** —— 这正是 SafeFlow 要解决的核心问题（[22][23] 忽略了这一点）。

### 1.3 轨迹安全的定义

**(4) 状态安全集**

$$\mathbb{C}_s=\{\bm{s}\in\mathbb{S}\mid h(\bm{s})\ge 0\}$$

**Assumption 1**：$\mathbb{C}_s$ 非空且无孤立点。

**(5) 安全轨迹（Definition 2）**

$$h(\bm{E}_k\bm{\mathcal{T}})\ge 0,\qquad \forall k=0,\dots,H$$

选择矩阵 $\bm{E}_k\in\mathbb{R}^{d_s\times d}$：

$$\bm{E}_k=[\bm{0}_{d_s\times kd_s},\;\bm{I}_{d_s},\;\bm{0}_{d_s\times(H-k)d_s}]\quad\Longrightarrow\quad \bm{s}^k=\bm{E}_k\bm{\mathcal{T}}$$

**Problem 1**：设计 FM 框架，使任意 $\bm{\mathcal{T}}_0\sim p_0$ 生成的 $\bm{\mathcal{T}}=\bm{\mathcal{T}}_1$ 都安全。

---

## 二、单一约束下的 SafeFlow（Section III-A）

### 2.1 引导流

**(6) 带引导项的 FM**

$$\frac{\mathrm{d}\bm{\psi}_t(\bm{\mathcal{T}}_0)}{\mathrm{d}t}=\bm{v}_t^{\bm\theta}\big(\bm{\psi}_t(\bm{\mathcal{T}}_0)\big)+\bm{u}_t,\qquad t\in[0,1]$$

引导项 $\bm{u}_t=[(\bm u_t^0)^\top,\dots,(\bm u_t^H)^\top]^\top\in\mathbb{R}^d$，$\bm u_t^k\in\mathbb{R}^{d_s}$。

**Definition 3（Prescribed-time safety）**：对任意初值 $\bm x(0)\in\mathbb{X}$，若存在 $T_d>0$ 使 $\bm x_t\in\mathbb{C},\;\forall t\ge T_d$，则系统为规定时间安全。

### 2.2 流匹配障碍函数 FMBF

**(7) FMBF 条件（Definition 4）**

$$\sup_{\bm u_t^k\in\mathbb{R}^{d_s}}\left\{\frac{\mathrm{d}h(\bm E_k\bm{\mathcal{T}}_t)}{\mathrm{d}\bm E_k\bm{\mathcal{T}}_t}\bm E_k\bm v_t^{\bm\theta}(\bm{\mathcal{T}}_0)+\frac{\mathrm{d}h(\bm E_k\bm{\mathcal{T}}_t)}{\mathrm{d}\bm E_k\bm{\mathcal{T}}_t}\bm u_t^k+\varphi\big(t,h(\bm E_k\bm{\mathcal{T}}_t)\big)h(\bm E_k\bm{\mathcal{T}}_t)\right\}\ge 0$$

对所有 $k=0,\dots,H$ 与 $t\in[0,1)$ 成立。

**(8) 增益函数 $\varphi$**（取代 CBF (3) 中的 class-$\mathcal{K}$ 函数 $\alpha$）

$$\varphi(t,h)=\begin{cases}\varphi_0, & \text{if } h\ge 0\\[2pt] \varphi_1(t), & \text{otherwise}\end{cases},\qquad t\in[0,1)$$

其中 $\varphi_0\in\mathbb{R}_{>0}$，$\varphi_1:[0,1)\to\mathbb{R}_{>0}$ 单调递增且

$$\int_0^{1^-}\varphi_1(s)\,\mathrm{d}s=\infty,\qquad \lim_{t\to 1^-}\varphi_1(t)=\infty\quad\text{（blow-up 函数）}$$

- 安全时（$h\ge0$）：$\varphi(t,h)h=\varphi_0 h$ 属于 class-$\mathcal{K}$，退化为普通 CBF；
- 不安全时（$h<0$）：$\varphi_1$ 在 $t\to1^-$ 时爆炸，**强制**在 $t=1$ 前进入安全集。

**(9) 指数型 blow-up**

$$\varphi_1(t)=-\dot{\xi}(t)/\xi(t),\qquad \xi(t)=\exp\big(\omega(1-t)\big)-1$$

**(10) 反多项式型 blow-up**

$$\varphi_1(t)=\omega/(1-t)^2,\qquad \omega\in\mathbb{R}_{>0}$$

### 2.3 安全性定理

**(11) 可行引导集（Theorem 1）**

$$\mathbb{U}_t=\prod_{k=0}^{H}\mathbb{U}_t^k$$

$$\mathbb{U}_t^k=\left\{\bm u_t^k\in\mathbb{R}^{d_s}\ \middle|\ \frac{\mathrm{d}h(\bm E_k\bm{\mathcal{T}}_t)}{\mathrm{d}\bm E_k\bm{\mathcal{T}}_t}\bm E_k\bm v_t^{\bm\theta}(\bm{\mathcal{T}}_0)+\frac{\mathrm{d}h(\bm E_k\bm{\mathcal{T}}_t)}{\mathrm{d}\bm E_k\bm{\mathcal{T}}_t}\bm u_t^k+\varphi\big(t,h(\bm E_k\bm{\mathcal{T}}_t)\big)h(\bm E_k\bm{\mathcal{T}}_t)\ge 0\right\}$$

**Theorem 1**：若 $h$ 是有效 FMBF 且 $\bm u_t\in\mathbb{U}_t$，则 (6) 为 safe flow matching，所有 $\bm{\mathcal{T}}_1$ 满足 Definition 2。

### 2.4 QP 求解引导项

**(12) 全局 QP**

$$\bm u_t=\arg\min_{\bm u=[(\bm u^0)^\top,\dots,(\bm u^H)^\top]^\top\in\mathbb{R}^d}\ \|\bm u\|^2$$

$$\text{s.t.}\quad a_t^k+(\bm b_t^k)^\top\bm u^k\ge 0,\qquad \forall k\in\{0,\dots,H\}$$

**(13)(14) QP 系数**

$$\bm b_t^k=\left(\frac{\partial h(\bm E_k\bm{\mathcal{T}}_t)}{\partial(\bm E_k\bm{\mathcal{T}}_t)}\right)^{\!\top}\in\mathbb{R}^{d_s}$$

$$a_t^k=(\bm b_t^k)^\top\bm E_k\bm v_t^{\bm\theta}(\bm{\mathcal{T}}_0)+\varphi\big(t,h(\bm E_k\bm{\mathcal{T}}_t)\big)h(\bm E_k\bm{\mathcal{T}}_t)\in\mathbb{R}$$

**(15) 解耦**（因 $\|\bm u\|^2=\sum_{k=0}^H\|\bm u^k\|^2$，可逐 $k$ 独立求解）

$$\bm u_t^k=\arg\min_{\bm u^k\in\mathbb{R}^{d_s}}\|\bm u^k\|^2\quad \text{s.t.}\quad a_t^k+(\bm b_t^k)^\top\bm u^k\ge 0$$

**(16) 闭式解（KKT 条件）**

$$\bm u_t^k=\begin{cases}\bm 0_{d_s\times 1}, & \text{if } a_t^k\ge 0\\[6pt] -\dfrac{\bm b_t^k a_t^k}{\|\bm b_t^k\|^2}, & \text{otherwise}\end{cases}$$

**Proposition 1**：若 $\mathrm{d}h(\bm s)/\mathrm{d}\bm s\ne\bm 0_{d_s\times1}$ 且 Assumption 1 成立，则 $h$ 为有效 FMBF，QP 恒可行且有上述闭式解。

### 2.5 终端安全滤波器

**(17)(18)** 补偿 $t\to1^-$ 的数值误差（可能 $\bm{\mathcal{T}}_{1^-}\notin\mathbb{C}_s^{H+1}$）：

$$\bm{\mathcal{T}}_1=\arg\min_{\bm{\mathcal{T}}\in\mathbb{R}^d}\ \|\bm{\mathcal{T}}-\bm{\mathcal{T}}_{1^-}\|\qquad \text{s.t.}\quad h(\bm E_k\bm{\mathcal{T}})\ge 0,\ \forall k=0,\dots,H$$

可行性由 $\mathbb{C}_s\ne\emptyset$ 保证（至少存在 $\bm{\mathcal{T}}_1=[\bm s_{safe}^\top,\dots,\bm s_{safe}^\top]^\top$）。

### 2.6 引导项有界性（Proposition 2，式 19–24）

**(19) 指数型导数**：$\dot{\varphi}_1(t)=\exp\big(-\omega(1-t)\big)\varphi_1^2(t)$

**(20) 反多项式型导数**：$\dot{\varphi}_1(t)=2\omega^{-1}\varphi_1^2(t)$

统一写为 $\dot{\varphi}_1(t)\le c_\varphi(t)\varphi_1^2(t)$，其中

$$c_\varphi(t)=\max\{\exp(-\omega(1-t)),\,2\omega^{-1}\}\in\mathbb{R}_+,\qquad \omega>2\ \Longrightarrow\ c_\varphi(t)<1$$

**(21) Lyapunov 候选函数**

$$V_t^k=\varphi_1^2(t)h^2(\bm E_k\bm{\mathcal{T}}_t)/2$$

**(22) 导数估计**（在 $h(\bm E_k\bm{\mathcal{T}}_t)<0$ 时）

$$\dot V_t^k=\varphi_1(t)\dot\varphi_1(t)h^2-\varphi_1^2(t)|h|\dot h\ \le\ c_\varphi(t)\varphi_1^3(t)h^2-\varphi_1^3(t)h^2=-2\big(1-c_\varphi(t)\big)\varphi_1(t)V_t^k$$

**(23) 比较原理**

$$V_t^k\le \exp\left(-2\int_0^t\big(1-c_\varphi(s)\big)\varphi_1(s)\,\mathrm{d}s\right)V_0^k\ \le\ V_0^k$$

**(24) 有界性结论**

$$\varphi_1(t)\,|h(\bm E_k\bm{\mathcal{T}}_t)|\ \le\ \varphi_1(0)\,|h(\bm E_k\bm{\mathcal{T}}_0)|$$

故 $a_t^k$ 有界 $\Rightarrow$ $\bm u_t$ 在 $t\in[0,1)$ 上有界（前提：$\|\bm v_t^{\bm\theta}(\bm{\mathcal{T}}_0)\|$ 有界）。

---

## 三、复合约束下的 SafeFlow（Section III-B）

**(25) 复合安全集**（$N$ 个障碍函数 $h_j:\mathbb{S}\to\mathbb{R}$，$j=1,\dots,N$）

$$\mathbb{C}_s=\bigcap_{j=1}^{N}\mathbb{C}_{s,j},\qquad \mathbb{C}_{s,j}=\{\bm s\in\mathbb{S}: h_j(\bm s)\ge0\}$$

**(26) CFMBF 条件（Definition 6）**，$\bm h(\cdot)=[h_j(\cdot)]_{j=1,\dots,N}$：

$$\frac{\mathrm{d}h_j(\bm E_k\bm{\mathcal{T}}_t)}{\mathrm{d}\bm E_k\bm{\mathcal{T}}_t}\bm E_k\bm v_t^{\bm\theta}(\bm{\mathcal{T}}_0)+\frac{\mathrm{d}h_j(\bm E_k\bm{\mathcal{T}}_t)}{\mathrm{d}\bm E_k\bm{\mathcal{T}}_t}\bm u_t^k+\varphi_j\big(t,h_j(\bm E_k\bm{\mathcal{T}}_t)\big)h_j(\bm E_k\bm{\mathcal{T}}_t)\ge 0$$

**(27) 逐约束增益**

$$\varphi_j(t,h)=\begin{cases}\varphi_{0,j}, & \text{if } h\ge0\\[2pt] \varphi_{1,j}(t), & \text{otherwise}\end{cases},\qquad \int_0^{1^-}\varphi_{1,j}(s)\,\mathrm{d}s=\infty$$

**(28) 可行集（Theorem 2）**

$$\mathbb{U}_t=\prod_{k=0}^{H}\mathbb{U}_t^k,\qquad \mathbb{U}_t^k=\bigcap_{j=1}^{N}\mathbb{U}_{j,t}^k$$

$$\mathbb{U}_{j,t}^k=\left\{\bm u_t^k\in\mathbb{R}^{d_s}\ \middle|\ \frac{\mathrm{d}h_j}{\mathrm{d}\bm E_k\bm{\mathcal{T}}_t}\bm E_k\bm v_t^{\bm\theta}(\bm{\mathcal{T}}_0)+\frac{\mathrm{d}h_j}{\mathrm{d}\bm E_k\bm{\mathcal{T}}_t}\bm u_t^k+\varphi_j(t,h_j)h_j\ge 0\right\}$$

**(29) 复合 QP**

$$\bm u_t=\arg\min_{\bm u\in\mathbb{R}^d}\|\bm u\|^2\quad \text{s.t.}\quad a_t^{j,k}+(\bm b_t^{j,k})^\top\bm u^k\ge0,\ \forall k=0,\dots,H,\ \forall j=1,\dots,N$$

**(30)(31) 系数**

$$\bm b_t^{j,k}=\left(\frac{\partial h_j(\bm E_k\bm{\mathcal{T}}_t)}{\partial(\bm E_k\bm{\mathcal{T}}_t)}\right)^{\!\top}$$

$$a_t^{j,k}=(\bm b_t^{j,k})^\top\bm E_k\bm v_t^{\bm\theta}(\bm{\mathcal{T}}_0)+\varphi_j\big(t,h_j(\bm E_k\bm{\mathcal{T}}_t)\big)h_j(\bm E_k\bm{\mathcal{T}}_t)$$

**解耦子问题**

$$\bm u_t^k=\arg\min_{\bm u^k\in\mathbb{R}^{d_s}}\|\bm u^k\|^2\quad \text{s.t.}\quad a_t^{j,k}+(\bm b_t^{j,k})^\top\bm u^k\ge0,\ \forall j\in\{1,\dots,N\}$$

**带松弛的可行 QP**（多约束下可行性不再自动保证，引入 $\delta_j^k\in\mathbb{R}_{\ge0}$）

$$\bm u_t^k=\arg\min_{\bm u^k\in\mathbb{R}^{d_s},\,\delta_j^k\in\mathbb{R}_{\ge0}}\ \|\bm u^k\|^2+\sum_{j=1}^{N}(\delta_j^k)^2$$

$$\text{s.t.}\quad a_t^{j,k}+(\bm b_t^{j,k})^\top\bm u^k+\delta_j^k\ge0,\ \forall j\in\{1,\dots,N\}$$

一个显式可行点：$\bm u^k=\bm 0_{d_s\times1}$，

$$\delta_j^k=\max\left\{0,\;-\frac{\partial h_j(\bm E_k\bm{\mathcal{T}}_t)}{\partial \bm E_k\bm{\mathcal{T}}_t}\bm E_k\bm v_t^{\bm\theta}(\bm{\mathcal{T}}_0)-\varphi_j\big(t,h_j(\bm E_k\bm{\mathcal{T}}_t)\big)h_j(\bm E_k\bm{\mathcal{T}}_t)\right\}$$

**(32) 复合终端安全滤波器**（Remark 1：松弛可能破坏 $h_j(\bm E_k\bm{\mathcal{T}}_{1^-})\ge0$，因此必须启用）

$$\bm{\mathcal{T}}_1=\arg\min_{\bm{\mathcal{T}}\in\mathbb{R}^d}\|\bm{\mathcal{T}}-\bm{\mathcal{T}}_{1^-}\|\quad \text{s.t.}\quad h_j(\bm E_k\bm{\mathcal{T}})\ge0,\ \forall k=0,\dots,H,\ \forall j=1,\dots,N$$

---

## 四、评价指标（Section IV-A）

**(33) KL 散度**（对末状态 $\bm s^H=\bm E_H\bm{\mathcal{T}}$ 做高斯核密度估计，得数据集分布 $P$ 与生成分布 $Q$）

$$D_{\mathrm{KL}}(P\|Q)=\frac{1}{N}\sum_{i=1}^{N}P\big((\bm s^H)_i\big)\log\frac{Q\big((\bm s^H)_i\big)}{P\big((\bm s^H)_i\big)}$$

**(34) 转角**：段向量 $\bm w^k=\bm s^k-\bm s^{k-1}$，$k=1,\dots,H$

$$\theta^k=\cos^{-1}\left(\frac{(\bm w^k)^\top\bm w^{k+1}}{\|\bm w^k\|\,\|\bm w^{k+1}\|}\right),\qquad k=1,\dots,H-1$$

**(35) 曲率平滑度 CS**

$$S_c=(H-1)^{-1}\sum_{k=1}^{H-1}\big(1-\cos\theta^k\big)$$

**(36) 二阶差分（加速度）**

$$\bm a^k=\bm w^{k+1}-\bm w^k=\bm s^{k+1}-2\bm s^k+\bm s^{k-1},\qquad k=1,\dots,H-1$$

**(37) 加速度平滑度 AS**

$$S_a=(H-2)^{-1}\sum_{i=2}^{H-1}\|\bm a^i\|$$

$S_c,S_a$ 越小越平滑。另有 Safety Rate 与 Inference Time 两项指标（无公式）。

---

## 五、实验设置公式

### 5.1 平面导航（Section IV-B，maze，$d_s=2$）

**(38) 椭圆障碍**

$$\bar{\mathbb{C}}=\bigcup_{j=1}^{3}\mathbb{O}_j,\qquad \mathbb{O}_j=\{\bm s\in\mathbb{R}^2: h_j(\bm s)\le 0\}$$

$$h_j(\bm s):=(\bm s-\bm c_j)^\top\bm Q_j(\bm s-\bm c_j)-1,\qquad \bm Q_j=\mathrm{diag}(a_j^{-2},\,b_j^{-2})$$

**(39)(40)(41) 障碍参数**

$$\bm c_1=[3.5,\,4.0]^\top,\quad \bm c_2=[8.0,\,3.0]^\top,\quad \bm c_3=[7.0,\,6.5]^\top$$

$$a_1=2.5,\ b_1=1.25;\qquad a_2=1.75,\ b_2=1;\qquad a_3=1,\ b_3=1.5$$

**(42) 实际使用的 blow-up 函数**（$\varphi_0=1$）

$$\varphi_1(t)=\begin{cases}1+4t^3, & t<\gamma\\[2pt] 1/(T-t), & t\ge\gamma\end{cases},\qquad \gamma=0.9$$

> 工程细节：CFMBF 仅在 $t\ge t^\star=0.5$ 后激活（前期让分布成形，后期强化约束）；积分器用 `torchdiffeq` 的 `dopri5`，初始步长 $\epsilon=0.001$，自适应分段。

### 5.2 机械臂操作（Section IV-C，Franka Research 3，7-DoF）

**(43) 目标关节角**

$$\bm q^*_{target}=[2.0,\ 1.5,\ 2.0,\ -2.0,\ 2.0,\ 3.0,\ 2.0]^\top,\qquad \bm q_0^*=\bm 0_{7\times1}$$

**(44) 球形障碍的障碍函数**

$$h_j(\bm E_k\bm{\mathcal{T}})=\|\bm x_e(\bm s^i)-\bm c_j\|-\beta r_j,\qquad i=0,\dots,H,\ j=1,2$$

$\bm x_e(\bm s^i)$ 为 MuJoCo 正运动学给出的末端位置（$\bm s^i\in\mathbb{R}^7$ 为关节角），膨胀系数 $\beta=0.25$。

**(45) 障碍位置**

$$\bm c_1=[-0.62,\,0.30,\,0.50]^\top,\quad \bm c_2=[-0.05,\,0.78,\,0.55]^\top,\qquad r_1=0.125,\ r_2=0.25\ \mathrm{m}$$

**(46)(47) 起点/终点偏差（SD / ED）**

$$S_D=\big\|\bm E_0\bm{\mathcal{T}}-[(\bm q_0^*)^\top,\ \bm 0_{14\times1}^\top]^\top\big\|$$

$$E_D=\big\|\bm E_H\bm{\mathcal{T}}-[(\bm q^*_{target})^\top,\ \bm 0_{14\times1}^\top]^\top\big\|$$

> 状态是 21 维（7 关节角 + 7 速度 + 7 加速度/力矩），故补 $\bm 0_{14\times1}$。

---

## 六、算法 1：SafeFlow 推理流程

1. 初始化 normalizer 与 CFMBF scheduler
2. $\hat{\bm{\mathcal{T}}}_0\sim\mathcal{N}(\bm 0_{d\times1},\bm I_d)$
3. $t\leftarrow0$，积分步长 $\epsilon\leftarrow0.001$
4. **while** $t<1$ **do**
5. &nbsp;&nbsp;$\hat{\bm v}_t^{\bm\theta}(\hat{\bm{\mathcal{T}}}_t)\leftarrow$ 网络输出（归一化空间）
6. &nbsp;&nbsp;$\bm v_t^{\bm\theta}(\hat{\bm{\mathcal{T}}}_t)\leftarrow$ 反归一化到状态空间
7. &nbsp;&nbsp;$\bm{\mathcal{T}}_t\leftarrow$ 反归一化轨迹 $\hat{\bm{\mathcal{T}}}_t$
8. &nbsp;&nbsp;$\bm u_t\leftarrow$ 解 CFMBF-QP (29)（用 $t,\bm{\mathcal{T}}_t,\bm v_t^{\bm\theta}$）
9. &nbsp;&nbsp;$\hat{\bm u}_t\leftarrow$ 把 $\bm u_t$ 归一化回 FM 空间
10. &nbsp;&nbsp;$\bm{\mathcal{T}}_{4th},\bm{\mathcal{T}}_{5th}\leftarrow$ 用 $\bm v^{\bm\theta}(\cdot,\cdot)+\hat{\bm u}_t$ 做 4/5 阶 Runge–Kutta
11. &nbsp;&nbsp;绝对误差 $\Delta_{\mathcal{T}}$、相对误差 $\Delta_{\mathcal{T},r}$
12. &nbsp;&nbsp;**if** 两者均未超阈值 **then** $t\leftarrow t+\epsilon$，$\bm{\mathcal{T}}\leftarrow\bm{\mathcal{T}}_{5th}$
13. &nbsp;&nbsp;按 $\Delta_{\mathcal{T}},\Delta_{\mathcal{T},r}$ 自适应调整 $\epsilon$
14. **end while**
15. $\bm{\mathcal{T}}_1\leftarrow$ 终端 CFMBF 安全滤波器 (32) 作用于 $\bm{\mathcal{T}}_{1^-}$

---

## 七、核心逻辑串联

$$\underbrace{h(\bm E_k\bm{\mathcal{T}}_t)}_{\text{(4)(5) 安全定义}}\ \xrightarrow[\ \varphi_1\to\infty\ ]{\text{(7)(8) FMBF}}\ \underbrace{\text{规定时间进入 }\mathbb{C}_s}_{\text{Def.3, Thm.1}}\ \xrightarrow{\text{(12)–(16) QP 闭式解}}\ \bm u_t\ \xrightarrow{\text{(6) 引导 ODE}}\ \bm{\mathcal{T}}_{1^-}\ \xrightarrow{\text{(17)/(32) 终端滤波}}\ \bm{\mathcal{T}}_1\ \text{绝对安全}$$

**与传统 CBF 的本质区别**：CBF 要求初值 $\bm x(0)\in\mathbb{C}$；FMBF 用在 $t\to1^-$ 爆炸的 $\varphi_1$ 取代 class-$\mathcal{K}$ 函数 $\alpha$，允许初值不安全，并在有限的"生成时间"内被强制拉入安全集 —— 这正好匹配 FM/扩散从无界高斯起步的事实，也是 SafeDiffuser [22]、CoBL [23] 只能给出概率性安全的原因。
