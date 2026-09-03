function out = safeflow_nn_train(n_steps, do_plot)
%SAFEFLOW_NN_TRAIN SafeFlow 基线的流匹配 MLP：261 -> 256^3 -> 260
%
%   out = safeflow_nn_train(5000, true)     % 小跑一次
%   out = safeflow_nn_train(20000, true)    % 完整训练
%
% 纯基础 MATLAB，不需要 Deep Learning Toolbox。
% 每点状态为 [x,y,dx/ds,dy/ds]，完整轨迹状态属于 R^(65*4)=R^260。
% 安全约束只依赖位置；切向特征由同一个 NN 流场联合生成。
% 只训练与采样，不含安全约束——那是下一步。

if nargin < 1 || isempty(n_steps), n_steps = 5000; end
if nargin < 2, do_plot = true; end
this_dir = fileparts(mfilename('fullpath'));

%% ---------- 数据 ----------
cfg = get_config();
rng(cfg.random_seed);
[pts, segment] = scenario_training_points(cfg, 65, cfg.n_train);   % (65, N, 4)
features = pts(:, :, 1:4);
% layout='dedup' : 65 点 x 4 特征 = 260 维（重建后的去重轨迹）
% layout='segment': 16 段 x 5 点 x 4 特征 = 320 维（第三层实际的分段布局，
%                   段边界点重复出现两次，与 cfg.segment_points_per_segment 一致）
layout = struct_field_default(cfg, 'safeflow_nn_layout', 'dedup');
switch layout
    case 'dedup'
        features_used = features;
        seg_index = [];
    case 'segment'
        spp = cfg.segment_points_per_segment;              % 5
        nSeg = (size(features,1) - 1) / (spp - 1);         % 16
        assert(nSeg == round(nSeg), '65 点无法整分成每段 %d 点。', spp);
        seg_index = zeros(nSeg*spp, 1);
        for sgi = 1:nSeg
            seg_index((sgi-1)*spp + (1:spp)) = (sgi-1)*(spp-1) + (1:spp);
        end
        features_used = features(seg_index, :, :);
    otherwise
        error('未知 layout: %s', layout);
end
XY = features_used(:, :, 1:2);
nP = size(features_used, 1);
N  = size(features_used, 2);
nF = size(features_used, 3);
D  = nP*nF;
X1 = reshape(permute(features_used, [3 1 2]), D, N); % [x;y;dx/ds;dy/ds] per point

mu_d = mean(X1, 2);
sd_d = std(X1, 0, 2);  sd_d(sd_d < 1e-12) = 1;
X1n  = (X1 - mu_d) ./ sd_d;                 % 标准化，和 GP 流程口径一致

fprintf('数据: %d 条轨迹 x %d 维\n', N, D);

%% ---------- 网络 ----------
H = 256;  Din = D + 1;  Dout = D;
weight_init_seed = struct_field_default(cfg, 'safeflow_nn_init_seed', 20260101);
rng(weight_init_seed);
P = { randn(H,Din)*sqrt(2/Din), randn(H,H)*sqrt(2/H), ...
      randn(H,H)*sqrt(2/H),     randn(Dout,H)*sqrt(2/H), ...
      zeros(H,1), zeros(H,1), zeros(H,1), zeros(Dout,1) };
m = cellfun(@(p) zeros(size(p)), P, 'UniformOutput', false);
v = m;
lr = 1e-3; be1 = 0.9; be2 = 0.999; epA = 1e-8; B = 256;
fprintf('网络: %d -> %d^3 -> %d,  参数 %d\n', Din, H, Dout, ...
    sum(cellfun(@numel, P)));

%% ---------- 训练 ----------
loss_hist = zeros(n_steps, 1);
tic;
for k = 1:n_steps
    idx = randi(N, 1, B);
    x1  = X1n(:, idx);
    x0  = randn(D, B);
    t   = rand(1, B);
    xt  = (1 - t) .* x0 + t .* x1;
    vt  = x1 - x0;                       % 流匹配监督速度

    [y, cache] = fwd(P, [t; xt]);
    diffv = y - vt;
    loss_hist(k) = mean(diffv.^2, 'all');
    G = bwd(P, cache, 2*diffv/(Dout*B));

    for j = 1:8
        m{j} = be1*m{j} + (1-be1)*G{j};
        v{j} = be2*v{j} + (1-be2)*(G{j}.^2);
        P{j} = P{j} - lr*(m{j}/(1-be1^k))./(sqrt(v{j}/(1-be2^k))+epA);
    end
    if mod(k, max(1,floor(n_steps/10))) == 0
        fprintf('  step %6d / %d   loss %.5f\n', k, n_steps, ...
            mean(loss_hist(max(1,k-199):k)));
    end
end
train_s = toc;
fprintf('训练耗时 %.1f s  (%.2f ms/step)\n', train_s, train_s/n_steps*1000);

%% ---------- 采样（无约束 RK4，100 步）----------
n_gen = 20; n_rk = 100; t_max = 0.996;
rng(cfg.first_level_rollout_seed);
z = randn(D, n_gen);
dt = t_max / n_rk;
tic;
for k = 1:n_rk
    tk = (k-1)*dt;
    k1 = fwd(P, [repmat(tk,        1,n_gen); z]);
    k2 = fwd(P, [repmat(tk+dt/2,   1,n_gen); z + dt/2*k1]);
    k3 = fwd(P, [repmat(tk+dt/2,   1,n_gen); z + dt/2*k2]);
    k4 = fwd(P, [repmat(tk+dt,     1,n_gen); z + dt*k3]);
    z  = z + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end
roll_s = toc;
fprintf('采样 %d 条 x %d 步耗时 %.2f s\n', n_gen, n_rk, roll_s);

Xgen = z .* sd_d + mu_d;                    % 反标准化
Fgen = permute(reshape(Xgen, nF, nP, n_gen), [2 3 1]); % (65,n_gen,4)
Pgen = Fgen(:,:,1:2);

out = struct('P', {P}, 'mu_d', mu_d, 'sd_d', sd_d, 'loss', loss_hist, ...
    'train_seconds', train_s, 'rollout_seconds', roll_s, ...
    'generated', Pgen, 'generated_features', Fgen, ...
    'segment', segment, 'X1', X1, 'n_points', nP, ...
    'layout', layout, 'seg_index', seg_index, ...
    'weight_init_seed', weight_init_seed, ...
    'data_seed', cfg.random_seed, 'n_train_steps', n_steps, ...
    'smoke_rollout_seed', cfg.first_level_rollout_seed, ...
    'segment_points_per_segment', cfg.segment_points_per_segment, ...
    'n_features_per_point', nF, ...
    'feature_order', {{'x','y','dx_ds','dy_ds'}});

%% ---------- 画图 ----------
if do_plot
    f = figure('Color','w','Position',[100 100 1100 460]);
    subplot(1,2,1);
    semilogy(movmean(loss_hist, 50), 'LineWidth', 1.2); grid on;
    xlabel('step'); ylabel('flow matching loss'); title('Training Loss');

    subplot(1,2,2); hold on;
    draw_track_segment(segment, 'HandleVisibility','off');
    for i = 1:N
        plot(XY(:,i,1), XY(:,i,2), '-', 'Color', [.75 .75 .75], ...
            'LineWidth', .8, 'HandleVisibility','off');
    end
    for i = 1:n_gen
        plot(Pgen(:,i,1), Pgen(:,i,2), '-', 'LineWidth', 1.4, ...
            'HandleVisibility','off');
    end
    plot(nan,nan,'-','Color',[.75 .75 .75],'DisplayName',sprintf('Training trajectories (%d)', N));
    plot(nan,nan,'-','Color',[0 .45 .74],'DisplayName',sprintf('Generated trajectories (%d)', n_gen));
    axis equal; grid on; legend('Location','best');
    title(sprintf('Generated Samples after %d Training Steps', n_steps));

    png = fullfile(this_dir, 'outputs', 'SafeFlowNN_smoke_test.png');
    exportgraphics(f, png, 'Resolution', 130);
    fprintf('图已保存: %s\n', png);
end
end

% =====================================================================
function [y, c] = fwd(P, x)
sil = @(z) z./(1+exp(-z));
z1 = P{1}*x  + P{5};  a1 = sil(z1);
z2 = P{2}*a1 + P{6};  a2 = sil(z2);
z3 = P{3}*a2 + P{7};  a3 = sil(z3);
y  = P{4}*a3 + P{8};
c = struct('x',x,'z1',z1,'a1',a1,'z2',z2,'a2',a2,'z3',z3,'a3',a3);
end

function G = bwd(P, c, dy)
dsl = @(z) (1./(1+exp(-z))).*(1 + z.*(1-1./(1+exp(-z))));
g4 = dy*c.a3.';                      gb4 = sum(dy,2);
d3 = (P{4}.'*dy).*dsl(c.z3);
g3 = d3*c.a2.';                      gb3 = sum(d3,2);
d2 = (P{3}.'*d3).*dsl(c.z2);
g2 = d2*c.a1.';                      gb2 = sum(d2,2);
d1 = (P{2}.'*d2).*dsl(c.z1);
g1 = d1*c.x.';                       gb1 = sum(d1,2);
G = {g1,g2,g3,g4,gb1,gb2,gb3,gb4};
end
