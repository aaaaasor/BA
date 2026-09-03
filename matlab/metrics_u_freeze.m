function metrics_u_freeze()
%METRICS_U_FREEZE 对 compare_u_freeze 的四种配置算 KL / CS / AS。
%
% 配对偏差（compare_u_freeze 里的"相对偏差"）回答的是"还是不是同一条轨迹"；
% KL/CS/AS 回答的是"质量有没有变"。生成模型里这是两个不同的问题。
% KL 参考网格由第一层训练数据端点建立一次，四种配置共用，因此可比。

this_dir = fileparts(mfilename('fullpath'));
L = load(fullfile(this_dir, 'outputs', 'Racing_UFreeze_Comparison.mat'));
variants = L.variants;

% ---- 重建 data_transform 与训练端点（确定性，走缓存）----
cfg = get_config();
rng(cfg.random_seed);
[first_pts, track_segment] = scenario_training_points(cfg, 5, cfg.n_train);
cfg.track_segment = track_segment;
rng(cfg.first_level_data_seed);
[~,~,~, target_points, ~,~,~,~, data_transform] = build_training_data( ...
    cfg.t_min, 1.0, cfg.n_time_slices, first_pts);
fdim = size(target_points, 3);
nPt  = size(target_points, 1);
fprintf('第一层: %d 点/条, %d 特征/点\n', nPt, fdim);

% ---- 训练集端点，作为 KL 的参考分布 ----
data_final = squeeze(target_points(end, :, 1:2));      % (n_train, 2)
ref = make_ref(data_final, 128);
fprintf('KL 参考: 带宽 [%.4g %.4g], 网格 128, 基于 %d 个训练端点\n', ...
    ref.bw, size(data_final,1));

fprintf('\n%-10s %8s %10s %10s %9s %9s\n', ...
    '配置', 'Safety', 'KL', 'CS', 'AS', '出界');
for vi = 1:size(variants,1)
    v = variants{vi,1};
    P = decode_all(L.res.(v).path, data_transform, fdim, nPt);   % (nPt, n, 2)
    [cs, as] = smoothness(P);
    fin = squeeze(P(end,:,:));
    [kl, nout] = kl_vs_ref(fin, ref);
    fprintf('%-10s %7.1f%% %10.4f %10.4f %9.4f %6d/%d\n', ...
        v, 100*L.res.(v).safety, kl, cs, as, nout, size(fin,1));
end

% 训练集自身作为 sanity check（KL 应接近 0）
[cs_d, as_d] = smoothness(permute(target_points(:,:,1:2), [1 2 3]));
kl_d = kl_vs_ref(data_final, ref);
fprintf('%-10s %7s %10.4f %10.4f %9.4f\n', '训练数据', '-', kl_d, cs_d, as_d);
end

% =====================================================================
function P = decode_all(path, dt, fdim, nPt)
n = size(path, 2);
P = zeros(nPt, n, 2);
for i = 1:n
    x_abs = squeeze(path(end, i, :)) .* dt.std(:) + dt.mean(:);
    M = reshape(x_abs, fdim, nPt);
    P(:, i, :) = M(1:2, :)';
end
end

function [cs, as] = smoothness(P)
n = size(P, 2);
c = nan(n,1); a = nan(n,1);
for i = 1:n
    xy = squeeze(P(:,i,1:2));
    w  = diff(xy,1,1);
    Lg = sqrt(sum(w.^2,2));
    den = Lg(1:end-1).*Lg(2:end);
    ct = ones(size(den));  ok = den > eps;
    ct(ok) = sum(w(1:end-1,:).*w(2:end,:),2)./den(ok);
    c(i) = mean(1 - min(max(ct,-1),1));
    a(i) = mean(sqrt(sum(diff(xy,2,1).^2,2)));
end
cs = mean(c, 'omitnan'); as = mean(a, 'omitnan');
end

function ref = make_ref(data_xy, gs)
sc = std(data_xy, 0, 1);
sp = max(data_xy,[],1) - min(data_xy,[],1);
sc = max(sc, max(sp*1e-3, 1e-8));
bw = sc * size(data_xy,1)^(-1/6);
lo = min(data_xy,[],1) - 6*bw;  hi = max(data_xy,[],1) + 6*bw;
xg = linspace(lo(1), hi(1), gs);  yg = linspace(lo(2), hi(2), gs);
[gx, gy] = meshgrid(xg, yg);
g = [gx(:), gy(:)];
p = kde(g, data_xy, bw);
step = [xg(2)-xg(1), yg(2)-yg(1)];
p = p / (sum(p)*prod(step));
ref = struct('xg',xg,'yg',yg,'grid',g,'p',p,'bw',bw,'step',step);
end

function [kl, nout] = kl_vs_ref(gen_xy, ref)
q = max(kde(ref.grid, gen_xy, ref.bw), realmin('double'));
pos = ref.p > 0;
kl = max(0, sum(ref.p(pos).*(log(ref.p(pos)) - log(q(pos)))) * prod(ref.step));
nout = nnz(gen_xy(:,1) < ref.xg(1) | gen_xy(:,1) > ref.xg(end) | ...
           gen_xy(:,2) < ref.yg(1) | gen_xy(:,2) > ref.yg(end));
end

function d = kde(g, s, bw)
d = zeros(size(g,1),1);
for i = 1:size(s,1)
    dv = (g - s(i,:)) ./ bw;
    d = d + exp(-0.5*sum(dv.^2,2));
end
d = d / max(size(s,1),1) / (2*pi*prod(bw));
end
