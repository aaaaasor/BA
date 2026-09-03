function diagnose_u_freeze(variant)
%DIAGNOSE_U_FREEZE 相对偏差是哪来的：一步之内的误差，还是逐步放大？
if nargin < 1 || isempty(variant), variant = 'per_step'; end
this_dir = fileparts(mfilename('fullpath'));
L = load(fullfile(this_dir, 'outputs', 'Racing_UFreeze_Comparison.mat'));
fprintf('=== %s vs per_stage ===\n', variant);
A = L.res.per_stage.path;   B = L.res.(variant).path;   % (n_t, n_s, dim)
t = L.res.per_stage.times;
[nT, nS, ~] = size(A);

% 每个时刻的偏差范数（对样本取均值），以及状态自身的量级
dev = zeros(nT,1); mag = zeros(nT,1);
for k = 1:nT
    d = 0; m = 0;
    for i = 1:nS
        d = d + norm(squeeze(A(k,i,:)) - squeeze(B(k,i,:)));
        m = m + norm(squeeze(A(k,i,:)));
    end
    dev(k) = d/nS;  mag(k) = m/nS;
end

fprintf('时刻      t        |A-B|        |A|      相对\n');
idx = unique(round(linspace(1, nT, 18)));
for k = idx
    fprintf('  %4d  %.4f  %.6e  %.4e  %7.3f%%\n', ...
        k, t(k), dev(k), mag(k), 100*dev(k)/max(mag(k),eps));
end

% 逐步增长率：dev(k+1)/dev(k)
r = dev(2:end)./max(dev(1:end-1), realmin);
fin = isfinite(r) & dev(1:end-1) > 1e-14;
fprintf('\n逐步放大因子 dev(k+1)/dev(k):  中位数 %.4f, 90%% 分位 %.4f, 最大 %.4f\n', ...
    median(r(fin)), quantile(r(fin), 0.9), max(r(fin)));
fprintf('若为纯放大, 100 步后累积 = %.3g\n', median(r(fin))^nT);

% 第一步就产生多少偏差？
fprintf('\n第 1 步后偏差 %.6e  (占最终 %.3f%%)\n', dev(2), 100*dev(2)/dev(end));
fprintf('第 10 步后    %.6e  (占最终 %.3f%%)\n', dev(min(11,nT)), ...
    100*dev(min(11,nT))/dev(end));
fprintf('最终          %.6e\n', dev(end));

f = figure('Color','w','Position',[100 100 900 380]);
subplot(1,2,1);
semilogy(t, max(dev, realmin), 'LineWidth', 1.4); grid on;
xlabel('t'); ylabel('mean |A-B|'); title('偏差随时间（对数轴）');
subplot(1,2,2);
plot(t, 100*dev./max(mag,eps), 'LineWidth', 1.4); grid on;
xlabel('t'); ylabel('相对偏差 %'); title('相对状态量级');
exportgraphics(f, fullfile(this_dir,'outputs','UFreeze_Divergence.png'), 'Resolution', 130);
fprintf('\n图: outputs/UFreeze_Divergence.png\n');
end
