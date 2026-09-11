function R = scan_1d_thresholds(thresholds, append_seed, n_roll)
%SCAN_1D_THRESHOLDS  Map KL against training-point retention for the 1D case.
%
% Instead of guessing which four thresholds might give a monotone KL curve,
% this sweeps a wide range and reports KL as a function of retention.  Any
% monotone stretch can then be read straight off the curve.
%
% Context: 40 rollout seeds and 50 append seeds were screened earlier, and the
% pair (threshold 0.20, retention ~15%) -> (threshold 0.15, retention ~21%) had
% the wrong order in 90/90 cases.  The question this answers is whether that
% inversion is a local feature of the 15-21% region or holds everywhere.
%
%   scan_1d_thresholds()                       % default wide sweep, seed 345
%   scan_1d_thresholds([0.3 0.2 0.1], 345)     % specific thresholds

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(thresholds)
    thresholds = [0.80 0.70 0.60 0.50 0.40 0.30 0.25 0.20 0.175 0.15 ...
                  0.125 0.10 0.08 0.06 0.05];
end
if nargin < 2 || isempty(append_seed), append_seed = 345; end
if nargin < 3 || isempty(n_roll),      n_roll = 1000; end

stem = ['1d case' char([20840 23616 71 80 35757 32451 38408 20540 23454 39564])];
dir25 = fullfile('outputs', [stem '_25x40']);
B = load(fullfile(dir25, 'Training_Data_and_Seeds.mat'), 'source_points', 'target_points');
H = load(fullfile(dir25, 'Manual_Hyperparameters.mat'), 'SigmaL', 'SigmaF');
sl = H.SigmaL; sf = H.SigmaF; sn = 0.25;
times_train = linspace(0, 1, 40)';
times_roll  = linspace(0, 1, 101)';
rollout_seed = 31;

rng(append_seed);
source = [B.source_points(:); randn(25,1)];
target = [B.target_points(:); sample_target_1d(25)];
vel = target - source;
nT = numel(times_train); nP = numel(source);
xs = zeros(nT, nP);
for ti = 1:nT
    xs(ti,:) = (1-times_train(ti))*source' + times_train(ti)*target';
end
X = [repmat(times_train, nP, 1), reshape(xs, [], 1)];
Y = reshape(repmat(vel', nT, 1), [], 1);
rng(rollout_seed); x0 = randn(n_roll, 1);

fprintf('threshold scan: append_seed %d, %d rollouts, %d candidate points\n', ...
    append_seed, n_roll, size(X,1));
fprintf('%-11s %-9s %-11s %-11s %-11s %-11s %-11s\n', 'threshold', 'points', ...
    'retention%', 'KL', 'endpt mean', 'endpt SD', '|endpt|<1');

nth = numel(thresholds);
ret = zeros(nth+1,1); kl = ret; emean = ret; esd = ret; enear = ret; npts = ret;
t0 = tic;
for i = 1:nth+1
    if i <= nth
        idx = select_points(X, sn, sf, sl, thresholds(i));
        thr = thresholds(i);
    else
        idx = (1:size(X,1))';
        thr = NaN;
    end
    npts(i) = numel(idx);
    ret(i) = 100 * numel(idx) / size(X,1);
    g = exact_gp(X(idx,:), Y(idx), sn, sf, sl);
    ep = rollout_endpoints(g, x0, times_roll);
    kl(i) = fixed_kde_kl_1d(target, ep);
    % Diagnostics for the "did the flow actually move the points" question.
    emean(i) = mean(ep); esd(i) = std(ep); enear(i) = mean(abs(ep) < 1);
    lbl = 'all data';
    if i <= nth, lbl = sprintf('%.3f', thr); end
    fprintf('%-11s %-9d %-11.2f %-11.5g %-11.4f %-11.4f %-11.3f\n', lbl, ...
        npts(i), ret(i), kl(i), emean(i), esd(i), enear(i));
end
fprintf('elapsed %.0f s\n', toc(t0));

R = table([thresholds(:); NaN], npts, ret, kl, emean, esd, enear, ...
    'VariableNames', {'threshold','points','retention_percent','KL', ...
    'endpoint_mean','endpoint_sd','frac_abs_lt_1'});
R = sortrows(R, 'retention_percent');

fprintf('\nsorted by retention (KL should fall going down if monotone):\n');
disp(R(:, {'threshold','retention_percent','KL','endpoint_sd','frac_abs_lt_1'}));
d = diff(R.KL);
fprintf('monotone decreasing across the whole scan: %d\n', all(d < 0));
bad = find(d >= 0);
for b = bad(:)'
    fprintf('  INVERSION  retention %6.2f%% -> %6.2f%%   KL %.5g -> %.5g\n', ...
        R.retention_percent(b), R.retention_percent(b+1), R.KL(b), R.KL(b+1));
end
[~, worst] = max(R.KL);
fprintf('\nworst KL at retention %.2f%% (threshold %.3f)\n', ...
    R.retention_percent(worst), R.threshold(worst));
fprintf('for reference, the target mixture has SD %.4f and %.3f of its mass in |x|<1\n', ...
    std(target), mean(abs(target) < 1));

out = fullfile('outputs', sprintf('Simple1D_ThresholdScan_seed%d.csv', append_seed));
writetable(R, out);
fprintf('wrote %s\n', out);
end

% =====================================================================
function idx = select_points(X, sn, sf, sl, threshold)
n = size(X, 1); Xt = X';
idx = zeros(n, 1); c = 0; L = [];
for i = 1:n
    if c == 0
        take = true;
    else
        v = L \ kern(Xt(:, idx(1:c)), Xt(:, i), sf, sl);
        take = sqrt(max(sf^2 - (v' * v), 0)) > threshold;
    end
    if take
        c = c + 1; idx(c) = i;
        L = chol_append(L, Xt(:, idx(1:c)), sn, sf, sl);
    end
end
idx = idx(1:c);
end

function L = chol_append(L, Xsub, sn, sf, sl)
c = size(Xsub, 2);
if c == 1, L = sqrt(sf^2 + sn^2); return; end
l = L \ kern(Xsub(:, 1:c-1), Xsub(:, c), sf, sl);
d = sqrt(max(sf^2 + sn^2 - (l' * l), eps));
L = [L, zeros(c-1, 1); l', d];
end

function m = exact_gp(Xs, Ys, sn, sf, sl)
Xt = Xs';
K = kern(Xt, Xt, sf, sl) + sn^2 * eye(size(Xs, 1));
L = chol(K, 'lower');
m = struct('X', Xt, 'alpha', L' \ (L \ Ys), 'SigmaF', sf, 'SigmaL', sl);
end

function k = kern(X, Q, sf, sl)
dx = reshape(X, 2, [], 1) - reshape(Q, 2, 1, []);
scaled = dx ./ reshape(sl(:), 2, 1, 1);
k = sf^2 * exp(-0.5 * squeeze(sum(scaled .^ 2, 1)));
if size(Q, 2) == 1, k = reshape(k, [], 1); end
end

function ep = rollout_endpoints(m, x0, times)
x = x0(:)';
for ti = 1:numel(times) - 1
    t = times(ti); h = times(ti+1) - t;
    k1 = mu(m, t, x);
    k2 = mu(m, t + h/2, x + h*k1/2);
    k3 = mu(m, t + h/2, x + h*k2/2);
    k4 = mu(m, t + h,   x + h*k3);
    x = x + h*(k1 + 2*k2 + 2*k3 + k4)/6;
end
ep = x(:);
end

function v = mu(m, t, x)
v = m.alpha' * kern(m.X, [repmat(t, 1, numel(x)); x], m.SigmaF, m.SigmaL);
end

function target = sample_target_1d(n)
component = rand(n, 1) > 0.5;
target = (-2.0 + 0.45*randn(n,1)) .* (~component) + ...
         ( 2.0 + 0.55*randn(n,1)) .* component;
end

function value = fixed_kde_kl_1d(reference, generated)
n = numel(reference); bw = max(1.06 * std(reference) * n^(-1/5), 1e-3);
grid = linspace(min(reference) - 6*bw, max(reference) + 6*bw, 1024)';
p = kde_1d(grid, reference, bw); q = kde_1d(grid, generated, bw);
p = p / sum(p); q = q / sum(q); fl = 1e-300;
value = sum(p .* log(max(p, fl) ./ max(q, fl)));
end

function d = kde_1d(grid, samples, bw)
d = zeros(size(grid));
for i = 1:numel(samples)
    d = d + exp(-.5 * ((grid - samples(i)) / bw) .^ 2);
end
d = d / max(numel(samples), 1);
end
