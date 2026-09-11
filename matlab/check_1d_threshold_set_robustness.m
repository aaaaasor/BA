function R = check_1d_threshold_set_robustness(sets, rollout_seeds, append_seed, n_roll)
%CHECK_1D_THRESHOLD_SET_ROBUSTNESS  Is a monotone KL curve stable, or luck?
%
% The wide threshold scan (scan_1d_thresholds) found several four-threshold
% sets whose KL decreases monotonically with retention -- but it measured each
% model once, with rollout_seed 31.  KL over rollout seeds has an SD of roughly
% 0.006-0.015, and some of the winning margins are smaller than that, so those
% orderings could be sampling noise.
%
% This re-evaluates each candidate set over many rollout seeds and reports the
% fraction of seeds for which the set is monotone.  A set that is monotone in,
% say, 95% of seeds is a real property; one that is monotone in 55% is a coin
% flip and must not be reported as a trend.
%
% The models are fitted once per threshold and reused across seeds, so the cost
% is dominated by the rollouts.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(sets)
    sets = { [0.80 0.60 0.40 0.20], ...
             [0.70 0.60 0.50 0.40], ...
             [0.70 0.60 0.30 0.05], ...
             [0.20 0.15 0.10 0.05] };      % the set currently in the thesis
end
if nargin < 2 || isempty(rollout_seeds), rollout_seeds = 1000:1029; end
if nargin < 3 || isempty(append_seed),   append_seed = 345; end
if nargin < 4 || isempty(n_roll),        n_roll = 1000; end

stem = ['1d case' char([20840 23616 71 80 35757 32451 38408 20540 23454 39564])];
dir25 = fullfile('outputs', [stem '_25x40']);
B = load(fullfile(dir25, 'Training_Data_and_Seeds.mat'), 'source_points', 'target_points');
H = load(fullfile(dir25, 'Manual_Hyperparameters.mat'), 'SigmaL', 'SigmaF');
sl = H.SigmaL; sf = H.SigmaF; sn = 0.25;
times_train = linspace(0, 1, 40)';
times_roll  = linspace(0, 1, 101)';

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

% Fit every threshold that appears in any candidate set, once.
all_thr = unique([sets{:}]);
models = containers.Map('KeyType', 'double', 'ValueType', 'any');
ret = containers.Map('KeyType', 'double', 'ValueType', 'any');
fprintf('fitting %d distinct thresholds\n', numel(all_thr));
for thr = all_thr
    idx = select_points(X, sn, sf, sl, thr);
    models(thr) = exact_gp(X(idx,:), Y(idx), sn, sf, sl);
    ret(thr) = 100 * numel(idx) / size(X,1);
    fprintf('  threshold %.3f -> %5.2f%% retention\n', thr, ret(thr));
end
all_model = exact_gp(X, Y, sn, sf, sl);

ns = numel(rollout_seeds);
KLs = containers.Map('KeyType', 'double', 'ValueType', 'any');
for thr = all_thr, KLs(thr) = zeros(ns,1); end
kl_all = zeros(ns,1);
fprintf('\nrolling out %d seeds x %d trajectories\n', ns, n_roll);
t0 = tic;
for k = 1:ns
    rng(rollout_seeds(k)); x0 = randn(n_roll, 1);
    for thr = all_thr
        v = KLs(thr);
        v(k) = fixed_kde_kl_1d(target, rollout_endpoints(models(thr), x0, times_roll));
        KLs(thr) = v;
    end
    kl_all(k) = fixed_kde_kl_1d(target, rollout_endpoints(all_model, x0, times_roll));
    if mod(k, 10) == 0, fprintf('  %d/%d  (%.0f s)\n', k, ns, toc(t0)); end
end

fprintf('\n=========== per-threshold KL over %d rollout seeds ===========\n', ns);
fprintf('%-11s %-11s %-12s %-12s\n', 'threshold', 'retention%', 'KL mean', 'KL SD');
for thr = all_thr
    v = KLs(thr);
    fprintf('%-11.3f %-11.2f %-12.6g %-12.6g\n', thr, ret(thr), mean(v), std(v));
end
fprintf('%-11s %-11.2f %-12.6g %-12.6g\n', 'all data', 100, mean(kl_all), std(kl_all));

fprintf('\n=========== candidate threshold sets ===========\n');
rows = cell(numel(sets), 1);
for si = 1:numel(sets)
    thr = sets{si};
    [~, ord] = sort(cellfun(@(t) ret(t), num2cell(thr)));   % retention ascending
    thr = thr(ord);
    M = zeros(ns, numel(thr) + 1);
    for j = 1:numel(thr), M(:,j) = KLs(thr(j)); end
    M(:,end) = kl_all;
    mono = all(diff(M, 1, 2) < 0, 2);
    fprintf('\nset %s   (retention %s)\n', mat2str(thr), ...
        mat2str(round(cellfun(@(t) ret(t), num2cell(thr)), 2)));
    fprintf('  monotone in %d/%d seeds (%.1f%%)\n', sum(mono), ns, 100*mean(mono));
    for j = 1:size(M,2)-1
        fprintf('    step %d -> %d correct in %5.1f%% of seeds  (mean gap %+.5g)\n', ...
            j, j+1, 100*mean(M(:,j) > M(:,j+1)), mean(M(:,j) - M(:,j+1)));
    end
    rows{si} = struct('set', mat2str(thr), 'monotone_rate', mean(mono));
end

R = struct('sets', {sets}, 'rollout_seeds', rollout_seeds, ...
    'summary', {rows}, 'retention', ret, 'KL', KLs, 'KL_all', kl_all);
save(fullfile('outputs', 'Simple1D_ThresholdSet_Robustness.mat'), '-struct', 'R');
fprintf('\nwrote outputs/Simple1D_ThresholdSet_Robustness.mat\n');
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
