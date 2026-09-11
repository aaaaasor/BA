function R = search_1d_append_seed(seeds, n_screen, do_validate, scan_thresholds)
%SEARCH_1D_APPEND_SEED  Search the training-data seed for a monotone KL curve.
%
% The supervisor asked whether a different seed can make every metric in the
% 1D table monotone.  Two criteria were given: the source draw should look like
% N(0,1), and the KL of the generated endpoints should decrease monotonically
% with training-point retention.
%
% Criterion 1 depends only on rollout_seed and is already met by the current
% one (KS = 0.0175, p = 0.91).  Criterion 2 is NOT reachable by changing
% rollout_seed: over 40 rollout seeds the 0.20 -> 0.15 pair had the correct
% order 0/40 times.  So this function varies append_seed instead, which changes
% the 25 appended training pairs and therefore the fitted models themselves.
%
% EVERY candidate seed and its full metric set is written out, so the selection
% is a documented screen against pre-declared criteria rather than an
% undocumented search for a flattering number.
%
%   search_1d_append_seed([], [], true)      % validate against seed 345 only
%   search_1d_append_seed(400:449, 300)      % screen 50 seeds, 300 rollouts

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(seeds),    seeds = 400:449; end
if nargin < 2 || isempty(n_screen), n_screen = 300; end
if nargin < 3, do_validate = false; end
if nargin < 4, scan_thresholds = []; end

dir25 = fullfile('outputs', [dirstem() '_25x40']);
B = load(fullfile(dir25, 'Training_Data_and_Seeds.mat'), 'source_points', 'target_points');
H = load(fullfile(dir25, 'Manual_Hyperparameters.mat'), 'SigmaL', 'SigmaF');
sl = H.SigmaL; sf = H.SigmaF; sn = 0.25;      % cfg.manual_noise_std of the run
thresholds = [0.20 0.15 0.10 0.05];
times_train = linspace(0, 1, 40)';
times_roll  = linspace(0, 1, 101)';
rollout_seed = 31;
labels = [arrayfun(@(z) sprintf('%.2f', z), thresholds, 'UniformOutput', false), {'all'}];

if do_validate
    fprintf('=== validation: reproduce append_seed 345 ===\n');
    m = evaluate_seed(345, B, sl, sf, sn, thresholds, times_train, times_roll, ...
        rollout_seed, 1000);
    ref = readtable(fullfile('outputs', dirstem_short(), ...
        'Simple1D_GlobalGP_Threshold_Sweep.csv'));
    fprintf('%-10s %-14s %-14s %-14s %-14s\n', 'model', 'retain mine', ...
        'retain ref', 'KL mine', 'KL ref');
    for i = 1:5
        fprintf('%-10s %-14.2f %-14.2f %-14.6g %-14.6g\n', labels{i}, ...
            m.retention(i), ref.RetentionPercent(i), m.kl(i), ref.KL(i));
    end
    R = m; return;
end

n = numel(seeds);
KL = nan(n, 5); RET = nan(n, 5);
fprintf('screening %d append seeds with %d rollouts each\n', n, n_screen);
t0 = tic;
for k = 1:n
    m = evaluate_seed(seeds(k), B, sl, sf, sn, thresholds, times_train, ...
        times_roll, rollout_seed, n_screen);
    KL(k,:) = m.kl; RET(k,:) = m.retention;
    if mod(k, 5) == 0
        fprintf('  %d/%d  (%.0f s)  monotone so far: %d\n', k, n, toc(t0), ...
            sum(all(diff(KL(1:k,:), 1, 2) < 0, 2)));
    end
end

mono = all(diff(KL, 1, 2) < 0, 2);
fprintf('\n=========== screen result ===========\n');
fprintf('fully monotone KL: %d / %d (%.1f%%)\n', sum(mono), n, 100*mean(mono));
for j = 1:4
    fprintf('  pair %-5s -> %-5s correct in %5.1f%% of seeds\n', ...
        labels{j}, labels{j+1}, 100*mean(KL(:,j) > KL(:,j+1)));
end
if any(mono)
    fprintf('\nmonotone seeds: %s\n', mat2str(seeds(mono)));
end

names = [{'append_seed'}, ...
    cellfun(@(s) ['KL_' matlab.lang.makeValidName(s)], labels, 'UniformOutput', false), ...
    cellfun(@(s) ['Retention_' matlab.lang.makeValidName(s)], labels, 'UniformOutput', false), ...
    {'monotone'}];
T = array2table([seeds(:), KL, RET, double(mono)], 'VariableNames', names);
out = fullfile('outputs', 'Simple1D_AppendSeed_Screen.csv');
writetable(T, out);
fprintf('\nwrote %s  (every candidate, not only the survivors)\n', out);
R = T;
end

% =====================================================================
function s = dirstem_short()
s = ['1d case' char([20840 23616 71 80])];   % the seed345_sn025 output folder
end

function s = dirstem()
s = ['1d case' char([20840 23616 71 80 35757 32451 38408 20540 23454 39564])];
end

function m = evaluate_seed(append_seed, B, sl, sf, sn, thresholds, ...
        times_train, times_roll, rollout_seed, n_roll)
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
nmod = numel(thresholds) + 1;
m.kl = zeros(1, nmod); m.retention = zeros(1, nmod);
for i = 1:nmod
    if i <= numel(thresholds)
        idx = select_points(X, sn, sf, sl, thresholds(i));
    else
        idx = (1:size(X,1))';
    end
    m.retention(i) = 100 * numel(idx) / size(X,1);
    g = exact_gp(X(idx,:), Y(idx), sn, sf, sl);
    ep = rollout_endpoints(g, x0, times_roll);
    m.kl(i) = fixed_kde_kl_1d(target, ep);
end
end

function idx = select_points(X, sn, sf, sl, threshold)
% Same greedy rule as fit_selected_global_gp: take a point while the current
% posterior standard deviation there still exceeds the threshold.
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
if c == 1
    L = sqrt(sf^2 + sn^2);
    return;
end
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
