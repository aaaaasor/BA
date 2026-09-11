function R = enumerate_1d_threshold_sets(grid, rollout_seeds, n_roll, append_seed)
%ENUMERATE_1D_THRESHOLD_SETS  Which equally spaced threshold set is monotone?
%
% Requirements from the supervisor, in order:
%   (a) the four thresholds must be equally spaced;
%   (b) the resulting retentions must spread out, not bunch at high values
%       (80/90/95% would carry no discrimination);
%   (c) KL must decrease as retention increases.
%
% Rather than guessing a set and testing it, this fits every threshold on a
% grid once, rolls each out over several seeds, and then scores EVERY equally
% spaced four-subset.  The monotonicity rate over rollout seeds separates a
% real ordering from one that holds for a single lucky draw -- the KL SD across
% rollout seeds is 0.006-0.015, which is larger than several of the gaps seen
% in the single-seed scan.
%
% Screening defaults are deliberately cheap; re-check any winner with
% check_1d_threshold_set_robustness at full sample size before adopting it.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(grid),          grid = 0.05:0.01:0.35; end
if nargin < 2 || isempty(rollout_seeds), rollout_seeds = 1000:1009; end
if nargin < 3 || isempty(n_roll),        n_roll = 500; end
if nargin < 4 || isempty(append_seed),   append_seed = 345; end

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

ng = numel(grid); ns = numel(rollout_seeds);
ret = zeros(ng,1); KL = zeros(ns, ng);
fprintf('fitting %d thresholds\n', ng);
t0 = tic;
mods = cell(ng,1);
for i = 1:ng
    idx = select_points(X, sn, sf, sl, grid(i));
    ret(i) = 100*numel(idx)/size(X,1);
    mods{i} = exact_gp(X(idx,:), Y(idx), sn, sf, sl);
    fprintf('  %.3f -> %6.2f%%  (%d pts, %.0f s)\n', grid(i), ret(i), numel(idx), toc(t0));
end
all_model = exact_gp(X, Y, sn, sf, sl);

fprintf('\nrolling out %d seeds x %d trajectories\n', ns, n_roll);
kl_all = zeros(ns,1);
for k = 1:ns
    rng(rollout_seeds(k)); x0 = randn(n_roll,1);
    for i = 1:ng
        KL(k,i) = fixed_kde_kl_1d(target, rollout_endpoints(mods{i}, x0, times_roll));
    end
    kl_all(k) = fixed_kde_kl_1d(target, rollout_endpoints(all_model, x0, times_roll));
    fprintf('  seed %d/%d (%.0f s)\n', k, ns, toc(t0));
end

fprintf('\n%-9s %-11s %-12s %-12s\n', 'threshold', 'retention%', 'KL mean', 'KL SD');
for i = 1:ng
    fprintf('%-9.3f %-11.2f %-12.6g %-12.6g\n', grid(i), ret(i), mean(KL(:,i)), std(KL(:,i)));
end
fprintf('%-9s %-11.2f %-12.6g %-12.6g\n', 'all', 100, mean(kl_all), std(kl_all));

% Enumerate equally spaced four-subsets on the grid.
step_q = round(diff(grid(1:2)), 10);
cand = [];
for si = 1:ng
    for stp = 1:floor((ng-1)/3)
        idx = si + (0:3)*stp;
        if idx(end) > ng, break; end
        i4 = fliplr(idx);                    % retention ascending
        r4 = ret(i4);
        if any(diff(r4) <= 0), continue; end
        M = [KL(:, i4), kl_all];
        mono = all(diff(M,1,2) < 0, 2);
        cand(end+1,:) = [grid(idx(1)), stp*step_q, r4(:)', mean(mono)]; %#ok<AGROW>
    end
end
C = array2table(cand, 'VariableNames', {'smallest_threshold','spacing', ...
    'ret1','ret2','ret3','ret4','monotone_rate'});
C.spread = C.ret4 - C.ret1;
C = sortrows(C, {'monotone_rate','spread'}, {'descend','descend'});
fprintf('\n===== equally spaced sets, best first (%d candidates) =====\n', height(C));
fprintf('%-10s %-9s %-32s %-8s %-8s\n', 'smallest', 'spacing', 'retention 1..4', 'spread', 'mono');
for i = 1:min(20, height(C))
    fprintf('%-10.3f %-9.3f %-32s %-8.1f %-8.2f\n', C.smallest_threshold(i), ...
        C.spacing(i), sprintf('%5.1f %5.1f %5.1f %5.1f', C.ret1(i), C.ret2(i), ...
        C.ret3(i), C.ret4(i)), C.spread(i), C.monotone_rate(i));
end
writetable(C, fullfile('outputs','Simple1D_EquallySpacedSets.csv'));
save(fullfile('outputs','Simple1D_EquallySpacedSets.mat'), 'grid','ret','KL', ...
    'kl_all','rollout_seeds','C');
fprintf('\nwrote outputs/Simple1D_EquallySpacedSets.csv\n');
R = C;
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
