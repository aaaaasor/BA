function T = evaluate_1d_threshold_set_full(thresholds, rollout_seeds, n_roll, append_seed)
%EVALUATE_1D_THRESHOLD_SET_FULL  Every table-7 metric for a threshold set.
%
% The equally spaced search only scored KL.  The thesis table also reports
% Time, mean/terminal posterior variance and their prior-normalised versions,
% and those have to be monotone too -- checking KL alone is not enough.
%
% Reports each metric averaged over several rollout seeds, with the SEM, and
% states which metrics are monotone in retention.  Timing is measured on a
% single warmed-up pass so it is comparable across models, but wall-clock time
% is noisy: it is reported for completeness, not as evidence.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(thresholds),    thresholds = [0.14 0.11 0.08 0.05]; end
if nargin < 2 || isempty(rollout_seeds), rollout_seeds = 2000:2019; end
if nargin < 3 || isempty(n_roll),        n_roll = 1000; end
if nargin < 4 || isempty(append_seed),   append_seed = 345; end

stem = ['1d case' char([20840 23616 71 80 35757 32451 38408 20540 23454 39564])];
dir25 = fullfile('outputs', [stem '_25x40']);
B = load(fullfile(dir25, 'Training_Data_and_Seeds.mat'), 'source_points', 'target_points');
H = load(fullfile(dir25, 'Manual_Hyperparameters.mat'), 'SigmaL', 'SigmaF');
sl = H.SigmaL; sf = H.SigmaF; sn = 0.25;
prior = sf^2;
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

thr = sort(thresholds, 'descend');       % retention ascending
nmod = numel(thr) + 1;
labels = [arrayfun(@(z) sprintf('%.3f', z), thr, 'UniformOutput', false), {'all'}];
mods = cell(nmod,1); ret = zeros(nmod,1);
fprintf('fitting %d models (append_seed %d)\n', nmod, append_seed);
for i = 1:nmod
    if i <= numel(thr)
        idx = select_points(X, sn, sf, sl, thr(i));
    else
        idx = (1:size(X,1))';
    end
    ret(i) = 100*numel(idx)/size(X,1);
    mods{i} = exact_gp(X(idx,:), Y(idx), sn, sf, sl);
    fprintf('  %-8s %6.2f%%  (%d points)\n', labels{i}, ret(i), numel(idx));
end

ns = numel(rollout_seeds);
KL = zeros(ns,nmod); MV = zeros(ns,nmod); TV = zeros(ns,nmod);
fprintf('\nrolling out %d seeds x %d trajectories\n', ns, n_roll);
for k = 1:ns
    rng(rollout_seeds(k)); x0 = randn(n_roll,1);
    for i = 1:nmod
        [ep, V] = rollout_with_variance(mods{i}, x0, times_roll);
        KL(k,i) = fixed_kde_kl_1d(target, ep);
        % one number per trajectory, then average -- same convention as the
        % table's point estimate
        MV(k,i) = mean(mean(V, 1));
        TV(k,i) = mean(V(end,:));
    end
    if mod(k,5)==0, fprintf('  %d/%d\n', k, ns); end
end

% Timing: warm up once, then time one full rollout per model.
tsec = zeros(nmod,1);
rng(rollout_seeds(1)); x0 = randn(n_roll,1);
for i = 1:nmod, rollout_with_variance(mods{i}, x0(1:50), times_roll); end
for i = 1:nmod
    tt = tic; rollout_with_variance(mods{i}, x0, times_roll); tsec(i) = toc(tt)/n_roll;
end

mv = mean(MV,1)'; tv = mean(TV,1)'; kl = mean(KL,1)';
mvn = mv/prior; tvn = tv/prior;
sem = @(A) (std(A,0,1)/sqrt(size(A,1)))';
sMV = sem(MV); sTV = sem(TV); sKL = sem(KL);

fprintf('\n============ metrics, mean over %d rollout seeds ============\n', ns);
fprintf('%-8s %-9s %-9s %-18s %-18s %-18s %-18s\n', 'thresh', 'retain%', ...
    'Time(s)', 'mean s2', 'term s2', 'mean s~2', 'term s~2');
for i = 1:nmod
    fprintf('%-8s %-9.2f %-9.4f %8.5f+-%-8.5f %8.5f+-%-8.5f %8.5f+-%-8.5f %8.5f+-%-8.5f\n', ...
        labels{i}, ret(i), tsec(i), mv(i), sMV(i), tv(i), sTV(i), ...
        mvn(i), sMV(i)/prior, tvn(i), sTV(i)/prior);
end
fprintf('\n%-8s %-9s\n', 'thresh', 'KL');
for i = 1:nmod
    fprintf('%-8s %8.5f+-%-8.5f\n', labels{i}, kl(i), sKL(i));
end

names = {'Time', 'mean_sigma2', 'term_sigma2', 'mean_sigma2_norm', ...
    'term_sigma2_norm', 'KL'};
cols  = {tsec, mv, tv, mvn, tvn, kl};
fprintf('\n============ monotonicity (should DECREASE with retention) ============\n');
for j = 1:numel(cols)
    v = cols{j}; d = diff(v);
    ok = all(d < 0);
    if strcmp(names{j}, 'Time'), ok = all(d > 0); end   % time should INCREASE
    fprintf('  %-20s %s', names{j}, ternary(ok, 'monotone', 'NOT monotone'));
    if ~ok
        bad = find(xor(d < 0, strcmp(names{j},'Time')) == 0);
        for b = bad(:)'
            fprintf('  [break %s -> %s: %.5g -> %.5g]', labels{b}, labels{b+1}, v(b), v(b+1));
        end
    end
    fprintf('\n');
end
% Per-seed monotonicity rate for the sampled metrics
fprintf('\nper-seed monotonicity rate (%d seeds):\n', ns);
fprintf('  KL          %5.1f%%\n', 100*mean(all(diff(KL,1,2)<0,2)));
fprintf('  mean s2     %5.1f%%\n', 100*mean(all(diff(MV,1,2)<0,2)));
fprintf('  term s2     %5.1f%%\n', 100*mean(all(diff(TV,1,2)<0,2)));

T = table(string(labels(:)), ret, tsec, mv, sMV, tv, sTV, ...
    mvn, sMV/prior, tvn, sTV/prior, kl, sKL, ...
    'VariableNames', {'model','retention_percent','time_s', ...
    'mean_sigma2','mean_sigma2_sem','term_sigma2','term_sigma2_sem', ...
    'mean_sigma2_norm','mean_sigma2_norm_sem','term_sigma2_norm', ...
    'term_sigma2_norm_sem','KL','KL_sem'});
out = fullfile('outputs', 'Simple1D_NewThresholdSet_Metrics.csv');
writetable(T, out);
fprintf('\nwrote %s\n', out);
end

% =====================================================================
function v = ternary(c,a,b), if c, v=a; else, v=b; end, end

function idx = select_points(X, sn, sf, sl, threshold)
n = size(X,1); Xt = X'; idx = zeros(n,1); c = 0; L = [];
for i = 1:n
    if c == 0
        take = true;
    else
        v = L \ kern(Xt(:, idx(1:c)), Xt(:, i), sf, sl);
        take = sqrt(max(sf^2 - (v'*v), 0)) > threshold;
    end
    if take
        c = c + 1; idx(c) = i;
        L = chol_append(L, Xt(:, idx(1:c)), sn, sf, sl);
    end
end
idx = idx(1:c);
end

function L = chol_append(L, Xsub, sn, sf, sl)
c = size(Xsub,2);
if c == 1, L = sqrt(sf^2 + sn^2); return; end
l = L \ kern(Xsub(:,1:c-1), Xsub(:,c), sf, sl);
d = sqrt(max(sf^2 + sn^2 - (l'*l), eps));
L = [L, zeros(c-1,1); l', d];
end

function m = exact_gp(Xs, Ys, sn, sf, sl)
Xt = Xs';
K = kern(Xt, Xt, sf, sl) + sn^2*eye(size(Xs,1));
L = chol(K, 'lower');
m = struct('X', Xt, 'L', L, 'alpha', L'\(L\Ys), 'SigmaF', sf, 'SigmaL', sl);
end

function k = kern(X, Q, sf, sl)
dx = reshape(X,2,[],1) - reshape(Q,2,1,[]);
scaled = dx ./ reshape(sl(:),2,1,1);
k = sf^2*exp(-0.5*squeeze(sum(scaled.^2,1)));
if size(Q,2) == 1, k = reshape(k,[],1); end
end

function [ep, V] = rollout_with_variance(m, x0, times)
x = x0(:)';
nt = numel(times); V = zeros(nt, numel(x));
V(1,:) = post_var(m, times(1), x);
for ti = 1:nt-1
    t = times(ti); h = times(ti+1)-t;
    k1 = mu(m,t,x); k2 = mu(m,t+h/2,x+h*k1/2);
    k3 = mu(m,t+h/2,x+h*k2/2); k4 = mu(m,t+h,x+h*k3);
    x = x + h*(k1+2*k2+2*k3+k4)/6;
    V(ti+1,:) = post_var(m, times(ti+1), x);
end
ep = x(:);
end

function v = mu(m, t, x)
v = m.alpha' * kern(m.X, [repmat(t,1,numel(x)); x], m.SigmaF, m.SigmaL);
end

function s2 = post_var(m, t, x)
Ks = kern(m.X, [repmat(t,1,numel(x)); x], m.SigmaF, m.SigmaL);
W = m.L \ Ks;
s2 = max(m.SigmaF^2 - sum(W.^2, 1), 0);
end

function target = sample_target_1d(n)
component = rand(n,1) > 0.5;
target = (-2.0+0.45*randn(n,1)).*(~component) + (2.0+0.55*randn(n,1)).*component;
end

function value = fixed_kde_kl_1d(reference, generated)
n = numel(reference); bw = max(1.06*std(reference)*n^(-1/5), 1e-3);
grid = linspace(min(reference)-6*bw, max(reference)+6*bw, 1024)';
p = kde_1d(grid, reference, bw); q = kde_1d(grid, generated, bw);
p = p/sum(p); q = q/sum(q); fl = 1e-300;
value = sum(p .* log(max(p,fl)./max(q,fl)));
end

function d = kde_1d(grid, samples, bw)
d = zeros(size(grid));
for i = 1:numel(samples), d = d + exp(-.5*((grid-samples(i))/bw).^2); end
d = d/max(numel(samples),1);
end
