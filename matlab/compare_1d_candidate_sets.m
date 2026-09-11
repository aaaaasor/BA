function T = compare_1d_candidate_sets(sets, rollout_seeds, n_roll, append_seed)
%COMPARE_1D_CANDIDATE_SETS  All six table metrics, for every candidate set.
%
% The equally spaced enumeration scored KL only.  The winning set
% (0.14/0.11/0.08/0.05) then turned out to have one weak step: mean sigma^2
% between 45% and 76% retention differs by 0.00076 against an SEM of 0.0016,
% so that step is inside the noise and only 80% of seeds order it correctly.
%
% This evaluates every KL-monotone candidate on all six metrics, so a set that
% is clean on all of them can be picked if one exists.  Distinct thresholds are
% fitted once and shared between sets, which is roughly twice as fast as
% running the sets separately.
%
% Monotone means: Time increases with retention, everything else decreases.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(sets)
    sets = { [0.14 0.11 0.08 0.05], ...      % spread 53.6, already tested
             [0.11 0.09 0.07 0.05], ...      % spread 45.0
             [0.13 0.11 0.09 0.07], ...      % spread 28.3
             [0.15 0.13 0.11 0.09], ...      % spread 19.2
             [0.11 0.10 0.09 0.08], ...      % spread 14.3
             [0.12 0.11 0.10 0.09] };        % spread 12.8
end
if nargin < 2 || isempty(rollout_seeds), rollout_seeds = 2000:2019; end
if nargin < 3 || isempty(n_roll),        n_roll = 1000; end
if nargin < 4 || isempty(append_seed),   append_seed = 345; end

stem = ['1d case' char([20840 23616 71 80 35757 32451 38408 20540 23454 39564])];
dir25 = fullfile('outputs', [stem '_25x40']);
B = load(fullfile(dir25, 'Training_Data_and_Seeds.mat'), 'source_points', 'target_points');
H = load(fullfile(dir25, 'Manual_Hyperparameters.mat'), 'SigmaL', 'SigmaF');
sl = H.SigmaL; sf = H.SigmaF; sn = 0.25; prior = sf^2;
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

thr_all = unique([sets{:}]);
nthr = numel(thr_all);
fprintf('fitting %d distinct thresholds\n', nthr);
mods = cell(nthr,1); ret = zeros(nthr,1);
for i = 1:nthr
    idx = select_points(X, sn, sf, sl, thr_all(i));
    ret(i) = 100*numel(idx)/size(X,1);
    mods{i} = exact_gp(X(idx,:), Y(idx), sn, sf, sl);
    fprintf('  %.3f -> %6.2f%% (%d pts)\n', thr_all(i), ret(i), numel(idx));
end
all_mod = exact_gp(X, Y, sn, sf, sl);

ns = numel(rollout_seeds);
KL = zeros(ns,nthr); MV = zeros(ns,nthr); TV = zeros(ns,nthr);
klA = zeros(ns,1); mvA = zeros(ns,1); tvA = zeros(ns,1);
fprintf('\nrolling out %d seeds x %d trajectories over %d models\n', ns, n_roll, nthr+1);
t0 = tic;
for k = 1:ns
    rng(rollout_seeds(k)); x0 = randn(n_roll,1);
    for i = 1:nthr
        [ep, V] = rollout_with_variance(mods{i}, x0, times_roll);
        KL(k,i) = fixed_kde_kl_1d(target, ep);
        MV(k,i) = mean(mean(V,1));  TV(k,i) = mean(V(end,:));
    end
    [ep, V] = rollout_with_variance(all_mod, x0, times_roll);
    klA(k) = fixed_kde_kl_1d(target, ep);
    mvA(k) = mean(mean(V,1)); tvA(k) = mean(V(end,:));
    if mod(k,5)==0, fprintf('  %d/%d (%.0f s)\n', k, ns, toc(t0)); end
end

% Timing per model, warmed up.
tsec = zeros(nthr,1);
rng(rollout_seeds(1)); x0 = randn(n_roll,1);
for i = 1:nthr, rollout_with_variance(mods{i}, x0(1:50), times_roll); end
for i = 1:nthr
    tt = tic; rollout_with_variance(mods{i}, x0, times_roll); tsec(i) = toc(tt)/n_roll;
end
rollout_with_variance(all_mod, x0(1:50), times_roll);
tt = tic; rollout_with_variance(all_mod, x0, times_roll); tsecA = toc(tt)/n_roll;

rows = {};
fprintf('\n================= candidate sets =================\n');
for si = 1:numel(sets)
    thr = sort(sets{si}, 'descend');
    ii = arrayfun(@(t) find(abs(thr_all - t) < 1e-12, 1), thr);
    r  = [ret(ii); 100];
    tS = [tsec(ii); tsecA];
    mvM = [MV(:,ii), mvA]; tvM = [TV(:,ii), tvA]; klM = [KL(:,ii), klA];
    mv = mean(mvM,1)'; tv = mean(tvM,1)'; kl = mean(klM,1)';
    sMV = (std(mvM,0,1)/sqrt(ns))'; sTV = (std(tvM,0,1)/sqrt(ns))';
    sKL = (std(klM,0,1)/sqrt(ns))';

    ok_t  = all(diff(tS) > 0);
    ok_mv = all(diff(mv) < 0);   ok_tv = all(diff(tv) < 0);
    ok_kl = all(diff(kl) < 0);
    % A step is "clean" when the gap exceeds the combined SEM of its two ends.
    clean_mv = all(abs(diff(mv)) > (sMV(1:end-1) + sMV(2:end)));
    clean_tv = all(abs(diff(tv)) > (sTV(1:end-1) + sTV(2:end)));
    clean_kl = all(abs(diff(kl)) > (sKL(1:end-1) + sKL(2:end)));

    fprintf('\nset %s   retention %s   spread %.1f\n', mat2str(thr), ...
        mat2str(round(r(1:4),2)), r(4)-r(1));
    fprintf('  monotone:  Time %d  mean s2 %d  term s2 %d  KL %d\n', ...
        ok_t, ok_mv, ok_tv, ok_kl);
    fprintf('  every step beyond noise:  mean s2 %d  term s2 %d  KL %d\n', ...
        clean_mv, clean_tv, clean_kl);
    fprintf('  per-seed monotone rate:  mean s2 %5.1f%%  term s2 %5.1f%%  KL %5.1f%%\n', ...
        100*mean(all(diff(mvM,1,2)<0,2)), 100*mean(all(diff(tvM,1,2)<0,2)), ...
        100*mean(all(diff(klM,1,2)<0,2)));
    if ~clean_mv
        d = abs(diff(mv)); s = sMV(1:end-1)+sMV(2:end);
        b = find(d <= s);
        for j = b(:)'
            fprintf('    weak mean s2 step %.2f%% -> %.2f%%: gap %.5f vs SEM sum %.5f\n', ...
                r(j), r(j+1), d(j), s(j));
        end
    end
    rows{end+1} = struct('set', mat2str(thr), 'retention', r(1:4)', ...
        'spread', r(4)-r(1), 'mean_s2', mv, 'mean_s2_sem', sMV, ...
        'term_s2', tv, 'term_s2_sem', sTV, 'KL', kl, 'KL_sem', sKL, ...
        'time', tS, 'all_monotone', ok_t && ok_mv && ok_tv && ok_kl, ...
        'all_clean', clean_mv && clean_tv && clean_kl); %#ok<AGROW>
end

fprintf('\n================= summary =================\n');
fprintf('%-26s %-8s %-12s %-10s\n', 'set', 'spread', 'monotone', 'all steps clean');
for i = 1:numel(rows)
    fprintf('%-26s %-8.1f %-12d %-10d\n', rows{i}.set, rows{i}.spread, ...
        rows{i}.all_monotone, rows{i}.all_clean);
end
save(fullfile('outputs','Simple1D_CandidateSets_FullMetrics.mat'), 'rows', ...
    'sets', 'thr_all', 'ret', 'KL', 'MV', 'TV', 'klA', 'mvA', 'tvA', ...
    'tsec', 'tsecA', 'rollout_seeds', 'prior');
fprintf('\nwrote outputs/Simple1D_CandidateSets_FullMetrics.mat\n');
T = rows;
end

% =====================================================================
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
x = x0(:)'; nt = numel(times); V = zeros(nt, numel(x));
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
W = m.L \ kern(m.X, [repmat(t,1,numel(x)); x], m.SigmaF, m.SigmaL);
s2 = max(m.SigmaF^2 - sum(W.^2,1), 0);
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
