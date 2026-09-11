function T = study_1d_kl_seed_sensitivity(n_seeds, seed0)
%STUDY_1D_KL_SEED_SENSITIVITY  Is the KL non-monotonicity in Table 7 real?
%
% The rollout seed only draws the 1000 initial samples; it changes neither the
% training data, the retained-point sets, nor the fitted GPs.  It is therefore
% a pure Monte-Carlo nuisance parameter, and averaging over it is legitimate
% in a way that picking a seed by its outcome is not.
%
% Models are rebuilt from the selected-point index sets stored in the archived
% sweep, so the posteriors are identical to the ones that produced Table 7.

if nargin < 1 || isempty(n_seeds), n_seeds = 40; end
if nargin < 2 || isempty(seed0),   seed0   = 1000; end
root = 'C:\Users\JieJi\BA\matlab'; cd(root);
d = fullfile('outputs', ['1d case' char([20840 23616 71 80])]);
S = load(fullfile(d, 'Simple1D_GlobalGP_Threshold_Sweep.mat'), 'sweep');
s = S.sweep;
H = load(fullfile(d, 'Manual_Hyperparameters.mat'), 'SigmaL', 'SigmaF', 'SigmaN');

X = s.X; Y = s.Y; tgt = s.target_points(:);
labels = [arrayfun(@(z) sprintf('thr %.2f', z), s.thresholds, 'UniformOutput', false), ...
          {'all data'}];
n_models = numel(labels);
retention = s.comparison.RetentionPercent;
kl_archived = s.comparison.KL;

models = cell(n_models, 1);
for m = 1:n_models
    idx = s.selected_indices{m}(:);
    models{m} = exact_gp(X(idx, :), Y(idx), H.SigmaN, H.SigmaF, H.SigmaL);
    fprintf('model %-10s  n = %5d  (retention %6.2f%%)\n', labels{m}, ...
        numel(idx), retention(m));
end

times = linspace(0, 1, s.cfg.n_rollout_steps + 1)';
n_roll = s.cfg.n_rollouts;

% Reproduce the archived KL first: same rollout seed, same code path.
rng(s.cfg.rollout_seed);
x0 = randn(n_roll, 1);
check = zeros(n_models, 1);
for m = 1:n_models
    ep = rollout_endpoints(models{m}, x0, times);
    check(m) = fixed_kde_kl_1d(tgt, ep);
end
fprintf('\nreproduction check against the archived sweep (rollout_seed = %d):\n', ...
    s.cfg.rollout_seed);
fprintf('%-10s %-14s %-14s %-12s\n', 'model', 'archived KL', 'recomputed', 'rel.diff');
for m = 1:n_models
    fprintf('%-10s %-14.6g %-14.6g %-12.2e\n', labels{m}, kl_archived(m), ...
        check(m), abs(check(m)-kl_archived(m))/max(abs(kl_archived(m)), eps));
end

seeds = seed0 + (0:n_seeds-1)';
KL = zeros(n_seeds, n_models);
tic;
for k = 1:n_seeds
    rng(seeds(k));
    x0 = randn(n_roll, 1);
    for m = 1:n_models
        ep = rollout_endpoints(models{m}, x0, times);
        KL(k, m) = fixed_kde_kl_1d(tgt, ep);
    end
    if mod(k, 5) == 0
        fprintf('  %d/%d seeds  (%.0f s elapsed)\n', k, n_seeds, toc);
    end
end

% Monotone = KL decreases as retention increases (models are ordered
% thr 0.20, 0.15, 0.10, 0.05, all data, i.e. retention ascending).
mono = all(diff(KL, 1, 2) < 0, 2);
fprintf('\n=========== KL over %d rollout seeds ===========\n', n_seeds);
fprintf('%-10s %-9s %-12s %-12s %-12s %-12s\n', 'model', 'retain%', ...
    'archived', 'mean', 'SD', 'median');
for m = 1:n_models
    fprintf('%-10s %-9.2f %-12.6g %-12.6g %-12.6g %-12.6g\n', labels{m}, ...
        retention(m), kl_archived(m), mean(KL(:,m)), std(KL(:,m)), median(KL(:,m)));
end
fprintf('\nmean KL monotone decreasing : %d\n', all(diff(mean(KL,1)) < 0));
fprintf('seeds with fully monotone KL: %d / %d (%.1f%%)\n', ...
    sum(mono), n_seeds, 100*mean(mono));
if any(mono)
    fprintf('  e.g. %s\n', mat2str(seeds(find(mono, min(8, sum(mono))))'));
end
% Which adjacent pair breaks it, and how often
fprintf('\nadjacent-pair inversion rate (KL(i) < KL(i+1) means correct order):\n');
for m = 1:n_models-1
    fprintf('  %-10s -> %-10s  correct in %5.1f%% of seeds\n', ...
        labels{m}, labels{m+1}, 100*mean(KL(:,m) > KL(:,m+1)));
end

T = array2table(KL, 'VariableNames', matlab.lang.makeValidName(labels));
T.seed = seeds; T.monotone = mono;
writetable(T, fullfile(d, 'Simple1D_KL_Seed_Sensitivity.csv'));
save(fullfile(d, 'Simple1D_KL_Seed_Sensitivity.mat'), 'KL', 'seeds', 'labels', ...
    'retention', 'kl_archived', 'mono');
fprintf('\nwrote %s\n', fullfile(d, 'Simple1D_KL_Seed_Sensitivity.csv'));
end

% =====================================================================
function m = exact_gp(Xs, Ys, sn, sf, sl)
Xt = Xs';                                   % kernel expects (dim x n)
K = kern(Xt, Xt, sf, sl) + sn^2 * eye(size(Xs, 1));
L = chol(K, 'lower');
m = struct('X', Xt, 'alpha', L' \ (L \ Ys), 'SigmaF', sf, 'SigmaL', sl);
end

function k = kern(X, Q, sf, sl)
dx = reshape(X, 2, [], 1) - reshape(Q, 2, 1, []);
scaled = dx ./ reshape(sl(:), 2, 1, 1);
k = sf^2 * exp(-0.5 * squeeze(sum(scaled.^2, 1)));
if size(Q, 2) == 1, k = reshape(k, [], 1); end
end

function ep = rollout_endpoints(m, x0, times)
% Vectorised over trajectories; identical arithmetic to plain_rollout_1d.
x = x0(:)';
for ti = 1:numel(times)-1
    t = times(ti); h = times(ti+1) - t;
    k1 = mu(m, t,       x);
    k2 = mu(m, t + h/2, x + h*k1/2);
    k3 = mu(m, t + h/2, x + h*k2/2);
    k4 = mu(m, t + h,   x + h*k3);
    x = x + h*(k1 + 2*k2 + 2*k3 + k4)/6;
end
ep = x(:);
end

function v = mu(m, t, x)
Q = [repmat(t, 1, numel(x)); x];
v = (m.alpha' * kern(m.X, Q, m.SigmaF, m.SigmaL));
end

function value = fixed_kde_kl_1d(reference, generated)
n = numel(reference); bw = max(1.06*std(reference)*n^(-1/5), 1e-3);
grid = linspace(min(reference)-6*bw, max(reference)+6*bw, 1024)';
p = kde_1d(grid, reference, bw); q = kde_1d(grid, generated, bw);
p = p/sum(p); q = q/sum(q); fl = 1e-300;
value = sum(p .* log(max(p, fl) ./ max(q, fl)));
end

function density = kde_1d(grid, samples, bw)
density = zeros(size(grid));
for i = 1:numel(samples)
    density = density + exp(-.5*((grid - samples(i))/bw).^2);
end
density = density / max(numel(samples), 1);
end
