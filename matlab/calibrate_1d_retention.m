function R = calibrate_1d_retention(thresholds, append_seed)
%CALIBRATE_1D_RETENTION  Map training threshold -> training-point retention.
%
% Retention is strongly non-linear in the threshold (2.7% at 0.80 but 75.7% at
% 0.05), so a set of evenly spaced thresholds gives badly bunched retentions.
% The thesis table wants the opposite: retentions spread evenly from near 0 to
% near 100.  This calibrates the mapping so the thresholds can be solved for.
%
% Selection only -- no GP rollouts -- so this is much cheaper than a full sweep.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(thresholds)
    thresholds = [0.30 0.20 0.15 0.12 0.10 0.09 0.08 0.07 0.065 0.06 ...
                  0.055 0.05 0.045 0.04 0.035 0.03];
end
if nargin < 2 || isempty(append_seed), append_seed = 345; end

stem = ['1d case' char([20840 23616 71 80 35757 32451 38408 20540 23454 39564])];
dir25 = fullfile('outputs', [stem '_25x40']);
B = load(fullfile(dir25, 'Training_Data_and_Seeds.mat'), 'source_points', 'target_points');
H = load(fullfile(dir25, 'Manual_Hyperparameters.mat'), 'SigmaL', 'SigmaF');
sl = H.SigmaL; sf = H.SigmaF; sn = 0.25;
times_train = linspace(0, 1, 40)';

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

fprintf('retention calibration, append_seed %d, %d candidate points\n', ...
    append_seed, size(X,1));
fprintf('%-11s %-9s %-11s\n', 'threshold', 'points', 'retention%');
n = numel(thresholds); pts = zeros(n,1); ret = zeros(n,1);
t0 = tic;
for i = 1:n
    idx = select_points(X, sn, sf, sl, thresholds(i));
    pts(i) = numel(idx); ret(i) = 100*numel(idx)/size(X,1);
    fprintf('%-11.3f %-9d %-11.2f\n', thresholds(i), pts(i), ret(i));
end
fprintf('elapsed %.0f s\n', toc(t0));

R = table(thresholds(:), pts, ret, 'VariableNames', ...
    {'threshold','points','retention_percent'});
writetable(R, fullfile('outputs', ...
    sprintf('Simple1D_RetentionCalibration_seed%d.csv', append_seed)));

% Solve for the thresholds that land on evenly spaced retentions.
targets = [20 40 60 80];
[r_sorted, o] = sort(ret); t_sorted = thresholds(o);
fprintf('\nthresholds giving evenly spaced retention (interpolated):\n');
fprintf('%-14s %-12s\n', 'target ret%', 'threshold');
for tg = targets
    if tg < min(r_sorted) || tg > max(r_sorted)
        fprintf('%-14.0f %-12s  (outside the calibrated range)\n', tg, '--');
    else
        fprintf('%-14.0f %-12.4f\n', tg, interp1(r_sorted, t_sorted, tg));
    end
end
fprintf('\nwrote outputs/Simple1D_RetentionCalibration_seed%d.csv\n', append_seed);
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

function k = kern(X, Q, sf, sl)
dx = reshape(X, 2, [], 1) - reshape(Q, 2, 1, []);
scaled = dx ./ reshape(sl(:), 2, 1, 1);
k = sf^2 * exp(-0.5 * squeeze(sum(scaled .^ 2, 1)));
if size(Q, 2) == 1, k = reshape(k, [], 1); end
end

function target = sample_target_1d(n)
component = rand(n, 1) > 0.5;
target = (-2.0 + 0.45*randn(n,1)) .* (~component) + ...
         ( 2.0 + 0.55*randn(n,1)) .* component;
end
