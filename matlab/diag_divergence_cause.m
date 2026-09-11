function diag_divergence_cause()
%DIAG_DIVERGENCE_CAUSE  Why do some level-2 trajectories diverge once the
% GP-variance constraints are removed?
%
% Uses the cached 100-step no-variance ablation rollout. For every sample it
% replays the GP posterior along the stored path (no re-integration) and
% records, per step:
%     |x|      state norm
%     sigma2   GP posterior variance (sum over output dims)
%     |mu|     flow-field magnitude
%     h_min    joint softmin CBF value
% Divergence hypothesis: the state leaves the training support, k*->0, so
% sigma2 -> sigma_F^2 (prior) and |mu| -> 0; the flow can no longer transport
% the point and the geometric CBF has nothing left to act on.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
R = load(fullfile('outputs', ...
    'Racing_NoVariance_L2T0999_Omega05_SecondLevel_Rollout.mat'), ...
    'saved_segment_rollout_constraint','segment_traj_path_10d', ...
    'segment_rollout_times');
C = load(fullfile('outputs','Racing_NoVariance_L2T0999_Omega05_Config.mat'));
cfg = C.test_config;
cst = R.saved_segment_rollout_constraint;
times = R.segment_rollout_times;  path = R.segment_traj_path_10d;
n_t = numel(times); n_s = size(path,2);

M = load(fullfile(root, cfg.cache.second_level_model_path));
mf = fieldnames(M); mc = [];
for i=1:numel(mf)
    v = M.(mf{i});
    if isstruct(v) && isfield(v,'model'), mc = v; break; end
end
mc = strip_model_for_prediction(mc);
model = mc.model;

% prior variance bound: sum of signal variances over output dims
sf2 = 0;
for j = 1:numel(model.output_models)
    om = model.output_models{j};
    try, sf2 = sf2 + om.SigmaF^2; catch, end
end
fprintf('prior variance bound sigma_F^2 = %.4f  (output dims %d)\n', ...
    sf2, numel(model.output_models));

xn = zeros(n_t, n_s); s2 = zeros(n_t, n_s);
mun = zeros(n_t, n_s); hmin = zeros(n_t, n_s);
parfor s = 1:n_s
    c = cst;
    c.anchor_clf_target = cst.anchor_clf_targets(s,:)';
    c.track_boundary_reference_s_min = cst.track_boundary_reference_s_min_targets(s,:);
    c.track_boundary_reference_s_max = cst.track_boundary_reference_s_max_targets(s,:);
    xc = zeros(n_t,1); sc = zeros(n_t,1); mc2 = zeros(n_t,1); hc = zeros(n_t,1);
    for k = 1:n_t
        x = reshape(path(k,s,:),[],1);
        xc(k) = norm(x);
        st = predict_gp_stats(model, [times(k); x]);
        sc(k) = st.sigma2;  mc2(k) = norm(st.mu);
        st.x = x;
        oi = obstacle_cbf_info(st, c, times(k));
        bi = track_boundary_cbf_info(st, c, times(k));
        ji = joint_safety_softmin_info(oi, bi, st, c, times(k));
        if isempty(ji.h_values), hc(k) = NaN; else, hc(k) = min(ji.h_values); end
    end
    xn(:,s)=xc; s2(:,s)=sc; mun(:,s)=mc2; hmin(:,s)=hc;
end

hend = hmin(end,:)';
div  = find(hend < -1e-2);            % catastrophic
marg = find(hend < 0 & hend >= -1e-2); % marginal
ok   = find(hend >= 0);
fprintf('\nsamples: %d total | %d diverged (h_end < -1e-2) | %d marginal | %d safe\n', ...
    n_s, numel(div), numel(marg), numel(ok));

fprintf('\n--- terminal state, group medians ---\n');
fprintf('%-12s %-8s %-12s %-12s %-12s %-12s\n','group','n','|x|','sigma2','sigma2/sF2','|mu|');
grp = {'diverged',div; 'marginal',marg; 'safe',ok};
for g = 1:3
    ix = grp{g,2};
    if isempty(ix), continue; end
    fprintf('%-12s %-8d %-12.4g %-12.4g %-12.4g %-12.4g\n', grp{g,1}, numel(ix), ...
        median(xn(end,ix)), median(s2(end,ix)), median(s2(end,ix))/sf2, ...
        median(mun(end,ix)));
end

if ~isempty(div)
    fprintf('\n--- onset trace for the %d diverged samples (medians over group) ---\n', numel(div));
    fprintf('%-8s %-8s %-11s %-11s %-11s %-11s\n','k','t','|x|','sigma2','|mu|','h_min');
    ks = unique(round(linspace(1, n_t, 15)));
    for k = ks
        fprintf('%-8d %-8.4f %-11.4g %-11.4g %-11.4g %-11.4g\n', k, times(k), ...
            median(xn(k,div)), median(s2(k,div)), median(mun(k,div)), median(hmin(k,div)));
    end
    fprintf('\n--- same steps, safe group for reference ---\n');
    fprintf('%-8s %-8s %-11s %-11s %-11s %-11s\n','k','t','|x|','sigma2','|mu|','h_min');
    for k = ks
        fprintf('%-8d %-8.4f %-11.4g %-11.4g %-11.4g %-11.4g\n', k, times(k), ...
            median(xn(k,ok)), median(s2(k,ok)), median(mun(k,ok)), median(hmin(k,ok)));
    end
    fprintf('\n--- per-diverged-sample onset (first k where sigma2 > 50%% of sF2) ---\n');
    fprintf('%-8s %-10s %-10s %-10s %-12s %-12s\n','sample','k_sig','t_sig','k_hneg','t_hneg','h_end');
    for ii = 1:min(numel(div), 12)
        s = div(ii);
        ksig = find(s2(:,s) > 0.5*sf2, 1, 'first');
        khn  = find(hmin(:,s) < 0, 1, 'first');
        fprintf('%-8d %-10s %-10s %-10s %-12s %-12.4g\n', s, ...
            num2str_or(ksig), num2str_t(ksig, times), ...
            num2str_or(khn), num2str_t(khn, times), hend(s));
    end
end
save(fullfile('outputs','Diag_Divergence_Cause.mat'), 'xn','s2','mun','hmin', ...
    'times','sf2','div','marg','ok');
fprintf('\nsaved outputs/Diag_Divergence_Cause.mat\n');
end

function s = num2str_or(k)
if isempty(k), s = '-'; else, s = num2str(k); end
end
function s = num2str_t(k, times)
if isempty(k), s = '-'; else, s = sprintf('%.4f', times(k)); end
end
