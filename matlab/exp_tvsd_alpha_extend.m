function exp_tvsd_alpha_extend(alphas, n_gen)
%EXP_TVSD_ALPHA_EXTEND  Push alpha for the linear-gamma TVSD baseline.
%
% The gamma-shape sweep showed the shape barely matters and alpha dominates:
% with linear gamma, KL went 3.95 -> 2.67 -> 1.24 for alpha = 1, 5, 10, and CS
% 0.735 -> 0.165 -> 0.156.  The mechanism is that the constraint only binds
% where alpha(s) < gamma_dot, i.e. s < gamma_dot/alpha, so a larger alpha
% narrows the affected band.  alpha*dt is only 0.0996 at alpha = 10, so there
% is a lot of headroom before the discrete-stability limit of 1.
%
% The QP is HARD.  The paper's Eq. 13 covers 'RoS diffuser else (12)', i.e. both
% RoSD and TVSD; only ReSD gets the relaxation variable of Eq. 14.  Earlier runs
% of this baseline left slack on, which made them not TVSD at all -- slack fired
% on 72-99%% of steps there.
%
% gamma is fixed at the linear schedule gamma(t) = -(1-t): constant closing
% rate, no free shape parameters, and gamma(0) = -1 covers every point since
% the obstacle level set bottoms out at -1.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(alphas), alphas = [10 20 50 100]; end
if nargin < 2 || isempty(n_gen), n_gen = 100; end
car = char([36187 36710]);
src = fullfile('outputs', [car 'safeflow_tbar05_paper']);
if ~exist(src, 'dir'), src = fullfile('outputs', [car 'safeflow']); end
N = load(fullfile(src, 'Racing_SafeFlow_NN_Net.mat'));
A = load(fullfile(src, 'Racing_SafeFlow_NN_Rollout.mat'));
af = fieldnames(A); arch = A.(af{1});
dt = 0.996/100;

fprintf('%-8s %-9s %-11s %-11s %-9s %-9s %-11s %-9s\n', 'alpha', 'alpha*dt', ...
    'obst safe%', 'bound safe%', 'KL', 'CS', 'obst min h', 'slack%');
M = cell(numel(alphas),1);
for i = 1:numel(alphas)
    al = alphas(i);
    o = struct('trajectory_seeds', arch.trajectory_seeds, 'u_per_stage', false, ...
        'fmincon_fallback', false, 'record_path', true, 'activation_time', 0, ...
        'terminal_filter', false, 'phi1_form', 'constant', 'phi1_switch_time', 0, ...
        'gamma_mode', 'linear', 'gamma_min', -1.0, ...
        'phi1_alpha', al, 'phi0', al, ...
        'slack_enabled', false);   % (13) is a hard QP; only ReSD uses (14)
    ws1 = warning('off','MATLAB:nearlySingularMatrix');
    ws2 = warning('off','MATLAB:singularMatrix');
    r = safeflow_nn_rollout(N.net, 'safeflow', n_gen, o);
    warning(ws1); warning(ws2);
    m = safeflow_run_metrics(r, N.net);
    [Ho, Hb] = final_h(r);
    m.obst_safe = 100*mean(Ho(:) >= -1e-8);
    m.bound_safe = 100*mean(Hb(:) >= -1e-8);
    m.obst_min = min(Ho(:));
    m.alpha = al;
    M{i} = m;
    fprintf('%-8g %-9.4f %-11.2f %-11.2f %-9.4f %-9.4f %-11.5f %-9.1f\n', ...
        al, al*dt, m.obst_safe, m.bound_safe, m.kl, m.cs, m.obst_min, ...
        100*m.slack_active/max(m.n_qp,1));
end
fprintf(['\nReference (same seeds, projection off):\n' ...
    '  SafeFlow           obst safe 96.06%%  KL 0.1145  CS 0.1980\n' ...
    '  conventional CBF   obst safe 96.02%%  KL 0.1147  CS 0.2072\n' ...
    '  FM (no guidance)   obst safe 93.54%%  KL 0.1236  CS 0.2101\n']);
S = struct('alphas', alphas, 'metrics', {M}, 'gamma', 'linear, gamma_min=-1', ...
    'n_gen', n_gen, 'dt', dt);
save(fullfile('outputs','TVSD_Alpha_Extend.mat'),'S','-v7.3');
fprintf('\nsaved outputs/TVSD_Alpha_Extend.mat\n');
end

function [Ho, Hb] = final_h(r)
P = r.points; [nPt, ng, ~] = size(P);
Ho = zeros(nPt, ng); Hb = zeros(nPt, ng);
for s = 1:ng
    for k = 1:nPt
        p = [P(k,s,1); P(k,s,2)];
        ho = inf; hb = inf;
        for jo = 1:size(r.obstacle.centers,2)
            ho = min(ho, obstacle_level_and_gradient(p, r.obstacle, jo));
        end
        for bi = 1:2
            hb = min(hb, evaluate_track_implicit_field(r.geometry.implicit_fields, bi, p));
        end
        Ho(k,s) = ho; Hb(k,s) = hb;
    end
end
end
