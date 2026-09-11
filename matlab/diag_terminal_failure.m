function diag_terminal_failure()
%DIAG_TERMINAL_FAILURE  Why does one terminal projection fail?
%
% Two candidate explanations for a sequential-linearisation failure:
%   A) the guidance phase left that trajectory far inside the infeasible set,
%      so the linearisation radius is not enough to walk back out;
%   B) the point is wedged between two constraints, where the feasible set is
%      locally thin and the linearisation oscillates.
% They are told apart by the PRE-projection h: under (A) the failing point is
% much deeper in violation than the rest; under (B) it is not.
root='C:\Users\JieJi\BA\matlab'; cd(root);
S = load('outputs/SafeFlowNN_results.mat','R','net');
post = S.R.safeflow;
fprintf('archived run: terminal_failed = %d, rescued = %d\n', ...
    post.terminal_failed, post.terminal_rescued_by_fmincon);

% Guidance only, no terminal projection -> the state the projection starts from.
pre = safeflow_nn_rollout(S.net, 'safeflow', size(post.points,2), ...
    struct('terminal_filter', false, 'fmincon_fallback', false));

[Hpre, ~] = hmap(pre);  [Hpost, ~] = hmap(post);
n = size(Hpre,2);
worst_pre  = min(Hpre, [], 1);
worst_post = min(Hpost, [], 1);
failed = find(worst_post < -1e-8);
fprintf('\ntrajectories still violating after projection: %s\n', mat2str(failed));

fprintf('\n--- pre-projection depth (min h over the 65 points) ---\n');
fprintf('  all 100:      median %+.4f   p10 %+.4f   min %+.4f\n', ...
    median(worst_pre), prctile(worst_pre,10), min(worst_pre));
for f = failed(:)'
    r = mean(worst_pre <= worst_pre(f));
    fprintf('  FAILED #%d:    %+.4f   -> deeper than %.0f%% of trajectories\n', ...
        f, worst_pre(f), 100*(1-r));
end
fprintf('\n--- post-projection ---\n');
for f = failed(:)'
    fprintf('  FAILED #%d: min h %+.4e  (moved from %+.4e)\n', f, worst_post(f), worst_pre(f));
end
fprintf('  all others: min h %+.4e\n', min(worst_post(setdiff(1:n,failed))));

% How many constraints are active at the failing point?
fprintf('\n--- constraint geometry at the failing terminal points ---\n');
for f = failed(:)'
    P = squeeze(post.points(:,f,:));
    [~, kbad] = min(min_h_row(P, post));
    p = P(kbad,:)';
    ho = zeros(size(post.obstacle.centers,2),1);
    for j=1:numel(ho), ho(j) = obstacle_level_and_gradient(p, post.obstacle, j); end
    hb = zeros(2,1);
    for b=1:2, hb(b) = evaluate_track_implicit_field(post.geometry.implicit_fields, b, p); end
    fprintf('  traj %d, point %d: obstacle h = %s\n', f, kbad, mat2str(round(ho',5)));
    fprintf('                     boundary h = %s\n', mat2str(round(hb',5)));
    fprintf('    constraints within 0.01 of active: %d\n', sum([ho;hb] < 0.01));
end
end

function [H, hb] = hmap(r)
P = r.points; nk = size(P,1); n = size(P,2); H = inf(nk,n); hb = [];
for i=1:n, H(:,i) = min_h_row(squeeze(P(:,i,:)), r); end
end

function h = min_h_row(P, r)
nk = size(P,1); h = inf(nk,1);
for k=1:nk
    p = [P(k,1); P(k,2)]; v = inf;
    for j=1:size(r.obstacle.centers,2)
        v = min(v, obstacle_level_and_gradient(p, r.obstacle, j));
    end
    for b=1:2
        v = min(v, evaluate_track_implicit_field(r.geometry.implicit_fields, b, p));
    end
    h(k) = v;
end
end
