function diag_edge_safety(n_samples)
%DIAG_EDGE_SAFETY  Do the SEGMENTS between generated points stay safe?
%
% Every Safety number in the comparison table is pointwise.  The evaluators say
% so explicitly -- "points inside track and outside physical obstacles; edges
% not checked" -- so a trajectory whose 65 sampled points all avoid the
% obstacles can still have the straight lines between them cut clean through.
% Nothing in the table would reveal that.
%
% This resamples each of the 64 segments of every generated trajectory and
% reports how often the interior of a segment is unsafe, which is the quantity
% the pointwise metric cannot see.  It changes no archived number; it measures
% something the archived numbers were never about.
%
% Linear interpolation between consecutive points is the right thing to check
% here: it is how the trajectories are drawn, and for a path that a vehicle
% follows it is the least favourable reading -- a smoother interpolant would
% cut corners differently, so treat this as an upper bound on edge violation.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_samples), n_samples = 20; end
car = char([36187 36710]);  trk = char([36187 36947]);

rows = { 'FM',       [car 'fm'],       'Racing_FM_NN'; ...
         'SafeFlow', [car 'safeflow'], 'Racing_SafeFlow_NN'; ...
         'RoSD',     [trk 'RoSD'],     'RoSD_Mech'; ...
         'ReSD',     [trk 'ReSD'],     'ReSD_Mech'; ...
         'TVSD',     [trk 'TVSD'],     'TVSD_Mech'; ...
         'PCFM',     [trk 'PCFM'],     'PCFM_Mech' };

fprintf(['interior samples per segment: %d (endpoints excluded, since those ' ...
    'are\nthe points the pointwise metric already covers)\n\n'], n_samples);
fprintf('%-10s %-12s %-12s %-13s %-13s %-12s\n', 'method', 'point safe%', ...
    'traj safe%', 'edge safe%', 'traj edge-safe%', 'worst h');
res = struct();
for i = 1:size(rows,1)
    f = fullfile('outputs', rows{i,2}, [rows{i,3} '_Rollout.mat']);
    if ~isfile(f), fprintf('%-10s (no archive)\n', rows{i,1}); continue; end
    A = load(f);  r = A.rollout;
    d = edge_stats(r, n_samples);
    res.(rows{i,1}) = d;
    fprintf('%-10s %-12.2f %-12.2f %-13.2f %-13.2f %+.5f\n', rows{i,1}, ...
        100*d.point_safe, 100*d.traj_point_safe, 100*d.edge_safe, ...
        100*d.traj_edge_safe, d.worst_edge_h);
end

fprintf(['\npoint safe%%      : share of the 65 sampled points that are safe\n' ...
    'traj safe%%       : share of trajectories whose 65 points are all safe ' ...
    '(the table''s Safety)\n' ...
    'edge safe%%       : share of segment interiors that are safe\n' ...
    'traj edge-safe%%  : share of trajectories whose points AND segment ' ...
    'interiors are all safe\n']);
save(fullfile('outputs', 'Edge_Safety_Comparison.mat'), 'res', 'n_samples');
fprintf('\nsaved outputs/Edge_Safety_Comparison.mat\n');
end

% =====================================================================
function d = edge_stats(r, ns)
P = r.points;  [nPt, ng, ~] = size(P);
lam = (1:ns)/(ns+1);                       % interior only

pt_ok  = true(nPt, ng);
seg_ok = true(nPt-1, ng);
worst  = inf;
for s = 1:ng
    for k = 1:nPt
        pt_ok(k,s) = h_of(r, [P(k,s,1); P(k,s,2)]) >= -1e-8;
    end
    for k = 1:nPt-1
        a = [P(k,s,1);   P(k,s,2)];
        b = [P(k+1,s,1); P(k+1,s,2)];
        for j = 1:ns
            q = a + lam(j)*(b - a);
            hv = h_of(r, q);
            worst = min(worst, hv);
            if hv < -1e-8, seg_ok(k,s) = false; end
        end
    end
end
d = struct('point_safe', mean(pt_ok(:)), ...
    'traj_point_safe', mean(all(pt_ok,1)), ...
    'edge_safe', mean(seg_ok(:)), ...
    'traj_edge_safe', mean(all(pt_ok,1) & all(seg_ok,1)), ...
    'worst_edge_h', worst, 'point_ok', pt_ok, 'segment_ok', seg_ok);
end

function hv = h_of(r, p)
hv = inf;
for jo = 1:size(r.obstacle.centers, 2)
    hv = min(hv, obstacle_level_and_gradient(p, r.obstacle, jo));
end
for bi = 1:2
    hv = min(hv, evaluate_track_implicit_field(r.geometry.implicit_fields, bi, p));
end
end
