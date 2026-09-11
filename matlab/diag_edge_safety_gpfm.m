function diag_edge_safety_gpfm(n_samples)
%DIAG_EDGE_SAFETY_GPFM  The GPFM row of the edge-safety comparison.
%
% The NN baselines store their 65 physical points directly, so diag_edge_safety
% can read them.  GPFM's archive stores the third-level rollout in standardised
% local-increment segment form (101 x 1600 x 20 = time x 100 trajectories x 16
% segments x 5 points x 4 features), so the points have to be rebuilt the same
% way main_demo does: denormalise with third_segment_data_transform, convert the
% local increments to global coordinates, then expand the segments to a point
% list.
%
% That gives 80 points per trajectory (16 x 5).  The evaluator reports 65
% because it merges the 15 coincident junction points.  Both are checked here:
% the 79 raw segments include the junctions, where the two neighbouring
% estimates need not coincide, and the junction gaps are reported separately so
% a jump there is not silently counted as an obstacle crossing.

root = 'C:\Users\JieJi\BA\matlab'; cd(root);
if nargin < 1 || isempty(n_samples), n_samples = 20; end
car = char([36187 36710]);
d = fullfile('outputs', [car 'baseline'], 'cache');

S = load(fullfile(d, 'Racing_ThirdLevel_Rollout_SerialTest.mat'), ...
    'third_traj_path_10d', 'third_segment_data_transform');
cfg = get_config();
rng(cfg.random_seed);
[~, segment] = scenario_training_points(cfg, 65, cfg.n_train);
obst = configure_racing_obstacles(segment, cfg.obstacle);
geom = build_track_boundary_geometry(segment, ...
    struct_field_default(cfg.track_boundary, 'n_spline_points', 400));

nPS  = cfg.segment_points_per_segment;
final_state = squeeze(S.third_traj_path_10d(end, :, :));      % (1600 x 20)
fdim = size(final_state, 2) / nPS;
local = final_state .* S.third_segment_data_transform.std' + ...
        S.third_segment_data_transform.mean';
glob  = local_increment_rows_to_global(local, fdim, nPS);

n_seg  = size(glob,1) / 100;
assert(mod(n_seg,1) == 0, 'segment count does not divide by 100 trajectories');
pts = segment_points_to_point_list(glob, 100, n_seg, nPS);     % (80 x 100 x fdim)
XY  = pts(:,:,1:2);
[nPt, ng, ~] = size(XY);
fprintf('rebuilt %d points x %d trajectories (%d segments x %d points)\n\n', ...
    nPt, ng, n_seg, nPS);

lam = (1:n_samples)/(n_samples+1);
pt_ok  = true(nPt, ng);
seg_ok = true(nPt-1, ng);
worst  = inf;
gaps   = zeros(n_seg-1, ng);
for s = 1:ng
    for k = 1:nPt
        pt_ok(k,s) = h_of(XY(k,s,1), XY(k,s,2), obst, geom) >= -1e-8;
    end
    for k = 1:nPt-1
        a = [XY(k,s,1);   XY(k,s,2)];
        b = [XY(k+1,s,1); XY(k+1,s,2)];
        for j = 1:n_samples
            q = a + lam(j)*(b - a);
            hv = h_of(q(1), q(2), obst, geom);
            worst = min(worst, hv);
            if hv < -1e-8, seg_ok(k,s) = false; end
        end
    end
    for j = 1:n_seg-1                       % junction between segment j and j+1
        k = j*nPS;
        gaps(j,s) = norm([XY(k,s,1)-XY(k+1,s,1); XY(k,s,2)-XY(k+1,s,2)]);
    end
end

% Junctions are where two independent segment estimates meet; a crossing there
% is a stitching artefact, not the model steering through an obstacle.
is_junction = false(nPt-1,1);
is_junction((1:n_seg-1)*nPS) = true;

fprintf('%-22s %-12s %-12s %-13s %-13s %-12s\n', 'GPFM', 'point safe%', ...
    'traj safe%', 'edge safe%', 'traj edge-safe%', 'worst h');
fprintf('%-22s %-12.2f %-12.2f %-13.2f %-13.2f %+.5f\n', 'all 79 segments', ...
    100*mean(pt_ok(:)), 100*mean(all(pt_ok,1)), 100*mean(seg_ok(:)), ...
    100*mean(all(pt_ok,1) & all(seg_ok,1)), worst);
so = seg_ok(~is_junction,:);
fprintf('%-22s %-12s %-12s %-13.2f %-13.2f\n', 'excluding junctions', '', '', ...
    100*mean(so(:)), 100*mean(all(pt_ok,1) & all(so,1)));
fprintf('\njunction gap: median %.5f, p95 %.5f, max %.5f (normalised units)\n', ...
    median(gaps(:)), quantile(gaps(:),0.95), max(gaps(:)));

res = struct('point_ok', pt_ok, 'segment_ok', seg_ok, 'is_junction', is_junction, ...
    'junction_gaps', gaps, 'worst_edge_h', worst, 'n_samples', n_samples, ...
    'points_per_trajectory', nPt);
save(fullfile('outputs', 'Edge_Safety_GPFM.mat'), 'res');
fprintf('\nsaved outputs/Edge_Safety_GPFM.mat\n');
end

function hv = h_of(x, y, obst, geom)
p = [x; y];
hv = inf;
for jo = 1:size(obst.centers, 2)
    hv = min(hv, obstacle_level_and_gradient(p, obst, jo));
end
for bi = 1:2
    hv = min(hv, evaluate_track_implicit_field(geom.implicit_fields, bi, p));
end
end
