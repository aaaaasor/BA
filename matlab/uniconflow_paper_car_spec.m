function spec = uniconflow_paper_car_spec(opts)
%UNICONFLOW_PAPER_CAR_SPEC Publicly stated UniConFlow car specification.
% This is a reconstruction from the paper, not an official configuration.
% Unknown quantities are NaN so callers must choose them explicitly.
if nargin < 1, opts = struct(); end
gf = @(f,d) local_default(opts,f,d);

spec.variant = 'uniconflow_paper_reconstructed';
% The paper states a 100-step horizon, i.e. 101 states and a 604-D trajectory.
% n_states is exposed so the same code can run the 65-state layout the other
% racing baselines use; anything other than 101 is a deliberate departure from
% the published setup and is flagged in spec.horizon_matches_paper.
spec.n_states = gf('n_states', 101);
assert(isscalar(spec.n_states) && spec.n_states >= 2 && ...
    spec.n_states == round(spec.n_states), 'n_states must be an integer >= 2.');
spec.horizon = spec.n_states - 1;
spec.state_dim = 4;
spec.action_dim = 2;
spec.trajectory_dim = spec.n_states*spec.state_dim + spec.horizon*spec.action_dim;
spec.horizon_matches_paper = (spec.n_states == 101);
spec.state_order = {'x','y','theta','v'};
spec.action_order = {'delta','tau'};
spec.wheelbase = 2.7;
% RK4 sub-steps per control interval.  The action is held constant across
% them, so this buys back the integration accuracy a coarser control grid
% would otherwise lose.  Every rollout in the pipeline reads it from here, so
% a single setting keeps constraints, CEM, metrics and the QP filter
% consistent -- a missed one would show up only as a silent dynamics residual.
spec.n_sub = gf('n_sub', 1);
assert(isscalar(spec.n_sub) && spec.n_sub>=1 && spec.n_sub==round(spec.n_sub), ...
    'n_sub must be a positive integer.');
spec.dt = gf('dt', NaN); % not published for the car experiment

% Section VI-C1 states these bounds.  Equations (112b,c) instead imply
% +/-100 and +/-pi for the two action coordinates.  The text is selected by
% default, but the contradiction is retained in the metadata.
bound_source = char(gf('action_bound_source','section_text'));
switch bound_source
    case 'section_text'
        spec.action_lower = [-1; -35];
        spec.action_upper = [ 1;  35];
    case 'equation_112'
        spec.action_lower = [-100; -pi];
        spec.action_upper = [ 100;  pi];
    otherwise
        error('action_bound_source must be section_text or equation_112.');
end
spec.action_bound_source = bound_source;
spec.action_bound_conflict = true;
spec.action_bound_note = ['Section VI-C1 states delta in [-1,1] and tau in ', ...
    '[-35,35], while Eqs. (112b,c) imply bounds 100 and pi.'];

spec.forward_obstacles = [ ...
    348, 290, 40, 10,  30; ...
    331, 307, 40, 10,  30; ...
    340, 298, 30, 10, -45];
spec.dataset.forward_trajectories = 5000;
spec.dataset.reverse_trajectories = 5000;
spec.dataset.states_per_trajectory = 101;
spec.dataset.actions_per_trajectory = 100;
spec.unpublished = {'car sampling time dt', 'FM architecture/training settings', ...
    'QP weights', 'CEM population/elite/variance/iterations', 'Npad', 'Nrec'};
end

function v = local_default(s,f,d)
if isfield(s,f), v=s.(f); else, v=d; end
end
