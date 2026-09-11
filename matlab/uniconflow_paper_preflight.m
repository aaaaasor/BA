function report = uniconflow_paper_preflight(net, states, actions, opts)
%UNICONFLOW_PAPER_PREFLIGHT Check whether a run can reproduce the car setup.
if nargin < 2, states=[]; end
if nargin < 3, actions=[]; end
if nargin < 4, opts=struct(); end
spec=uniconflow_paper_car_spec(opts);
issues={}; warnings={};

if nargin<1 || isempty(net)
    issues{end+1}='Missing a joint state-action FM network.';
else
    if ~isfield(net,'mu_d') || numel(net.mu_d)~=spec.trajectory_dim
        got=NaN; if isfield(net,'mu_d'), got=numel(net.mu_d); end
        issues{end+1}=sprintf('FM dimension is %g; the paper car trajectory requires 604.',got);
    end
    if isfield(net,'feature_order') && ~isequal(net.feature_order, ...
            {'x','y','theta','v','delta','tau'})
        warnings{end+1}='Network feature_order is not the paper car state-action order.';
    end
end

if isempty(states) || isempty(actions)
    issues{end+1}='Missing joint OCP state-action demonstrations.';
else
    if size(states,1)~=4 || size(states,2)~=101
        issues{end+1}=sprintf('states are %s; expected 4x101xN.',mat2str(size(states)));
    end
    if size(actions,1)~=2 || size(actions,2)~=100
        issues{end+1}=sprintf('actions are %s; expected 2x100xN.',mat2str(size(actions)));
    end
    if size(states,3)~=size(actions,3)
        issues{end+1}='State/action demonstration counts differ.';
    elseif size(states,3)~=10000
        warnings{end+1}=sprintf('Dataset has %d trajectories; the paper reports 10000.',size(states,3));
    end
end
if ~isfinite(spec.dt)
    issues{end+1}='Car sampling time dt is unpublished and must be supplied explicitly.';
end
warnings{end+1}=spec.action_bound_note;
warnings{end+1}='Official project page still lists code as coming soon.';

report=struct('ready',isempty(issues),'variant',spec.variant,'spec',spec, ...
    'issues',{issues},'warnings',{warnings});
if nargout==0
    fprintf('UniConFlow paper reconstruction ready: %s\n',string(report.ready));
    for i=1:numel(issues), fprintf('  BLOCKER: %s\n',issues{i}); end
    for i=1:numel(warnings), fprintf('  WARNING: %s\n',warnings{i}); end
end
end
