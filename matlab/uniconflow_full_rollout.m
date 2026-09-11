function out = uniconflow_full_rollout(net, n_gen, opts)
%UNICONFLOW_FULL_ROLLOUT  End-to-end adapted UniConFlow inference.
%   Stage 1 performs PTZF-guided FM sampling.  Stage 2 applies the paper's
%   frozen/violation/recovery/global CEM refinement.  Options may be supplied
%   in opts.stage1 and opts.stage2; flat options are accepted for backwards
%   compatibility and are passed to both stages.
if nargin < 2 || isempty(n_gen), n_gen=100; end
if nargin < 3, opts=struct(); end
if isfield(opts,'stage1'), o1=opts.stage1; else, o1=opts; end
if isfield(opts,'stage2'), o2=opts.stage2; else, o2=opts; end
% A top-level seed table belongs to Stage 1 even when nested options are used.
if isfield(opts,'trajectory_seeds') && ~isfield(o1,'trajectory_seeds')
    o1.trajectory_seeds=opts.trajectory_seeds;
end
r1=uniconflow_nn_rollout(net,n_gen,o1);
out=uniconflow_cem_refine(r1,o2);
out.options.pipeline='uniconflow_full_rollout';
out.options.requested_options=opts;
end
