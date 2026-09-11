function out = uniconflow_paper_full_rollout(net, scur, opts)
%UNICONFLOW_PAPER_FULL_ROLLOUT Reconstructed paper car inference pipeline.
if nargin<3,opts=struct();end
if isfield(opts,'stage1'),o1=opts.stage1;else,o1=opts;end
if isfield(opts,'stage2'),o2=opts.stage2;else,o2=opts;end
if isfield(opts,'dt')
    if ~isfield(o1,'dt'),o1.dt=opts.dt;end
end
if isfield(o1,'state_constraint') && ~isfield(o2,'state_constraint')
    o2.state_constraint=o1.state_constraint;
end
r1=uniconflow_paper_guided_sample(net,scur,o1);
out=uniconflow_paper_cem_refine(r1,o2);
out.options.requested=opts;
end
