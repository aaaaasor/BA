function out = uniconflow_paper_certify(out,scene,opts)
%UNICONFLOW_PAPER_CERTIFY Deterministic final action-space certification.
if nargin<3,opts=struct();end
if isfield(out,'certified')&&all(out.certified),return;end
n=size(out.states,3);infos=cell(n,1);after=zeros(n,1);
cfg=struct('trust_region',[0.15;3],'activation_margin',0.05, ...
    'max_iterations',50,'slack_penalty',1e8,'target_margin',1e-6);
f=fieldnames(opts);for k=1:numel(f),cfg.(f{k})=opts.(f{k});end
for q=1:n
    [U,S,info]=uniconflow_paper_control_safety_filter(out.states(:,:,q), ...
        out.actions(:,:,q),out.spec,scene.state_constraint,cfg);
    out.actions(:,:,q)=U;out.states(:,:,q)=S;infos{q}=info;after(q)=info.violations;
end
out.trajectory=uniconflow_paper_pack('pack',out.states,out.actions);
out.state_violations_after=after;
tol=1e-8;out.action_feasible=squeeze(all(out.actions>=out.spec.action_lower-tol & ...
    out.actions<=out.spec.action_upper+tol,[1 2]));
out.certified=(after==0)&out.action_feasible(:);
out.safety_filter_info=infos;
out.safety_filter_adaptation=true;
end
