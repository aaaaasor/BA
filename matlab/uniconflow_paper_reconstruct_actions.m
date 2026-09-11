function [A,S] = uniconflow_paper_reconstruct_actions(Sref,spec,opts)
%UNICONFLOW_PAPER_RECONSTRUCT_ACTIONS Bounded one-step inverse initializer.
% It converts a generated state reference into a dynamically exact initial
% action rollout before the paper's CEM terminal refinement.
if nargin<3,opts=struct();end
heading_weight=local_default(opts,'heading_weight',0.2);
opt=optimset('Display','off','TolX',1e-7);
n_sub=1; if isfield(spec,'n_sub'), n_sub=spec.n_sub; end
S=zeros(size(Sref));A=zeros(2,size(Sref,2)-1);S(:,1)=Sref(:,1);
for k=1:size(A,2)
    sk=S(:,k);tau=(Sref(4,k+1)-sk(4))/spec.dt;
    tau=min(max(tau,spec.action_lower(2)),spec.action_upper(2));
    target=Sref(1:2,k+1);theta_target=Sref(3,k+1);
    scale=max(norm(target-Sref(1:2,k)),1);
    objective=@(delta)step_cost(delta,tau,sk,target,theta_target, ...
        scale,spec.dt,spec.wheelbase,heading_weight,n_sub);
    delta=fminbnd(objective,spec.action_lower(1),spec.action_upper(1),opt);
    A(:,k)=[delta;tau];
    S(:,k+1)=uniconflow_paper_car_dynamics('step',sk,A(:,k), ...
        spec.dt,spec.wheelbase,n_sub);
end
end

function j=step_cost(delta,tau,s,target,theta_target,scale,dt,L,w,n_sub)
sn=uniconflow_paper_car_dynamics('step',s,[delta;tau],dt,L,n_sub);
ep=(sn(1:2)-target)/scale;
eth=atan2(sin(sn(3)-theta_target),cos(sn(3)-theta_target));
j=ep.'*ep+w*eth^2;
end
function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
