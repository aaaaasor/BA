function [U,S,info] = uniconflow_paper_feasible_homotopy( ...
    Sref,U0,donor_actions,donor_ids,spec,state_constraint_batch,opts)
%UNICONFLOW_PAPER_FEASIBLE_HOMOTOPY Deterministic feasible-library fallback.
% Convex action homotopies preserve action bounds. Every candidate is rolled
% out through the exact dynamics and only strictly safe candidates are used.
if nargin<7,opts=struct();end
step=local_default(opts,'lambda_step',0.02);tol=local_default(opts,'tolerance',1e-8);
lambdas=0:step:1;if lambdas(end)<1,lambdas(end+1)=1;end
nd=size(donor_actions,3);nl=numel(lambdas);nc=nd*nl;
% Horizon from the data rather than the paper's 100, so the 65-state layout
% blends the same way.
Hc=size(U0,2);
assert(size(donor_actions,2)==Hc, ...
    'Donor actions have %d columns but the failed trajectory has %d.', ...
    size(donor_actions,2),Hc);
C=zeros(2,Hc,nc);owner=zeros(nc,1);lam=zeros(nc,1);z=0;
for j=1:nd
    for k=1:nl
        z=z+1;lam(z)=lambdas(k);owner(z)=j;
        C(:,:,z)=(1-lam(z))*U0+lam(z)*donor_actions(:,:,j);
    end
end
Sall=rollout_batch(C,Sref(:,1),spec);V=state_constraint_batch(Sall);
safe=all(V<=tol,1).';assert(any(safe),'No certified donor candidate remained safe.');
dp=Sall(1:2,:,:)-Sref(1:2,:);cost=reshape(mean(sum(dp.^2,1),2),[],1);
% Prefer smaller homotopy motion first; use state-reference RMSE as tie-break.
K=[~safe,lam,cost];[~,ord]=sortrows(K,[1 2 3]);best=ord(1);
U=C(:,:,best);S=Sall(:,:,best);
info=struct('certified',true,'donor_index',donor_ids(owner(best)), ...
    'lambda',lam(best),'position_rmse',sqrt(cost(best)), ...
    'candidates_evaluated',nc,'adaptation_note',['Deterministic feasible ' ...
    'action homotopy used only after CEM and QP certification failed.']);
end

function S=rollout_batch(U,s0,spec)
n=size(U,3);H=size(U,2);ns=H+1;
n_sub=1; if isfield(spec,'n_sub'), n_sub=spec.n_sub; end
S=zeros(4,ns,n);S(:,1,:)=repmat(s0,1,1,n);
for k=1:H
    sk=reshape(S(:,k,:),4,n);ak=reshape(U(:,k,:),2,n);
    S(:,k+1,:)=reshape(uniconflow_paper_car_dynamics( ...
        'step',sk,ak,spec.dt,spec.wheelbase,n_sub),4,1,n);
end
end
function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
