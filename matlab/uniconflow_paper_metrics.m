function m = uniconflow_paper_metrics(out, scene, tol)
%UNICONFLOW_PAPER_METRICS SR-S, SR-A, AR, TSR and dynamics residual.
if nargin<3||isempty(tol),tol=1e-8;end
n=size(out.states,3);safe=false(n,1);action=false(n,1);kc=zeros(n,1);
ns=size(out.states,2);H=ns-1;   % layout follows the data, not the paper's 101
for q=1:n
    h=-inf(1,ns);
    for k=1:ns,h(k)=scene.state_constraint(out.states(:,k,q),k-1);end
    safe(q)=all(h<=tol);
    A=out.actions(:,:,q);action(q)=all(A>=out.spec.action_lower-tol,'all')&& ...
        all(A<=out.spec.action_upper+tol,'all');
    r=zeros(4,H);
    for k=1:H
        r(:,k)=out.states(:,k+1,q)-uniconflow_paper_car_dynamics( ...
            'step',out.states(:,k,q),A(:,k),out.spec.dt,out.spec.wheelbase);
    end
    kc(q)=sqrt(mean(r.^2,'all'));
end
m=struct('sr_s',mean(safe),'sr_a',mean(safe),'ar',mean(action), ...
    'tsr',mean(safe&action&(kc<=tol)),'kc_f',mean(kc), ...
    'state_safe_per_trajectory',safe,'action_safe_per_trajectory',action, ...
    'kc_f_per_trajectory',kc,'tolerance',tol);
end
