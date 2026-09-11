function [Sproj, diag] = uniconflow_terminal_project_points(S, scene, opts)
%UNICONFLOW_TERMINAL_PROJECT_POINTS Project only unsafe geometric points.
% The five state constraints use h<=0. Each selected point solves successive
% minimum-norm linearized half-space projections in its two position entries.
if nargin<3, opts=struct(); end
tol=field_default(opts,'violation_tol',5e-4);
margin=field_default(opts,'target_margin',1e-6);
max_iter=field_default(opts,'max_iterations',30);
assert(size(S,1)==4,'Expected states with shape 4 x horizon x samples.');
if ismatrix(S),S=reshape(S,size(S,1),size(S,2),1);end
Sproj=S; nq=size(S,3); nk=size(S,2);
before_max=-inf(nq,1); after_max=before_max;
projected_points=zeros(nq,1); failed_points=zeros(nq,1);
max_displacement=zeros(nq,1); total_displacement=zeros(nq,1);
iterations=zeros(nq,nk);

for q=1:nq
    for k=1:nk
        s=Sproj(:,k,q);
        [h,~]=scene.state_constraint(s,k-1);h=h(:);
        before_max(q)=max(before_max(q),max(h));
        if max(h)<=tol,continue;end
        projected_points(q)=projected_points(q)+1;
        p0=s(1:2);
        converged=false;
        for it=1:max_iter
            [h,G]=scene.state_constraint(s,k-1);h=h(:);
            if max(h)<=-margin,converged=true;iterations(q,k)=it-1;break;end
            Gxy=reshape(G,numel(h),4);Gxy=Gxy(:,1:2);
            dp=minnorm_halfspaces(-Gxy,h+margin);
            if any(~isfinite(dp)),break;end
            current=max(h);accepted=false;
            for ls=0:12
                trial=s;trial(1:2)=s(1:2)+(0.5^ls)*dp;
                ht=scene.state_constraint(trial,k-1);ht=ht(:);
                if max(ht)<=-margin || max(ht)<current-1e-12
                    s=trial;accepted=true;break;
                end
            end
            if ~accepted,break;end
            iterations(q,k)=it;
        end
        hf=scene.state_constraint(s,k-1);hf=hf(:);
        if max(hf)<=tol,converged=true;end
        if ~converged,failed_points(q)=failed_points(q)+1;end
        d=norm(s(1:2)-p0);
        max_displacement(q)=max(max_displacement(q),d);
        total_displacement(q)=total_displacement(q)+d;
        Sproj(:,k,q)=s;
    end
    for k=1:nk
        h=scene.state_constraint(Sproj(:,k,q),k-1);h=h(:);
        after_max(q)=max(after_max(q),max(h));
    end
end
diag=struct('violation_tol',tol,'target_margin',margin, ...
    'before_max_h',before_max,'after_max_h',after_max, ...
    'projected_points',projected_points,'failed_points',failed_points, ...
    'max_displacement',max_displacement, ...
    'total_displacement',total_displacement,'iterations',iterations, ...
    'all_certified',all(after_max<=tol)&&all(failed_points==0));
end

function v=field_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
