function [U,S,info] = uniconflow_paper_control_safety_filter(Sref,U0,spec,state_constraint,opts)
%UNICONFLOW_PAPER_CONTROL_SAFETY_FILTER Deterministic action-space safety repair.
% Sequential convexification uses exact RK4 sensitivities, bounded controls,
% soft linearized inequalities, and accepts only lexicographic improvement.
if nargin<5,opts=struct();end
gf=@(f,d)local_default(opts,f,d);
max_iter=gf('max_iterations',30);activation=gf('activation_margin',0.02);
target=gf('target_margin',1e-6);Pslack=gf('slack_penalty',1e6);
trust=gf('trust_region',[0.08;1.5]);tol=gf('tolerance',1e-8);
U=max(U0,spec.action_lower);U=min(U,spec.action_upper);s0=Sref(:,1);
% Layout from the data; n_sub from the spec so this filter integrates exactly
% like the constraints and the CEM rollout it is repairing.
% Named nH, not H: line 22 below reuses H for the QP Hessian.
nH=size(U,2); ns=nH+1; nu=2*nH;
n_sub=1; if isfield(spec,'n_sub'), n_sub=spec.n_sub; end
[S,D]=rollout_sensitivity(U,s0);key=state_key(S,U,U0);
trace=zeros(max_iter+1,4);trace(1,:)=key;n_qp=0;
qpopt=optimoptions('quadprog','Display','off','Algorithm','interior-point-convex');
for it=1:max_iter
    [h,A]=linearized_rows(S,D,activation);
    if key(1)==0,break;end
    m=numel(h);n_qp=n_qp+1;
    H=spdiags([ones(nu,1);Pslack*ones(m,1)],0,nu+m,nu+m);
    Ai=[A,-speye(m)];bi=-h-target;
    lo=max(repmat(spec.action_lower,nH,1)-U(:),repmat(-trust,nH,1));
    hi=min(repmat(spec.action_upper,nH,1)-U(:),repmat( trust,nH,1));
    [x,~,flag]=quadprog(H,zeros(nu+m,1),Ai,bi,[],[], ...
        [lo;zeros(m,1)],[hi;inf(m,1)],[],qpopt);
    if flag<=0||isempty(x),break;end
    du=reshape(x(1:nu),2,nH);accepted=false;
    for beta=[1,0.5,0.25,0.125,0.0625]
        Ut=max(U+beta*du,spec.action_lower);Ut=min(Ut,spec.action_upper);
        [St,Dt]=rollout_sensitivity(Ut,s0);kt=state_key(St,Ut,U0);
        if lex_less(kt,key),U=Ut;S=St;D=Dt;key=kt;accepted=true;break;end
    end
    trace(it+1,:)=key;
    if ~accepted,break;end
end
last=find(any(trace~=0,2),1,'last');if isempty(last),last=1;end
info=struct('certified',key(1)==0,'iterations',last-1,'n_qp',n_qp, ...
    'violations',key(1),'max_violation',key(2),'trace',trace(1:last,:), ...
    'options',opts,'adaptation_note',['Deterministic action-space terminal ' ...
    'filter added after the paper-reconstructed CEM for project certification.']);

    function [X,J]=rollout_sensitivity(Ac,x0)
        X=zeros(4,ns);X(:,1)=x0;J=zeros(4,nu,ns);
        for kk=1:nH
            [X(:,kk+1),Fs,Fa]=uniconflow_paper_car_dynamics( ...
                'step_jacobian',X(:,kk),Ac(:,kk),spec.dt,spec.wheelbase,n_sub);
            J(:,:,kk+1)=Fs*J(:,:,kk);cols=2*kk+(-1:0);
            J(:,cols,kk+1)=J(:,cols,kk+1)+Fa;
        end
    end
    function [hv,Av]=linearized_rows(X,J,margin)
        hv=zeros(0,1);Av=zeros(0,nu);
        for kk=1:ns
            [hh,G]=state_constraint(X(:,kk),kk-1);hh=hh(:);G=reshape(G,numel(hh),4);
            use=hh>-margin;
            if any(use),hv=[hv;hh(use)];Av=[Av;G(use,:)*J(:,:,kk)];end %#ok<AGROW>
        end
        Av=sparse(Av);
    end
    function k=state_key(X,Ac,Aref)
        hh=zeros(ns,1);
        for kk=1:ns,v=state_constraint(X(:,kk),kk-1);hh(kk)=max(v(:));end
        pos=max(hh,0);k=[nnz(hh>tol),max([0;hh]),sum(pos.^2),sum((Ac-Aref).^2,'all')];
    end
end

function tf=lex_less(a,b)
tf=false;for k=1:numel(a),if a(k)<b(k)-1e-12,tf=true;return;elseif a(k)>b(k)+1e-12,return;end,end
end
function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
