function [c, J, detail] = uniconflow_paper_constraints(T, spec, scur, state_constraint)
%UNICONFLOW_PAPER_CONSTRAINTS Paper Eqs. (63),(70),(73),(75),(77).
% c=[g;h] uses the paper convention g>=0 (squared equality residual) and
% h<=0 for inequalities. J=d[c]/dT. state_constraint(s,k) must return
% [h,dhds], where each row of dhds corresponds to one h value.
if nargin<4, state_constraint=[]; end
assert(isvector(T),'T must be one trajectory vector.');
assert(numel(scur)==4,'scur must contain [x;y;theta;v].');
assert(isfinite(spec.dt),'spec.dt is required for dynamics constraints.');
% The layout is read off the trajectory, so this serves the paper's 101-state
% case and the 65-state layout used to match the other racing baselines.
D=numel(T); [ns,H]=uniconflow_paper_layout(D);
n_sub=struct_field_or(spec,'n_sub',1);
T=T(:); scur=scur(:); [S,A]=uniconflow_paper_pack('unpack',T);
S=S(:,:,1); A=A(:,:,1); Jg=zeros(1,D); g=0;

r0=S(:,1)-scur; g=g+r0.'*r0; Jg(state_idx(0,ns))=2*r0.';
dyn_res=zeros(4,H);
for k=0:H-1
    sk=S(:,k+1); ak=A(:,k+1);
    [fk,Fs,Fa]=step_jacobian(sk,ak,spec.dt,spec.wheelbase,n_sub);
    r=S(:,k+2)-fk; dyn_res(:,k+1)=r; g=g+r.'*r;
    is=state_idx(k,ns); ia=action_idx(k); in=state_idx(k+1,ns);
    Jg(is)=Jg(is)-2*r.'*Fs;
    Jg(ia)=Jg(ia)-2*r.'*Fa;
    Jg(in)=Jg(in)+2*r.';
end

h_state=[];Jh_state=sparse(0,D);
if ~isempty(state_constraint)
    [hk0,Gk0]=state_constraint(S(:,1),0);hk0=hk0(:);m=numel(hk0);
    h_state=zeros(ns*m,1);nzmax=ns*m*4;
    ir=zeros(nzmax,1);jc=ir;vv=ir;pos=0;
    h_state(1:m)=hk0;
    [rr,cc]=ndgrid(1:m,state_idx(0,ns));ids=pos+(1:numel(rr));
    G0=reshape(Gk0,m,4);ir(ids)=rr(:);jc(ids)=cc(:);vv(ids)=G0(:);pos=pos+numel(rr);
    for k=1:H
        [hk,Gk]=state_constraint(S(:,k+1),k);hk=hk(:);
        assert(numel(hk)==m,'state_constraint row count must be constant over time.');
        rows=k*m+(1:m);h_state(rows)=hk;
        [rr,cc]=ndgrid(rows,state_idx(k,ns));ids=pos+(1:numel(rr));
        G=reshape(Gk,m,4);ir(ids)=rr(:);jc(ids)=cc(:);vv(ids)=G(:);pos=pos+numel(rr);
    end
    Jh_state=sparse(ir,jc,vv,ns*m,D);
end

% Paper-style differentiable squared action constraints.  The selected bounds
% are recorded in spec because Section VI-C1 and Eq. (112) conflict.
b=max(abs([spec.action_lower(:),spec.action_upper(:)]),[],2);
na=2*H;
h_action=zeros(na,1);ir=(1:na).';jc=zeros(na,1);vv=zeros(na,1);
for k=0:H-1
    ak=A(:,k+1); ia=action_idx(k);
    rows=2*k+(1:2);h_action(rows)=ak.^2-b.^2;
    jc(rows)=ia(:);vv(rows)=2*ak;
end
h_action=h_action(:);Jh_action=sparse(ir,jc,vv,na,D);
h=[h_state;h_action];Jh=[Jh_state;Jh_action];
c=[g;h];J=[sparse(Jg);Jh];
detail=struct('g',g,'h',h,'dynamics_residual',dyn_res, ...
    'initial_residual',r0,'states',S,'actions',A);
end

function idx=state_idx(k,ns)
% Interleaved layout: each of the first ns-1 blocks is 4 states + 2 actions,
% and the terminal state occupies the final 4 entries.
if k<ns-1, idx=6*k+(1:4); else, idx=6*(ns-1)+(1:4); end
end
function v=struct_field_or(s,f,d)
if isfield(s,f), v=s.(f); else, v=d; end
end
function idx=action_idx(k)
idx=6*k+(5:6);
end
function [f,Fs,Fa]=step_jacobian(s,a,dt,L,n_sub)
[f,Fs,Fa]=uniconflow_paper_car_dynamics('step_jacobian',s,a,dt,L,n_sub);
end
