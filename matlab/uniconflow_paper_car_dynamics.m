function varargout = uniconflow_paper_car_dynamics(mode, s, a, dt, L, n_sub)
%UNICONFLOW_PAPER_CAR_DYNAMICS Eq. (109) and an RK4 discrete map.
% State s=[x;y;theta;v], action a=[delta;tau].
%
% n_sub splits the control interval dt into n_sub RK4 sub-steps while the
% action is held constant across them.  This decouples the trajectory
% representation from the integration accuracy: a coarser control grid (say 65
% states instead of the paper's 101) can keep the same local truncation error
% by integrating each of its longer intervals with several sub-steps.  The
% local RK4 error scales with the sub-step to the fifth power, so n_sub=2
% already more than compensates a 1.56x longer interval.
if nargin < 5 || isempty(L), L=2.7; end
if nargin < 6 || isempty(n_sub), n_sub=1; end
assert(isscalar(n_sub) && n_sub>=1 && n_sub==round(n_sub), ...
    'n_sub must be a positive integer.');
switch lower(char(mode))
    case 'continuous'
        varargout{1} = rhs(s,a,L);
    case 'step'
        assert(nargin>=4 && isscalar(dt) && isfinite(dt) && dt>0, ...
            'A positive car sampling time dt is required; the paper does not publish it.');
        h=dt/n_sub; sn=s;
        for i=1:n_sub, sn=rk4(sn,a,h,L); end
        varargout{1}=sn;
    case 'step_jacobian'
        assert(nargin>=4 && isscalar(dt) && isfinite(dt) && dt>0, ...
            'A positive car sampling time dt is required; the paper does not publish it.');
        assert(isequal(size(s),[4 1])&&isequal(size(a),[2 1]), ...
            'step_jacobian expects one 4x1 state and one 2x1 action.');
        h=dt/n_sub; sn=s; Fs=eye(4); Fa=zeros(4,2);
        for i=1:n_sub
            [sn,Fs_i,Fa_i]=rk4_jacobian(sn,a,h,L);
            % The action is shared by every sub-step, so its sensitivity
            % accumulates through the later sub-steps' state Jacobians.
            Fa=Fs_i*Fa+Fa_i;
            Fs=Fs_i*Fs;
        end
        varargout={sn,Fs,Fa};
    otherwise
        error('mode must be continuous, step, or step_jacobian.');
end
end

function sn=rk4(s,a,dt,L)
k1=rhs(s,a,L); k2=rhs(s+0.5*dt*k1,a,L);
k3=rhs(s+0.5*dt*k2,a,L); k4=rhs(s+dt*k3,a,L);
sn=s+dt*(k1+2*k2+2*k3+k4)/6;
end

function [sn,Fs,Fa]=rk4_jacobian(s,a,dt,L)
[k1,A1,B1]=rhs_jacobian(s,a,L);
X2s=eye(4)+0.5*dt*A1;X2a=0.5*dt*B1;
[k2,A2,B2]=rhs_jacobian(s+0.5*dt*k1,a,L);
K2s=A2*X2s;K2a=A2*X2a+B2;
X3s=eye(4)+0.5*dt*K2s;X3a=0.5*dt*K2a;
[k3,A3,B3]=rhs_jacobian(s+0.5*dt*k2,a,L);
K3s=A3*X3s;K3a=A3*X3a+B3;
X4s=eye(4)+dt*K3s;X4a=dt*K3a;
[k4,A4,B4]=rhs_jacobian(s+dt*k3,a,L);
K4s=A4*X4s;K4a=A4*X4a+B4;
sn=s+dt*(k1+2*k2+2*k3+k4)/6;
Fs=eye(4)+dt*(A1+2*K2s+2*K3s+K4s)/6;
Fa=dt*(B1+2*K2a+2*K3a+K4a)/6;
end

function ds=rhs(s,a,L)
assert(size(s,1)==4 && size(a,1)==2,'Expected s:4xN and a:2xN.');
if size(a,2)==1 && size(s,2)>1, a=repmat(a,1,size(s,2)); end
assert(size(s,2)==size(a,2),'State/action batch mismatch.');
th=s(3,:); v=s(4,:); delta=a(1,:); tau=a(2,:);
ds=[v.*cos(th); v.*sin(th); v./L.*tan(delta); tau];
end

function [ds,A,B]=rhs_jacobian(s,a,L)
ds=rhs(s,a,L);th=s(3);v=s(4);delta=a(1);
A=zeros(4,4);B=zeros(4,2);
A(1,3)=-v*sin(th);A(1,4)=cos(th);
A(2,3)= v*cos(th);A(2,4)=sin(th);
A(3,4)=tan(delta)/L;
B(3,1)=v/(L*cos(delta)^2);B(4,2)=1;
end
