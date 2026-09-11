function out = uniconflow_paper_guided_sample_rk4(net, scur, opts)
%UNICONFLOW_PAPER_GUIDED_SAMPLE_RK4 Fixed-grid physical-time RK4 Stage 1.
% Keeps the reconstructed UniConFlow PTZF/slack-QP equations unchanged and
% changes only the ODE discretization. The singular endpoint t=1 is excluded.
if nargin<3, opts=struct(); end
gf=@(f,d)local_default(opts,f,d);
spec=uniconflow_paper_car_spec(opts);
assert(isfinite(spec.dt),'opts.dt must be supplied; car dt is unpublished.');
assert(isfield(net,'mu_d'),'A joint state-action FM network is required.');
D=numel(net.mu_d); [ns,H]=uniconflow_paper_layout(D);
assert(D==spec.trajectory_dim, ...
    'Network is %d-D but spec expects %d-D; pass opts.n_states=%d.', ...
    D,spec.trajectory_dim,ns);
n_gen=gf('n_gen',1); n_steps=gf('n_steps',100); seed=gf('seed',11);
trajectory_seeds=gf('trajectory_seeds',[]);
if isempty(trajectory_seeds)
    seed_stream=RandStream('mt19937ar','Seed',seed);
    trajectory_seeds=randi(seed_stream,2^31-2,n_gen,1);
else
    trajectory_seeds=double(trajectory_seeds(:));
    assert(numel(trajectory_seeds)==n_gen, ...
        'trajectory_seeds must contain exactly n_gen entries.');
end
t_end=gf('t_end',0.999); cg=gf('c_g',1); cpt=gf('c_PT',3);
assert(cpt>cg && t_end>0 && t_end<1);
P_u=gf('P_u',1); P_delta=gf('P_delta',1e4);
ptzf_terminal_time=gf('ptzf_terminal_time',1);
ptzf_time_shift=gf('ptzf_time_shift',0);
endpoint_extension=gf('endpoint_extension',false);
time_varying_gain=gf('time_varying_gain',true);
gain_cap=gf('gain_cap',inf);
endpoint_gain=gf('endpoint_gain',cpt);
assert(ptzf_terminal_time>0 && ptzf_time_shift>=0);
state_constraint=gf('state_constraint',[]);
record_path=gf('record_path',false);
record_control_trace=gf('record_control_trace',false);

mu=net.mu_d(:); sd=net.sd_d(:); sd(sd==0)=1;
Sdiag=spdiags(sd,0,D,D);
z=zeros(D,n_gen);
for n=1:n_gen
    stream=RandStream('mt19937ar','Seed',trajectory_seeds(n));
    z(:,n)=randn(stream,D,1);
end
z0=z; gbar0=zeros(1,n_gen); hbar0=cell(1,n_gen);
for n=1:n_gen
    Tp=z(:,n).*sd+mu;
    [cv,~,~]=uniconflow_paper_constraints(Tp,spec,scur,state_constraint);
    gbar0(n)=2*cv(1); hbar0{n}=cv(2:end);
end

times=linspace(0,t_end,n_steps+1).'; dt=t_end/n_steps;
path=[];
if record_path, path=zeros(D,n_gen,n_steps+1); path(:,:,1)=z; end
if record_control_trace, utrace=zeros(D,n_gen,n_steps,'single');
else, utrace=zeros(0,0,0,'single'); end
gtrace=zeros(n_steps,n_gen); htrace=zeros(n_steps,n_gen);
active_trace=zeros(n_steps,n_gen); rho_max_trace=-inf(n_steps,n_gen);
qp_residual_trace=zeros(n_steps,n_gen);

for ell=1:n_steps
    t=times(ell);
    [k1,u1,d1]=rhs(t,z);
    [k2,u2]=rhs(t+dt/2,z+(dt/2)*k1);
    [k3,u3]=rhs(t+dt/2,z+(dt/2)*k2);
    [k4,u4]=rhs(t+dt,z+dt*k3);
    z=z+(dt/6)*(k1+2*k2+2*k3+k4);
    if record_path, path(:,:,ell+1)=z; end %#ok<AGROW>
    if record_control_trace
        utrace(:,:,ell)=single((u1+2*u2+2*u3+u4)/6);
    end
    gtrace(ell,:)=d1.g;
    htrace(ell,:)=d1.h;
    active_trace(ell,:)=d1.active;
    rho_max_trace(ell,:)=d1.rho_max;
    qp_residual_trace(ell,:)=d1.qp_residual;
    if any(~isfinite(z(:)))
        error('UniConFlow:RK4NonFinite', ...
            'Fixed-grid RK4 became non-finite at step %d (t=%.9g).',ell,t+dt);
    end
end

T=z.*sd+mu; S=zeros(4,ns,n_gen); A=zeros(2,H,n_gen);
gfinal=zeros(1,n_gen); hfinal=zeros(1,n_gen);
for n=1:n_gen
    [S(:,:,n),A(:,:,n)]=uniconflow_paper_pack('unpack',T(:,n));
    cv=uniconflow_paper_constraints(T(:,n),spec,scur,state_constraint);
    gfinal(n)=cv(1);
    if numel(cv)>1,hfinal(n)=max(cv(2:end));else,hfinal(n)=-inf;end
end
out=struct('mode','uniconflow_paper_reconstructed_stage1_rk4_fixed', ...
    'trajectory',T,'states',S,'actions',A,'normalized_state',z, ...
    'z0',z0,'z_path',path,'u_trace',utrace, ...
    'u_trace_t',times(1:end-1),'g_final',gfinal, ...
    'h_max_final',hfinal,'g_trace',gtrace,'h_trace',htrace, ...
    'active_trace',active_trace,'rho_max_trace',rho_max_trace, ...
    'qp_residual_trace',qp_residual_trace,'time_grid',times, ...
    'solver_grid',times,'n_actual_steps',n_steps,'spec',spec, ...
    'trajectory_seeds',trajectory_seeds, ...
    'seed_convention',['Each trajectory has an independent mt19937ar stream; ' ...
        'z0(:,q)=randn(stream(trajectory_seeds(q)),D,1).'], ...
    'options',opts,'not_parameter_exact',true);

    function [dz,uall,diag]=rhs(tnow,znow)
        v=nn_fwd(net.P,[repmat(tnow,1,n_gen);znow]);
        dz=zeros(size(znow)); uall=zeros(size(znow));
        diag=struct('g',zeros(1,n_gen),'h',-inf(1,n_gen), ...
            'active',zeros(1,n_gen),'rho_max',-inf(1,n_gen), ...
            'qp_residual',zeros(1,n_gen));
        teff=(tnow+ptzf_time_shift)/ptzf_terminal_time;
        if endpoint_extension && teff>=1
            decay=0; shape_dot=0; gain_multiplier=endpoint_gain/cpt;
        else
            remaining=1-teff;
            if remaining<=0
                error('UniConFlow:PTZFEndpoint', ...
                    'PTZF effective time reached its singular endpoint.');
            end
            decay=exp(-cg*teff/remaining);
            shape_dot=(1/ptzf_terminal_time)/remaining^2;
            if time_varying_gain
                gain_multiplier=remaining^-2;
            else
                gain_multiplier=1;
            end
        end
        gain=min(cpt*gain_multiplier,gain_cap);
        for q=1:n_gen
            Tp=znow(:,q).*sd+mu;
            [cv,Jp]=uniconflow_paper_constraints(Tp,spec,scur,state_constraint);
            Jz=Jp*Sdiag; bars=[gbar0(q)*decay;hbar0{q}*decay];
            bardot=-cg*shape_dot*bars;
            rho=Jz*v(:,q)-gain*(bars-cv)-bardot;
            active=rho>0; u=zeros(D,1);
            if any(active)
                E=Jz(active,:); rr=rho(active);
                M=(1/P_delta)*eye(nnz(active))+(1/P_u)*(E*E.');
                u=-(1/P_u)*E.'*(M\rr);
                diag.qp_residual(q)=max([0;E*(v(:,q)+u)- ...
                    gain*(bars(active)-cv(active))-bardot(active)]);
            end
            dz(:,q)=v(:,q)+u; uall(:,q)=u;
            diag.g(q)=cv(1);
            if numel(cv)>1,diag.h(q)=max(cv(2:end));end
            diag.active(q)=nnz(active);
            if ~isempty(rho),diag.rho_max(q)=max(rho);end
        end
    end
end

function y=nn_fwd(P,x)
sil=@(q)q./(1+exp(-q));
y=P{4}*sil(P{3}*sil(P{2}*sil(P{1}*x+P{5})+P{6})+P{7})+P{8};
end

function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
