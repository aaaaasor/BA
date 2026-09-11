function out = uniconflow_paper_guided_sample(net, scur, opts)
%UNICONFLOW_PAPER_GUIDED_SAMPLE Reconstructed Stage-1 UniConFlow.
% The trajectory layout follows the network, so the paper's 101-state (604-D)
% case and the 65-state (388-D) layout that matches the other racing
% baselines both run through this code unchanged.
% This implements the published PTZF/slack-QP equations, but cannot reproduce
% unpublished car dt, QP weights, solver grid, or the unreleased track data.
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
t_end=gf('t_end',0.99); cg=gf('c_g',1); cpt=gf('c_PT',2);
assert(cpt>cg && t_end>0 && t_end<1);
P_u=gf('P_u',1); P_delta=gf('P_delta',1000);
% One scalar P_delta weights the slack on every constraint row alike, but the
% rows are not alike: row 1 of c=[g;h] is the squared dynamics-consistency
% residual (an equality), the rest are safety inequalities.  Splitting them
% lets the equality be enforced hard -- so the generated states and actions
% agree and the later action reconstruction barely moves the trajectory --
% while the inequalities keep their slack.  Both default to P_delta, so an
% unset option reproduces the previous behaviour exactly.
P_delta_g=gf('P_delta_g',P_delta);
P_delta_h=gf('P_delta_h',P_delta);
% Algorithm 2 line 3 offers the QP (50) or its closed form (51).
% 'closed_form' keeps the existing behaviour; 'quadprog' solves (50) directly,
% which is the form the paper lists first and stays well posed when the
% active constraint rows are nearly collinear.
qp_solver=char(gf('qp_solver','closed_form'));
qp_fallbacks=0;
qpopt=optimoptions('quadprog','Display','off', ...
    'Algorithm','interior-point-convex');
state_constraint=gf('state_constraint',[]);
record_path=gf('record_path',false);
record_control_trace=gf('record_control_trace',false);
transform_switch=min(gf('transform_switch',0.9),t_end);
transform_step=gf('transform_step',0.25);

mu=net.mu_d(:); sd=net.sd_d(:); sd(sd==0)=1;Sdiag=spdiags(sd,0,D,D);
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

dt_base=1/n_steps;tearly=(0:dt_base:transform_switch).';
if isempty(tearly)||tearly(end)<transform_switch,tearly(end+1,1)=transform_switch;end
tau=tearly./(1-tearly);
if transform_switch<t_end
    tail=(tau(end):transform_step:t_end/(1-t_end)).';
    if tail(end)<t_end/(1-t_end),tail(end+1,1)=t_end/(1-t_end);end
    tau=unique([tau;tail(2:end)],'stable');
end
times=tau./(1+tau);n_actual=numel(tau)-1;path=[];
if record_path, path=zeros(D,n_gen,n_actual+1); path(:,:,1)=z; end
if record_control_trace,utrace=zeros(D,n_gen,n_actual,'single');
else,utrace=zeros(0,0,0,'single');end
gtrace=zeros(n_actual,n_gen); htrace=zeros(n_actual,n_gen);
active_trace=zeros(n_actual,n_gen);
rho_max_trace=-inf(n_actual,n_gen); qp_residual_trace=zeros(n_actual,n_gen);
for ell=1:n_actual
    ta=tau(ell); t=ta/(1+ta); dta=tau(ell+1)-ta;
    v=nn_fwd(net.P,[repmat(t,1,n_gen);z]); dz=zeros(size(z));
    decay=exp(-cg*t/(1-t)); blow=1/(1-t)^2;
    for n=1:n_gen
        Tp=z(:,n).*sd+mu;
        [cv,Jp]=uniconflow_paper_constraints(Tp,spec,scur,state_constraint);
        Jz=Jp*Sdiag; gb=gbar0(n)*decay; hb=hbar0{n}*decay;
        bars=[gb;hb]; bardot=-cg*blow*bars;
        wrow=[P_delta_g;P_delta_h*ones(numel(bars)-1,1)];
        rho=Jz*v(:,n)-cpt*blow*(bars-cv)-bardot;
        active=rho>0; active_trace(ell,n)=nnz(active);
        if ~isempty(rho),rho_max_trace(ell,n)=max(rho);end
        u=zeros(D,1);
        if any(active)
            E=Jz(active,:); rr=rho(active); m=nnz(active);
            switch qp_solver
                case 'closed_form'
                    % Eq. (51): Woodbury solution of the EQUALITY-constrained
                    % problem.  Cheap, but it lets the slack go negative and
                    % inverts M=(1/P_delta)I+(1/P_u)EE', near singular
                    % whenever the active rows are close to collinear.
                    M=diag(1./wrow(active))+(1/P_u)*(E*E.');
                    u=-(1/P_u)*E.'*(M\rr);
                case 'quadprog'
                    % Eq. (50) as written: minimise
                    %   (P_u/2)|u|^2 + (P_delta/2)|d|^2
                    %   s.t.  E u - d <= -rho,  d >= 0.
                    % Same optimum as (51) when every active row is tight,
                    % but the slack stays non-negative and EE' need not be
                    % well conditioned.
                    Hq=spdiags([P_u*ones(D,1);wrow(active)],0,D+m,D+m);
                    Aq=[sparse(E),-speye(m)];
                    x=quadprog(Hq,zeros(D+m,1),Aq,-rr,[],[], ...
                        [-inf(D,1);zeros(m,1)],[],[],qpopt);
                    if isempty(x)
                        M=diag(1./wrow(active))+(1/P_u)*(E*E.');
                        u=-(1/P_u)*E.'*(M\rr);   % fall back if QP fails
                        qp_fallbacks=qp_fallbacks+1;
                    else
                        u=x(1:D);
                    end
                otherwise
                    error('qp_solver must be closed_form or quadprog.');
            end
            qp_residual_trace(ell,n)=max([0;E*(v(:,n)+u)- ...
                cpt*blow*(bars(active)-cv(active))-bardot(active)]);
        end
        if record_control_trace,utrace(:,n,ell)=single(u);end
        dz(:,n)=(1-t)^2*(v(:,n)+u); % dt/dtau=(1-t)^2
        gtrace(ell,n)=cv(1);
        if isempty(cv(2:end)), htrace(ell,n)=-inf;
        else, htrace(ell,n)=max(cv(2:end)); end
    end
    z=z+dta*dz;
    if record_path, path(:,:,ell+1)=z; end %#ok<AGROW>
end

T=z.*sd+mu; S=zeros(4,ns,n_gen); A=zeros(2,H,n_gen);
gfinal=zeros(1,n_gen); hfinal=zeros(1,n_gen);
for n=1:n_gen
    [S(:,:,n),A(:,:,n)]=uniconflow_paper_pack('unpack',T(:,n));
    cv=uniconflow_paper_constraints(T(:,n),spec,scur,state_constraint);
    gfinal(n)=cv(1); if numel(cv)>1,hfinal(n)=max(cv(2:end));else,hfinal(n)=-inf;end
end
out=struct('mode','uniconflow_paper_reconstructed_stage1','trajectory',T, ...
    'states',S,'actions',A,'normalized_state',z,'z0',z0,'z_path',path, ...
    'u_trace',utrace,'u_trace_t',times(1:end-1), ...
    'g_final',gfinal,'h_max_final',hfinal,'g_trace',gtrace, ...
    'h_trace',htrace,'active_trace',active_trace, ...
    'rho_max_trace',rho_max_trace,'qp_residual_trace',qp_residual_trace, ...
    'time_grid',times, ...
    'solver_grid',tau,'n_actual_steps',n_actual,'spec',spec, ...
    'trajectory_seeds',trajectory_seeds, ...
    'seed_convention',['Each trajectory has an independent mt19937ar stream; ' ...
        'z0(:,q)=randn(stream(trajectory_seeds(q)),D,1).'], ...
    'qp_solver',qp_solver,'qp_fallbacks',qp_fallbacks, ...
    'P_delta_g',P_delta_g,'P_delta_h',P_delta_h, ...
    'options',opts,'not_parameter_exact',true);
end

function y=nn_fwd(P,x)
sil=@(q)q./(1+exp(-q));
y=P{4}*sil(P{3}*sil(P{2}*sil(P{1}*x+P{5})+P{6})+P{7})+P{8};
end
function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
