function test_uniconflow_paper_core
%TEST_UNICONFLOW_PAPER_CORE Deterministic smoke tests for reconstructed core.
spec=uniconflow_paper_car_spec(struct('dt',0.1));
assert(spec.trajectory_dim==604 && spec.n_states==101);
% Analytic RK4 sensitivities must agree with complex-step derivatives.
rs=RandStream('mt19937ar','Seed',90210);
for it=1:20
    sj=[10*randn(rs,2,1);pi*(2*rand(rs)-1);30*rand(rs)];
    aj=[1.8*rand(rs)-0.9;60*rand(rs)-30];
    [fa,Fs,Fa]=uniconflow_paper_car_dynamics('step_jacobian',sj,aj,0.1,2.7);
    fc=uniconflow_paper_car_dynamics('step',sj,aj,0.1,2.7);
    Fsc=zeros(4,4);Fac=zeros(4,2);e=1e-20;
    for j=1:4,q=sj;q(j)=q(j)+1i*e;Fsc(:,j)=imag( ...
            uniconflow_paper_car_dynamics('step',q,aj,0.1,2.7))/e;end
    for j=1:2,q=aj;q(j)=q(j)+1i*e;Fac(:,j)=imag( ...
            uniconflow_paper_car_dynamics('step',sj,q,0.1,2.7))/e;end
    assert(norm(fa-fc,inf)<1e-13&&norm(Fs-Fsc,inf)<1e-11&&norm(Fa-Fac,inf)<1e-11);
end
s=zeros(4,101,2); a=zeros(2,100,2);
for n=1:2
    s(:,1,n)=[0;0;0;2];
    for k=1:100
        a(:,k,n)=[0.05*sin(k/10);0.1*cos(k/7)];
        s(:,k+1,n)=uniconflow_paper_car_dynamics('step',s(:,k,n),a(:,k,n),spec.dt,spec.wheelbase);
    end
end
T=uniconflow_paper_pack('pack',s,a); [sr,ar]=uniconflow_paper_pack('unpack',T);
assert(isequal(size(T),[604 2]));
assert(max(abs(sr(:)-s(:)))<1e-12 && max(abs(ar(:)-a(:)))<1e-12);
for n=1:2
    for k=1:100
        sn=uniconflow_paper_car_dynamics('step',sr(:,k,n),ar(:,k,n),spec.dt,spec.wheelbase);
        assert(norm(sn-sr(:,k+1,n),inf)<1e-11);
    end
end
r=uniconflow_paper_preflight([],s,a,struct('dt',0.1));
assert(~r.ready && any(contains(r.issues,'network')));
net=uniconflow_paper_train(s,a,1,struct('dt',0.1,'hidden_width',8, ...
    'batch_size',2,'seed',7));
r2=uniconflow_paper_preflight(net,s,a,struct('dt',0.1));
assert(r2.ready && numel(net.mu_d)==604);
[c,J]=uniconflow_paper_constraints(T(:,1),spec,s(:,1,1),[]);
assert(c(1)<1e-20 && all(c(2:end)<=0) && isequal(size(J),[201 604]));
o=uniconflow_paper_guided_sample(net,s(:,1,1),struct('dt',0.1, ...
    'n_steps',2,'t_end',0.1,'n_gen',1,'seed',3));
assert(isequal(size(o.trajectory),[604 1]) && all(isfinite(o.trajectory),'all'));
o2=uniconflow_paper_cem_refine(o,struct('cem_population',4,'cem_elite',2, ...
    'cem_iterations',1,'max_passes',1,'seed',5));
assert(size(o2.states,1)==4 && size(o2.states,2)==101 && size(o2.states,3)==1);
assert(size(o2.actions,1)==2 && size(o2.actions,2)==100 && size(o2.actions,3)==1);
for k=1:100
    sn=uniconflow_paper_car_dynamics('step',o2.states(:,k),o2.actions(:,k),0.1,2.7);
    assert(norm(sn-o2.states(:,k+1),inf)<1e-10);
end
fprintf(['uniconflow_paper_core: PASS (604-D pack, RK4 dynamics, ', ...
    'constraints, training, PTZF-QP, window CEM smoke)\n']);
end
