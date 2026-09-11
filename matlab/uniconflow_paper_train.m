function net = uniconflow_paper_train(states, actions, n_steps, opts)
%UNICONFLOW_PAPER_TRAIN Train a joint state-action FM reconstruction.
% states: 4 x n_states x N, actions: 2 x (n_states-1) x N.  The paper does not publish its network
% architecture or optimizer settings; all such choices are recorded below.
if nargin<3 || isempty(n_steps), n_steps=5000; end
if nargin<4, opts=struct(); end
gf=@(f,d) local_default(opts,f,d);
spec=uniconflow_paper_car_spec(opts);
assert(size(states,1)==4,'states must be 4 x n_states x N.');
assert(size(actions,1)==2,'actions must be 2 x horizon x N.');
assert(size(actions,2)==size(states,2)-1, ...
    'actions must have n_states-1 columns (%d vs %d).',size(actions,2),size(states,2)-1);
assert(size(states,3)==size(actions,3),'states/actions batch mismatch.');
% The layout follows the data, so the same trainer serves the paper's 101-state
% case and the 65-state layout used to match the other racing baselines.
spec=uniconflow_paper_car_spec(setfield(opts,'n_states',size(states,2))); %#ok<SFLD>

X1=double(uniconflow_paper_pack('pack',states,actions));
N=size(X1,2); D=size(X1,1); assert(D==spec.trajectory_dim);
mu=mean(X1,2); sd=std(X1,0,2); sd(sd<1e-12)=1; X1n=(X1-mu)./sd;
H=gf('hidden_width',256); B=min(gf('batch_size',256),N);
lr=gf('learning_rate',1e-3); seed=gf('seed',20260101);
rng(seed); Din=D+1;
P={randn(H,Din)*sqrt(2/Din),randn(H,H)*sqrt(2/H), ...
   randn(H,H)*sqrt(2/H),randn(D,H)*sqrt(2/H), ...
   zeros(H,1),zeros(H,1),zeros(H,1),zeros(D,1)};
m=cellfun(@(x)zeros(size(x)),P,'UniformOutput',false); v=m;
be1=.9; be2=.999; ep=1e-8; loss=zeros(n_steps,1);
train_timer=tic; report_every=max(1,floor(n_steps/10));
for k=1:n_steps
    idx=randi(N,1,B); x1=X1n(:,idx); x0=randn(D,B); t=rand(1,B);
    xt=(1-t).*x0+t.*x1; target=x1-x0;
    [pred,c]=fwd(P,[t;xt]); e=pred-target; loss(k)=mean(e.^2,'all');
    G=bwd(P,c,2*e/(D*B));
    for j=1:8
        m{j}=be1*m{j}+(1-be1)*G{j}; v{j}=be2*v{j}+(1-be2)*G{j}.^2;
        P{j}=P{j}-lr*(m{j}/(1-be1^k))./(sqrt(v{j}/(1-be2^k))+ep);
    end
    if mod(k,report_every)==0 || k==n_steps
        fprintf('paper joint FM step %d/%d loss %.6g\n',k,n_steps, ...
            mean(loss(max(1,k-report_every+1):k)));
    end
end
train_seconds=toc(train_timer);
net=struct('P',{P},'mu_d',mu,'sd_d',sd,'loss',loss, ...
    'trajectory_dim',D,'horizon',size(actions,2),'state_dim',4,'action_dim',2, ...
    'n_states',size(states,2), ...
    'state_order',{{'x','y','theta','v'}},'action_order',{{'delta','tau'}}, ...
    'feature_order',{{'x','y','theta','v','delta','tau'}}, ...
    'layout','interleaved_state_action','variant',spec.variant, ...
    'n_training_trajectories',N,'train_seconds',train_seconds, ...
    'training_options',struct( ...
    'n_steps',n_steps,'hidden_width',H,'batch_size',B,'learning_rate',lr, ...
    'seed',seed,'paper_hyperparameters_published',false));
end

function [y,c]=fwd(P,x)
sil=@(z)z./(1+exp(-z)); z1=P{1}*x+P{5}; a1=sil(z1);
z2=P{2}*a1+P{6}; a2=sil(z2); z3=P{3}*a2+P{7}; a3=sil(z3);
y=P{4}*a3+P{8}; c=struct('x',x,'z1',z1,'a1',a1,'z2',z2, ...
    'a2',a2,'z3',z3,'a3',a3);
end
function G=bwd(P,c,dy)
sig=@(z)1./(1+exp(-z)); dsl=@(z)sig(z).*(1+z.*(1-sig(z)));
g4=dy*c.a3.'; gb4=sum(dy,2); d3=(P{4}.'*dy).*dsl(c.z3);
g3=d3*c.a2.'; gb3=sum(d3,2); d2=(P{3}.'*d3).*dsl(c.z2);
g2=d2*c.a1.'; gb2=sum(d2,2); d1=(P{2}.'*d2).*dsl(c.z1);
G={d1*c.x.',g2,g3,g4,sum(d1,2),gb2,gb3,gb4};
end
function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
