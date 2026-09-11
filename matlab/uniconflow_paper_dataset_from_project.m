function data = uniconflow_paper_dataset_from_project(opts)
%UNICONFLOW_PAPER_DATASET_FROM_PROJECT Convert the shared racing paths.
% The source contains geometry only.  We restore metric coordinates, resample
% each path to n_states states, construct bounded bicycle controls that track it,
% and store the exact RK4 rollout as the training state.  This preserves the
% shared path prior while making every demonstration dynamically consistent.
if nargin<1,opts=struct();end
gf=@(f,d)local_default(opts,f,d);cfg=get_config();rng(cfg.random_seed);
ntrain=gf('n_trajectories',cfg.n_train);target_speed=gf('target_speed_mps',30);
% n_states sets the control grid.  dt is derived from the horizon so a coarser
% grid still covers the same path length at the same target speed; without
% that the trajectory would simply be truncated and KL/CS/AS would compare
% different arcs.  spec.n_sub then restores the integration accuracy the
% longer control interval would otherwise cost.
n_states=gf('n_states',101); horizon=n_states-1;
heading_weight=gf('heading_tracking_weight',0.20);
[points,segment]=scenario_training_points(cfg,65,ntrain);
ref_metric=zeros(2,n_states,ntrain);lengths=zeros(1,ntrain);
for n=1:ntrain
    pu=squeeze(points(:,n,1:2));pm=(pu-segment.transform.offset)/segment.transform.scale;
    ds=sqrt(sum(diff(pm,1,1).^2,2));ss=[0;cumsum(ds)];lengths(n)=ss(end);
    sq=linspace(0,ss(end),n_states);ref_metric(:,:,n)=interp1(ss,pm,sq,'pchip').';
end
if isfield(opts,'dt'),dt=opts.dt;dt_source='explicit';
else,dt=median(lengths)/(horizon*target_speed);
    dt_source=sprintf('median_length/(%d*target_speed_mps)',horizon);end
assert(isfinite(dt)&&dt>0,'dt must be positive.');
spec=uniconflow_paper_car_spec(struct('dt',dt,'action_bound_source','section_text', ...
    'n_states',n_states,'n_sub',gf('n_sub',1)));
S=zeros(4,n_states,ntrain);A=zeros(2,horizon,ntrain);track_rmse=zeros(ntrain,1);
max_track_error=zeros(ntrain,1);opt=optimset('Display','off','TolX',1e-8);
for n=1:ntrain
    P=ref_metric(:,:,n);
    dP=[gradient(P(1,:));gradient(P(2,:))];
    theta=unwrap(atan2(dP(2,:),dP(1,:)));
    seglen=sqrt(sum(diff(P,1,2).^2,1));vtarget=[seglen/dt,seglen(end)/dt];
    S(:,1,n)=[P(:,1);theta(1);vtarget(1)];
    for k=1:horizon
        sk=S(:,k,n);tau=(vtarget(k+1)-sk(4))/dt;
        tau=min(max(tau,spec.action_lower(2)),spec.action_upper(2));
        objective=@(delta)step_cost(delta,tau,sk,P(:,k+1),theta(k+1), ...
            max(seglen(k),1),dt,spec.wheelbase,heading_weight,spec.n_sub);
        delta=fminbnd(objective,spec.action_lower(1),spec.action_upper(1),opt);
        A(:,k,n)=[delta;tau];
        S(:,k+1,n)=uniconflow_paper_car_dynamics('step',sk,A(:,k,n),dt, ...
            spec.wheelbase,spec.n_sub);
    end
    err=sqrt(sum((S(1:2,:,n)-P).^2,1));track_rmse(n)=sqrt(mean(err.^2));
    max_track_error(n)=max(err);
end
dyn_max=0;
for n=1:ntrain
    for k=1:horizon
        sn=uniconflow_paper_car_dynamics('step',S(:,k,n),A(:,k,n),dt, ...
            spec.wheelbase,spec.n_sub);
        dyn_max=max(dyn_max,norm(sn-S(:,k+1,n),inf));
    end
end
ref_unit=ref_metric*segment.transform.scale+reshape(segment.transform.offset,2,1,1);
data=struct('states',S,'actions',A,'reference_metric',ref_metric, ...
    'reference_unit',ref_unit,'dt',dt,'dt_source',dt_source, ...
    'target_speed_mps',target_speed,'spec',spec,'segment',segment, ...
    'heading_tracking_weight',heading_weight, ...
    'source_points',points,'n_trajectories',ntrain,'n_states',n_states, ...
    'tracking_rmse_m',track_rmse,'max_tracking_error_m',max_track_error, ...
    'dynamics_residual_max',dyn_max,'action_bound_source','section_text', ...
    'conversion_note',['Actions are reconstructed by bounded one-step tracking; ', ...
    'they are not the unpublished OCP actions used by the paper.']);
end

function j=step_cost(delta,tau,s,target,theta_target,scale,dt,L,heading_weight,n_sub)
sn=uniconflow_paper_car_dynamics('step',s,[delta;tau],dt,L,n_sub);
ep=(sn(1:2)-target)/scale;eth=atan2(sin(sn(3)-theta_target),cos(sn(3)-theta_target));
j=ep.'*ep+heading_weight*eth^2;
end
function v=local_default(s,f,d)
if isfield(s,f),v=s.(f);else,v=d;end
end
