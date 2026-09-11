function exp_l1_refine(step_list)
%EXP_L1_REFINE  Does the level-1 boundary-component overshoot shrink with dt?
%
% Same question as exp_step_refine_ek but for level 1, and resolved per
% component: the BOUNDARY rows (which are inside the QP with weight 1) are
% reported separately from the joint soft-min row.
%
%   e_k = h_k*exp(-phi0*dt) - h_{k+1}      (>0 : below what the CBF allows)
%
% Scaling identifies the source:
%   32x per halving -> RK4 truncation O(dt^5)
%    4x per halving -> sample-and-hold  O(dt^2)
%    2x per halving -> first-order leak O(dt)
%   ~1x (no shrink) -> not a discretization error at all (e.g. a genuine
%                      discontinuity in h from the windowed reference search)
if nargin<1||isempty(step_list), step_list=[100 200 400]; end
root='C:\Users\JieJi\BA\matlab'; cd(root);
R=load(fullfile('outputs','Racing_NoVariance_FirstLevel_Rollout.mat'), ...
  'x_init','rollout_times','saved_first_rollout_constraint','traj_path_10d');
C=load(fullfile('outputs','Racing_NoVariance_Config.mat')); cfg=C.variance_ablation_config;
c=R.saved_first_rollout_constraint;
M=load(fullfile(root,cfg.cache.first_level_model_path));
mf=fieldnames(M); mc=[];
for i=1:numel(mf), v=M.(mf{i}); if isstruct(v)&&isfield(v,'model'), mc=v; break; end; end
mc=strip_model_for_prediction(mc);
t=R.rollout_times; t1=t(end); phi0=struct_field_default(c,'joint_safety_phi0',2);
fprintf('L1: phi0=%g t1=%.5f n_samples=%d\n', phi0, t1, size(R.x_init,1));
res=struct([]);
for si=1:numel(step_list)
  ns=step_list(si); fprintf('\n===== n_steps=%d =====\n', ns);
  tt=tic; [times,path]=rk4_rollout(mc,R.x_init,t(1),t1,ns,c,[],cfg.parallel);
  fprintf('rollout %.1f s\n', toc(tt));
  if ns==100
    fprintf('reproduces cached path: max|diff| = %.3e\n', ...
      max(abs(path(:)-R.traj_path_10d(:))));
  end
  r=analyze(times,path,c,phi0,0.90); r.n_steps=ns; r.dt=times(2)-times(1);
  if isempty(res), res=r; else, res(end+1)=r; end %#ok<AGROW>
end
fprintf('\n================ SUMMARY (t >= 0.90) ================\n');
fprintf('%-7s %-9s | %-11s %-11s | %-11s %-11s | %-7s\n','steps','dt', ...
  'BND e_p99','BND e_med+','SOFT e_p99','SOFT e_med+','nviol');
for i=1:numel(res)
  fprintf('%-7d %-9.5f | %-11.4e %-11.4e | %-11.4e %-11.4e | %-7d\n', ...
    res(i).n_steps,res(i).dt,res(i).b_p99,res(i).b_med,res(i).s_p99,res(i).s_med, ...
    res(i).n_viol);
end
fprintf('\n--- shrink per halving of dt ---\n');
for i=2:numel(res)
  fprintf('%d -> %d :  BND p99 %.2fx  BND med %.2fx  |  SOFT p99 %.2fx  SOFT med %.2fx\n', ...
    res(i-1).n_steps,res(i).n_steps, ...
    res(i-1).b_p99/max(res(i).b_p99,realmin), res(i-1).b_med/max(res(i).b_med,realmin), ...
    res(i-1).s_p99/max(res(i).s_p99,realmin), res(i-1).s_med/max(res(i).s_med,realmin));
end
fprintf('\nreference: O(dt^5)=32x  O(dt^2)=4x  O(dt)=2x  discontinuity=1x\n');
save(fullfile('outputs','Exp_L1_Refine.mat'),'res','step_list');
end

function r=analyze(times,path,cst,phi0,tw)
n_t=numel(times); n_s=size(path,2);
k0=max(find(times>=tw,1,'first')-1,1); dt=diff(times);
eb=cell(n_s,1); es=cell(n_s,1); hv=cell(n_s,1);
parfor s=1:n_s
  c=cst;
  if isfield(cst,'anchor_clf_targets'), c.anchor_clf_target=cst.anchor_clf_targets(s,:)'; end
  if isfield(cst,'track_boundary_reference_s_min_targets')
    c.track_boundary_reference_s_min=cst.track_boundary_reference_s_min_targets(s,:);
    c.track_boundary_reference_s_max=cst.track_boundary_reference_s_max_targets(s,:);
  end
  B=[]; S=[];
  for k=k0:n_t
    x=reshape(path(k,s,:),[],1);
    st=struct('x',x,'mu',zeros(numel(x),1),'sigma2',0);
    oi=obstacle_cbf_info(st,c,times(k)); bi=track_boundary_cbf_info(st,c,times(k));
    ji=joint_safety_softmin_info(oi,bi,st,c,times(k));
    bb=nan(1,5); ss=nan(1,5);
    for p=1:5
      m=bi.row_point_indices==p;
      if any(m), bb(p)=min(bi.h_values(m)./max(vecnorm(bi.A(m,:),2,2),1e-6)); end
      m2=ji.row_point_indices==p;
      if any(m2), ss(p)=min(ji.h_values(m2)); end
    end
    if isempty(B), B=nan(n_t-k0+1,5); S=B; end
    B(k-k0+1,:)=bb; S(k-k0+1,:)=ss;
  end
  d=dt(k0:end);
  ab=B(1:end-1,:).*exp(-phi0*d); e1=ab-B(2:end,:); o1=isfinite(e1)&B(1:end-1,:)>=0;
  as=S(1:end-1,:).*exp(-phi0*d); e2=as-S(2:end,:); o2=isfinite(e2)&S(1:end-1,:)>=0;
  eb{s}=e1(o1); es{s}=e2(o2); hv{s}=S(end,:)';
end
EB=vertcat(eb{:}); ES=vertcat(es{:}); HV=vertcat(hv{:});
pb=EB(EB>0); ps=ES(ES>0);
r=struct('b_p99',prctile(EB,99),'b_med',median([pb;0]), ...
         's_p99',prctile(ES,99),'s_med',median([ps;0]), ...
         'n_viol',sum(HV<0));
end
