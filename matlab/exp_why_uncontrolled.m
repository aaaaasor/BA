function exp_why_uncontrolled()
%EXP_WHY_UNCONTROLLED  Was the CBF row actually satisfied when the QP solved it?
%
% Re-runs the level-1 no-variance rollout with the control trace on, then for
% the last steps recomputes, for the joint soft-min row of every controlled
% point, the CBF residual
%       r = grad_h * (mu + u) + phi * h        (constraint is r >= 0)
% at each RK4 stage k1..k4, using the stage's own state and mu but the frozen
% u from k1.  If r(k1) >= 0 the QP did satisfy the constraint at the sampling
% instant, and any realized violation came from within the step.
root='C:\Users\JieJi\BA\matlab'; cd(root);
R=load(fullfile('outputs','Racing_NoVariance_FirstLevel_Rollout.mat'), ...
  'x_init','rollout_times','saved_first_rollout_constraint');
C=load(fullfile('outputs','Racing_NoVariance_Config.mat'));
cfg=C.variance_ablation_config;
c=R.saved_first_rollout_constraint;
c.control_trace_enabled=true; c.control_trace_u_only=false;
c.diagnostics=false;  % control trace is only collected when diagnostics is off
M=load(fullfile(root,cfg.cache.first_level_model_path));
mf=fieldnames(M); mc=[];
for i=1:numel(mf), v=M.(mf{i}); if isstruct(v)&&isfield(v,'model'), mc=v; break; end; end
mc=strip_model_for_prediction(mc);
t=R.rollout_times; t1=t(end); ns=numel(t)-1;
fprintf('re-running L1: %d samples, %d steps, t1=%.5f\n', size(R.x_init,1), ns, t1);
tt=tic;
[times,path,diag]=rk4_rollout(mc,R.x_init,t(1),t1,ns,c,[],cfg.parallel);
diag = diag.hocbf;
if ~isfield(diag,'n_trace_entries')
  diag.n_trace_entries = size(diag.trace_u, 1);   % finalize already trimmed
end
fprintf('rollout %.1f s ; trace entries %d\n', toc(tt), diag.n_trace_entries);
save(fullfile('outputs','Exp_WhyUncontrolled_raw.mat'),'times','path','diag','-v7.3');

n=diag.n_trace_entries;
si=diag.trace_sample_idx(1:n); ki=diag.trace_step_idx(1:n);
gi=diag.trace_stage_idx(1:n);  tv=diag.trace_t(1:n);
U=diag.trace_u(1:n,:); MU=diag.trace_mu(1:n,:);
phi0=struct_field_default(c,'joint_safety_phi0',2);
om  =struct_field_default(c,'joint_safety_phi1_omega',0.5);
last=ns;   % final step index
fprintf('\n=== joint soft-min CBF residual at each RK4 stage, FINAL step ===\n');
fprintf('(r>=0 means the constraint is satisfied at that stage)\n');
fprintf('%-8s %-6s %-6s %-9s %-12s %-12s %-12s\n','sample','pt','stage','t','h','r','r/|grad||mu+u|');
viol=[];
for s=1:size(path,2)
  cc=c;
  if isfield(c,'anchor_clf_targets'), cc.anchor_clf_target=c.anchor_clf_targets(s,:)'; end
  if isfield(c,'track_boundary_reference_s_min_targets')
    cc.track_boundary_reference_s_min=c.track_boundary_reference_s_min_targets(s,:);
    cc.track_boundary_reference_s_max=c.track_boundary_reference_s_max_targets(s,:);
  end
  xe=reshape(path(end,s,:),[],1);
  st=struct('x',xe,'mu',zeros(numel(xe),1),'sigma2',0);
  oi=obstacle_cbf_info(st,cc,t(end)); bi=track_boundary_cbf_info(st,cc,t(end));
  ji=joint_safety_softmin_info(oi,bi,st,cc,t(end));
  if isempty(ji.h_values)||all(ji.h_values>=0), continue; end
  bad=ji.row_point_indices(ji.h_values<0);
  for g=1:4
    idx=find(si==s & ki==last & gi==g,1);
    if isempty(idx), continue; end
    % state at this stage is not stored; use k1 state for g=1 (exact), and
    % report only the k1 stage where state == path(last,:) exactly.
    if g>1, continue; end
    xk=reshape(path(last,s,:),[],1);
    stk=struct('x',xk,'mu',MU(idx,:)','sigma2',0);
    oik=obstacle_cbf_info(stk,cc,tv(idx)); bik=track_boundary_cbf_info(stk,cc,tv(idx));
    jik=joint_safety_softmin_info(oik,bik,stk,cc,tv(idx));
    u=U(idx,:)';
    for r=1:numel(jik.h_values)
      if ~ismember(jik.row_point_indices(r),bad), continue; end
      grad=-jik.A(r,:);                       % A = -grad
      h=jik.h_values(r);
      if h>=0, ph=phi0; else, ph=om/max(1-tv(idx),eps)^2; end
      res=grad*(stk.mu+u)+ph*h;
      sc=norm(grad)*norm(stk.mu+u);
      fprintf('%-8d %-6d %-6d %-9.5f %-12.4g %-12.4g %-12.4g\n', ...
        s, jik.row_point_indices(r), g, tv(idx), h, res, res/max(sc,eps));
      viol(end+1,:)=[s jik.row_point_indices(r) h res]; %#ok<AGROW>
    end
  end
end
if ~isempty(viol)
  fprintf('\nsummary over %d violating rows at the final step (k1 stage):\n', size(viol,1));
  fprintf('  residual r >= -1e-9 (constraint SATISFIED by the QP) : %d\n', sum(viol(:,4)>=-1e-9));
  fprintf('  residual r <  -1e-9 (QP itself did NOT satisfy it)   : %d\n', sum(viol(:,4)< -1e-9));
  fprintf('  median r = %.4g   min r = %.4g\n', median(viol(:,4)), min(viol(:,4)));
end
save(fullfile('outputs','Exp_WhyUncontrolled.mat'),'viol');
end
