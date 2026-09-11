function m = uniconflow_metrics(r, tol)
%UNICONFLOW_METRICS  Paper-style certification metrics for adapted pipeline.
if nargin<2 || isempty(tol), tol=1e-8; end
base=safeflow_nn_metrics(r);
n=r.n_gen;
if isfield(r,'violations_after')
    state_ok=r.violations_after(:)==0;
else
    state_ok=false(n,1);
end
if isfield(r,'action_violations_after')
    action_ok=r.action_violations_after(:)==0;
else
    st=r.options.stage2;
    action_ok=all(r.nu>=st.nu_min-tol & r.nu<=st.nu_max+tol,1).' & ...
        all(abs(r.kappa)<=st.kappa_max+tol,1).';
end
kc_f=sqrt(max(r.g_final(:),0)/max(size(r.points,1)-1,1));
dyn_ok=kc_f<=tol;
m=base;
m.sr_s=mean(state_ok);
% The returned states are forward rollouts of the returned actions, hence the
% action-rollout safety rate equals state safety for the final trajectory.
m.sr_a=m.sr_s;
m.ar=mean(action_ok);
m.tsr=mean(state_ok & action_ok & dyn_ok);
m.kc_f=mean(kc_f);
m.kc_f_per=kc_f;
m.certified=state_ok & action_ok & dyn_ok;
end
