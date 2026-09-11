function test_uniconflow_paper_project_dataset
%TEST_UNICONFLOW_PAPER_PROJECT_DATASET Integration test on shared track data.
d=uniconflow_paper_dataset_from_project(struct('n_trajectories',2));
assert(size(d.states,1)==4&&size(d.states,2)==101&&size(d.states,3)==2);
assert(size(d.actions,1)==2&&size(d.actions,2)==100&&size(d.actions,3)==2);
assert(d.dynamics_residual_max<1e-10&&strcmp(d.action_bound_source,'section_text'));
assert(all(d.actions>=d.spec.action_lower-1e-12,'all'));
assert(all(d.actions<=d.spec.action_upper+1e-12,'all'));
scene=uniconflow_paper_project_scene();
[h,G]=scene.state_constraint(d.states(:,1,1),0);
assert(isscalar(h)&&isequal(size(G),[1 4])&&all(isfinite([h,G])));
fprintf(['project dataset: PASS dt=%.6g s, RMSE median=%.3f m, ', ...
    'max error=%.3f m, dyn=%.3g\n'],d.dt,median(d.tracking_rmse_m), ...
    max(d.max_tracking_error_m),d.dynamics_residual_max);
end
