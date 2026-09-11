function diag_sample7_fine(s)
if nargin<1, s = 7; end
root='C:\Users\JieJi\BA\matlab'; cd(root);
R = load(fullfile('outputs','Racing_NoVariance_L2T0999_Omega05_SecondLevel_Rollout.mat'), ...
    'saved_segment_rollout_constraint','segment_traj_path_10d','segment_rollout_times');
D = load(fullfile('outputs','Diag_Divergence_Cause.mat'));
cst=R.saved_segment_rollout_constraint; times=R.segment_rollout_times;
omega = struct_field_default(cst,'joint_safety_phi1_omega',NaN);
phi0  = struct_field_default(cst,'joint_safety_phi0',NaN);
phimax= struct_field_default(cst,'joint_safety_phi1_max',inf);
hard_t= struct_field_default(cst,'obstacle_slack_hard_after_time',NaN);
hard_b= struct_field_default(cst,'track_boundary_slack_hard_after_time',NaN);
fprintf('phi0=%g omega=%g phi1_max=%g | obstacle hard_after=%g  boundary hard_after=%g\n', ...
    phi0, omega, phimax, hard_t, hard_b);
fprintf('slack_enabled=%d obstacle_slack=%d boundary_slack=%d\n', ...
    struct_field_default(cst,'slack_enabled',false), ...
    struct_field_default(cst,'obstacle_slack_enabled',false), ...
    struct_field_default(cst,'track_boundary_slack_enabled',false));
fprintf('\nsample %d, steps 84..101\n', s);
fprintf('%-5s %-8s %-9s %-10s %-11s %-11s %-11s %-11s\n', ...
    'k','t','|x|','sigma2','|mu|','h_min','phi(t,h)','req_hdot');
for k = 84:numel(times)
    t = times(k); h = D.hmin(k,s);
    if h >= 0, ph = phi0; else, ph = min(omega/max(1-t,eps)^2, phimax); end
    fprintf('%-5d %-8.4f %-9.4g %-10.4g %-11.4g %-11.4g %-11.4g %-11.4g\n', ...
        k, t, D.xn(k,s), D.s2(k,s), D.mun(k,s), h, ph, -ph*h);
end
end
