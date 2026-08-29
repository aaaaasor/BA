function [u, exitflag, slack, residuals, iterations, seconds, ...
    row_contributions, equality_contribution, objective_contribution] = ...
    solve_control_least_squares_qp(A,b,types,cfg,t,stats,terminal_info, ...
    integral_residual,terminal_residual,G,e)
% min 0.5*||W*u||^2 + 0.5*||G*u+e||^2 + existing slack cost.
% Exact completion of the square reuses the existing constrained solver,
% including its closed-form backend, slack semantics and feasibility checks.
% H=R'*R; u=center+R\z turns the objective into 0.5*||z||^2+constant.
% Thus its zero-control shortcut now tests the TRUE objective minimizer.
timer=tic;
n=numel(stats.mu);
weights=struct_field_default(cfg,'control_weight',ones(n,1));
weights=weights(:);e=e(:);
validateattributes(weights,{'numeric'},{'numel',n,'real','finite','positive'});
if size(G,2)~=n || size(G,1)~=numel(e) || ...
        any(~isfinite(G),'all') || any(~isfinite(e))
    error('Invalid control least-squares objective dimensions/values.');
end
H=diag(weights.^2)+G'*G;
R=chol((H+H')/2);
center=-(R\(R'\(G'*e)));
transformed=cfg;
transformed.control_weight=ones(n,1);
E=struct_field_default(cfg,'endpoint_hold_velocity_matrix',[]);
if ~isempty(E)
    transformed.endpoint_hold_velocity_matrix=E/R;
end
transformed_stats=stats;
transformed_stats.mu=R*(stats.mu(:)+center);
[z,exitflag,slack,residuals,iterations,~,parts,eq_part]=solve_slack_qp( ...
    A/R,b(:)-A*center,types,transformed,t,transformed_stats, ...
    terminal_info,integral_residual,terminal_residual);
u=center+R\z;
row_contributions=(R\parts')';
equality_contribution=R\eq_part;
% This is objective-driven control, not an obstacle/equality multiplier.
objective_contribution=center;
seconds=toc(timer);
end
