function report = diagnose_variance_gradient_consistency(model, t, x, v)
%DIAGNOSE_VARIANCE_GRADIENT_CONSISTENCY Compare analytic and true FD beta gradients.
% The analytic LoG-GP gradient is the one used by the variance QP.  The
% central-difference gradient differentiates the complete public
% predict_variance result, including any state-dependent expert routing.

x = x(:);
gp_input = [t; x];
stats = predict_gp_stats(model, gp_input);
analytic = [stats.sigma2_t, stats.sigma2_grad_x];
fd = zeros(size(analytic));
for dim_idx = 1:numel(gp_input)
    step = 1e-5 * max(1.0, abs(gp_input(dim_idx)));
    z_plus = gp_input;
    z_minus = gp_input;
    z_plus(dim_idx) = z_plus(dim_idx) + step;
    z_minus(dim_idx) = z_minus(dim_idx) - step;
    fd(dim_idx) = (summed_variance(model, z_plus) - ...
        summed_variance(model, z_minus)) / (2.0 * step);
end

analytic_norm = norm(analytic);
fd_norm = norm(fd);
relative_error = norm(analytic - fd) / max(fd_norm, eps);
cosine_similarity = dot(analytic, fd) / ...
    max(analytic_norm * fd_norm, eps);
flow_direction = [1; v(:)];
analytic_total_derivative = analytic * flow_direction;
fd_total_derivative = fd * flow_direction;

report = struct( ...
    'analytic_gradient', analytic, ...
    'finite_difference_gradient', fd, ...
    'analytic_norm', analytic_norm, ...
    'finite_difference_norm', fd_norm, ...
    'relative_error', relative_error, ...
    'cosine_similarity', cosine_similarity, ...
    'analytic_total_derivative', analytic_total_derivative, ...
    'finite_difference_total_derivative', fd_total_derivative);

fprintf(['Variance-gradient consistency at t=%.6g: analytic norm=%.6g, ', ...
    'FD norm=%.6g, relative error=%.6g, cosine=%.6g, ', ...
    'd beta/dt analytic=%.6g, FD=%.6g.\n'], t, analytic_norm, ...
    fd_norm, relative_error, cosine_similarity, ...
    analytic_total_derivative, fd_total_derivative);
end

function beta = summed_variance(model, gp_input)
beta = 0.0;
for output_idx = 1:numel(model.output_models)
    beta = beta + model.output_models{output_idx}.predict_variance(gp_input);
end
end
