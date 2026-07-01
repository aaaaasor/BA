function print_uncertainty_sigma_diagnostic(level_label, uncertainty_values)
if isempty(uncertainty_values)
    return;
end
beta = aggregate_variance_values(uncertainty_values);
sigma = sqrt(beta);
disp([level_label, ' diagnose beta=sum sigma_i^2 mean/max: ', ...
    num2str(mean(beta)), ' / ', num2str(max(beta))]);
disp([level_label, ' diagnose sigma=sqrt(beta) mean/max: ', ...
    num2str(mean(sigma)), ' / ', num2str(max(sigma))]);
end
