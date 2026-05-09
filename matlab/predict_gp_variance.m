function y_var = predict_gp_variance(model, x_test)
Ktx = rbf_kernel(x_test, model.x_train, model.gp);
solve_term = model.L \ Ktx';
prior_var = model.gp.signal_variance * ones(size(x_test, 1), 1);
y_var = max(prior_var - sum(solve_term .^ 2, 1)', 0);
end
