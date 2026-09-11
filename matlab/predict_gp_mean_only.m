% GP mean prediction for frozen RK4 sub-stages (k2, k3, k4).
% Variance and its gradient are frozen at k1 together with the QP control.
function mu = predict_gp_mean_only(model, gp_input)
output_models = model.output_models;
y_dim = numel(output_models);
mu = zeros(y_dim,1);
for output_idx = 1:y_dim
	mu(output_idx) = output_models{output_idx}.predict_mean(gp_input);
end
end
