function model = fit_gp_model(x_train, y_train, gp)
K = rbf_kernel(x_train, x_train, gp);
L = chol(K + gp.noise_variance * eye(size(K, 1)), 'lower');
alpha = L' \ (L \ y_train(:));

model.x_train = x_train;
model.alpha = alpha;
model.gp = gp;
end
