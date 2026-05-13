function [grad, y_var] = predict_gp_variance_grad(model, x_test)
x_test = double(x_test);
[y_var, Ktx, solve_term] = predict_gp_variance(model, x_test);
weights = model.L' \ solve_term;
weights = weights';

diff = reshape(x_test, size(x_test, 1), 1, size(x_test, 2)) - ...
    reshape(model.x_train, 1, size(model.x_train, 1), size(model.x_train, 2));
dk_dx = -(diff / (model.gp.length_scale_xy ^ 2)) .* reshape(Ktx, size(Ktx, 1), size(Ktx, 2), 1);
grad = -2.0 * squeeze(sum(dk_dx .* reshape(weights, size(weights, 1), size(weights, 2), 1), 2));
if isvector(grad)
    grad = reshape(grad, 1, []);
end
end
